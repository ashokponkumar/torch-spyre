#!/bin/bash
# Discovery worker runner — self-contained bash script
# Reads JOB_COMPLETION_INDEX, looks up test file, runs pytest, writes results to PVC.
#
# Expected environment:
#   JOB_COMPLETION_INDEX  — set by K8s Indexed Job (0-based)
#   RUN_ID                — discovery run identifier
#   PYTORCH_ROOT          — path to pytorch clone (default: /workspace/pytorch)
#   TORCH_SPYRE_ROOT      — path to torch-spyre clone (default: /workspace/torch-spyre)
#   PER_TEST_TIMEOUT      — per-test timeout in seconds (default: 300)
#   POD_TIMEOUT           — overall pod timeout in seconds (default: 14400)
#
# Mounted at /mnt/config/:
#   file_list.txt         — one test file per line (63 lines)
#   discovery_config.yaml — test suite config for mandatory_success mode

set -euo pipefail

INDEX="${JOB_COMPLETION_INDEX:?JOB_COMPLETION_INDEX not set}"
RUN_ID="${RUN_ID:?RUN_ID not set}"
PYTORCH_ROOT="${PYTORCH_ROOT:-/workspace/pytorch}"
TORCH_SPYRE_ROOT="${TORCH_SPYRE_ROOT:-/workspace/torch-spyre}"
PER_TEST_TIMEOUT="${PER_TEST_TIMEOUT:-300}"
POD_TIMEOUT="${POD_TIMEOUT:-14400}"

# Results directory on PVC
RESULTS_DIR="/mnt/devwork/discovery/${RUN_ID}/workers/${INDEX}"
mkdir -p "${RESULTS_DIR}"

# Look up the test file for this index
FILE_LIST="/mnt/config/file_list.txt"
TEST_FILE=$(sed -n "$((INDEX + 1))p" "${FILE_LIST}")

if [ -z "${TEST_FILE}" ]; then
    echo "[Worker ${INDEX}] ERROR: No test file at index ${INDEX}"
    echo "no_file" > "${RESULTS_DIR}/exitcode.txt"
    touch "${RESULTS_DIR}/done.marker"
    exit 1
fi

echo "[Worker ${INDEX}] Starting: ${TEST_FILE}"
echo "${TEST_FILE}" > "${RESULTS_DIR}/test_file.txt"

# Record start time
date -u +"%Y-%m-%dT%H:%M:%S+00:00" > "${RESULTS_DIR}/start_time.txt"

# Set up environment for Spyre upstream tests
export FLEX_COMPUTE=SENTIENT
export FLEX_DEVICE=PF
export TOKENIZERS_PARALLELISM=false
export PYTORCH_TESTING_DEVICE_ONLY_FOR=privateuse1
export TORCH_TEST_DEVICES="${TORCH_SPYRE_ROOT}/tests/spyre_test_base_common.py"
export PYTORCH_TEST_CONFIG="/mnt/config/discovery_config.yaml"
export PYTHONPATH="${TORCH_SPYRE_ROOT}/tests:${PYTORCH_ROOT}/test:${PYTHONPATH:-}"

TEST_PATH="${PYTORCH_ROOT}/test/${TEST_FILE}"

# Phase 1: Collect tests (quick, for reference)
echo "[Worker ${INDEX}] Phase 1: Collecting tests..."
timeout 120 python3 -m pytest --collect-only -q "${TEST_PATH}" \
    > "${RESULTS_DIR}/collected.txt" 2>&1 || true

COLLECTED_COUNT=$(grep -c '::' "${RESULTS_DIR}/collected.txt" 2>/dev/null || echo "0")
echo "[Worker ${INDEX}] Collected ${COLLECTED_COUNT} tests"

# Phase 2: Run tests with pytest-forked for segfault isolation
echo "[Worker ${INDEX}] Phase 2: Running tests..."

# Run pytest in a subshell to capture exit code even on signals
set +e
timeout "${POD_TIMEOUT}" python3 -m pytest \
    "${TEST_PATH}" \
    --forked \
    --junitxml="${RESULTS_DIR}/results.xml" \
    --timeout="${PER_TEST_TIMEOUT}" \
    -v \
    --tb=short \
    > "${RESULTS_DIR}/stdout.log" 2> "${RESULTS_DIR}/stderr.log"
EXIT_CODE=$?
set -e

echo "${EXIT_CODE}" > "${RESULTS_DIR}/exitcode.txt"

# Detect signal-based exits
SIGNAL=""
if [ "${EXIT_CODE}" -eq 139 ]; then
    SIGNAL="SIGSEGV"
elif [ "${EXIT_CODE}" -eq 137 ]; then
    SIGNAL="OOM_KILLED"
elif [ "${EXIT_CODE}" -eq 124 ]; then
    SIGNAL="TIMEOUT"
fi

# Phase 3: Build summary JSON
echo "[Worker ${INDEX}] Phase 3: Writing summary..."

# Parse JUnit XML for counts if it exists
TOTAL=0 PASSED=0 FAILED=0 ERRORS=0 SKIPPED=0 DURATION="0.0"
if [ -f "${RESULTS_DIR}/results.xml" ]; then
    # Extract from testsuite attributes
    TOTAL=$(grep -oP 'tests="\K[0-9]+' "${RESULTS_DIR}/results.xml" | head -1 || echo 0)
    FAILED=$(grep -oP 'failures="\K[0-9]+' "${RESULTS_DIR}/results.xml" | head -1 || echo 0)
    ERRORS=$(grep -oP 'errors="\K[0-9]+' "${RESULTS_DIR}/results.xml" | head -1 || echo 0)
    SKIPPED=$(grep -oP 'skipped="\K[0-9]+' "${RESULTS_DIR}/results.xml" | head -1 || echo 0)
    DURATION=$(grep -oP 'time="\K[0-9.]+' "${RESULTS_DIR}/results.xml" | head -1 || echo "0.0")
    PASSED=$((TOTAL - FAILED - ERRORS - SKIPPED))
    [ "${PASSED}" -lt 0 ] && PASSED=0
fi

END_TIME=$(date -u +"%Y-%m-%dT%H:%M:%S+00:00")
START_TIME=$(cat "${RESULTS_DIR}/start_time.txt")

cat > "${RESULTS_DIR}/summary.json" << SUMMARY_EOF
{
  "file": "${TEST_FILE}",
  "index": ${INDEX},
  "start_time": "${START_TIME}",
  "end_time": "${END_TIME}",
  "exit_code": ${EXIT_CODE},
  "signal": ${SIGNAL:+"\"${SIGNAL}\""}${SIGNAL:-null},
  "total": ${TOTAL},
  "passed": ${PASSED},
  "failed": ${FAILED},
  "errors": ${ERRORS},
  "skipped": ${SKIPPED},
  "duration": ${DURATION}
}
SUMMARY_EOF

# Done marker (LAST — signals completion)
touch "${RESULTS_DIR}/done.marker"

echo "[Worker ${INDEX}] Done: ${TEST_FILE} — exit=${EXIT_CODE} passed=${PASSED} failed=${FAILED} errors=${ERRORS} skipped=${SKIPPED}"
