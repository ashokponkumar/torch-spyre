# Upstream Test Discovery System for Spyre

Automated system to run all 63 PyTorch upstream test files (using `instantiate_device_type_tests()`) on Spyre cards, with subprocess isolation, cluster parallelization, and comprehensive reporting.

## Quick Start

```bash
# Dry run — generate YAML without applying
make dry-run

# Single file test
python -m discovery launch-orchestrator --files test_complex.py --parallelism 1

# Small parallel test
python -m discovery launch-orchestrator --files test_complex.py,test_dlpack.py,test_indexing.py --parallelism 3

# Full run (63 files, 40 pods)
make discover

# Monitor
make status
make logs

# Fetch results when done
make fetch

# Compare against previous run
make diff BASELINE=2026-03-26-1430
```

## Architecture

```
LOCAL MACHINE → launch-orchestrator → ORCHESTRATOR POD → INDEXED JOB (63 workers)
                                           ↓
                                     Aggregate results
                                     Generate config
                                     Diff vs baseline
                                           ↓
LOCAL MACHINE ← fetch-results ← PVC artifacts
```

The orchestrator pod runs on a regular worker node (no Spyre card). Worker pods each get 1 Spyre card and run a single test file with 3-layer segfault protection.

## Commands

| Command | Description |
|---------|-------------|
| `launch-orchestrator` | Start a discovery run (creates orchestrator pod) |
| `status` | Check orchestrator + job progress |
| `logs` | Tail orchestrator pod logs |
| `fetch-results` | Copy results from PVC to local machine |
| `diff` | Compare two completed runs |
| `resume` | Resume a failed/interrupted run |
| `clean` | Delete K8s resources for a run |
| `list-runs` | List all discovery runs on the PVC |

## Outputs

All outputs are written to the PVC at `/mnt/devwork/discovery/<run-id>/`:

- `discovery_results.json` — unified test results
- `test_suite_config_daily.yaml` — generated CI config
- `diff_report.md` — regression/fix report (if baseline provided)
- `orchestrator.log` — full orchestrator log
- `status.json` — final run status
- `workers/<index>/` — per-worker results, logs, and heartbeats

## Dynamic Scaling

The orchestrator auto-scales parallelism based on cluster capacity:
- Probes available Spyre cards every 60s
- Scales up when cards free up (e.g., someone deletes their dev pod)
- Scales down when cluster gets busy
- Reserves cards for other users (`--reserve N`, default 5)

## Resumability

If the orchestrator pod dies or you disconnect:
```bash
python -m discovery resume --run-id <id>
```
This creates a new orchestrator pod that reads state from the PVC and resumes.
