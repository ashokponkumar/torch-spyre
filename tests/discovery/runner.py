"""
Worker pod test runner — executes a single test file with segfault protection.

Runs inside each worker pod of the Indexed Job.
Reads JOB_COMPLETION_INDEX to determine which file to run.

Three-layer segfault protection:
  1. pytest-forked: forks each test, parent survives SIGSEGV
  2. subprocess wrapper: catches file-level crashes (exit code 139)
  3. K8s restartPolicy: Never — don't restart crashed pods
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from discovery.constants import (
    DEFAULT_HEARTBEAT_INTERVAL,
    DEFAULT_PER_TEST_TIMEOUT,
    DISCOVERY_BASE_DIR,
    PVC_MOUNT_PATH,
    SPYRE_ENV_VARS,
)
from discovery.manifest import load_manifest


def get_run_dir(run_id: str) -> Path:
    return Path(PVC_MOUNT_PATH) / DISCOVERY_BASE_DIR / run_id


def get_worker_dir(run_id: str, index: int) -> Path:
    return get_run_dir(run_id) / "workers" / str(index)


class HeartbeatWriter:
    """Writes periodic heartbeat files to PVC so orchestrator can detect stuck pods."""

    def __init__(self, worker_dir: Path, interval: int = DEFAULT_HEARTBEAT_INTERVAL):
        self.worker_dir = worker_dir
        self.interval = interval
        self.current_test = "initializing"
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def update(self, test_name: str) -> None:
        self.current_test = test_name

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._write()
            self._stop.wait(self.interval)

    def _write(self) -> None:
        hb = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "current_test": self.current_test,
            "pid": os.getpid(),
        }
        hb_path = self.worker_dir / "heartbeat.json"
        try:
            with open(hb_path, "w") as f:
                json.dump(hb, f)
        except OSError:
            pass  # PVC write failure — not fatal


def setup_env(pytorch_root: str) -> Dict[str, str]:
    """Set up environment variables for running upstream tests on Spyre."""
    env = os.environ.copy()
    env.update(SPYRE_ENV_VARS)
    env["PYTORCH_ROOT"] = pytorch_root
    # Ensure pytest can find the test helpers
    env["PYTHONPATH"] = f"{pytorch_root}/test:{env.get('PYTHONPATH', '')}"
    return env


def collect_tests(
    test_file: str, pytorch_root: str, env: Dict[str, str], worker_dir: Path
) -> Optional[str]:
    """Run pytest --collect-only to discover test IDs. Returns collected.txt path or None."""
    collected_path = worker_dir / "collected.txt"
    test_path = os.path.join(pytorch_root, "test", test_file)

    cmd = [
        sys.executable, "-m", "pytest",
        "--collect-only", "-q",
        test_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
            cwd=os.path.join(pytorch_root, "test"),
        )
        with open(collected_path, "w") as f:
            f.write(result.stdout)
        return str(collected_path)
    except subprocess.TimeoutExpired:
        with open(collected_path, "w") as f:
            f.write("# collection timed out\n")
        return str(collected_path)
    except Exception as e:
        with open(collected_path, "w") as f:
            f.write(f"# collection error: {e}\n")
        return str(collected_path)


def run_tests(
    test_file: str,
    pytorch_root: str,
    env: Dict[str, str],
    worker_dir: Path,
    test_timeout: int = DEFAULT_PER_TEST_TIMEOUT,
    heartbeat: Optional[HeartbeatWriter] = None,
) -> Dict[str, Any]:
    """Run a single test file with subprocess isolation.

    Returns a summary dict with pass/fail counts and exit code.
    """
    test_path = os.path.join(pytorch_root, "test", test_file)
    results_xml = worker_dir / "results.xml"
    stdout_log = worker_dir / "stdout.log"
    stderr_log = worker_dir / "stderr.log"
    exitcode_file = worker_dir / "exitcode.txt"

    cmd = [
        sys.executable, "-m", "pytest",
        test_path,
        "--forked",
        f"--junitxml={results_xml}",
        f"--timeout={test_timeout}",
        "-v",
        "--tb=short",
        "-x",  # stop on first failure within forked — keeps run time reasonable
    ]

    if heartbeat:
        heartbeat.update(f"running: {test_file}")

    summary: Dict[str, Any] = {
        "file": test_file,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "exit_code": None,
        "signal": None,
        "total": 0,
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "duration": 0.0,
    }

    try:
        with open(stdout_log, "w") as out_f, open(stderr_log, "w") as err_f:
            # Use per-pod timeout (4h default) as the outer subprocess timeout
            pod_timeout = int(os.environ.get("POD_TIMEOUT", "14400"))
            result = subprocess.run(
                cmd,
                stdout=out_f,
                stderr=err_f,
                env=env,
                timeout=pod_timeout,
                cwd=os.path.join(pytorch_root, "test"),
            )
            summary["exit_code"] = result.returncode

    except subprocess.TimeoutExpired:
        summary["exit_code"] = -1
        summary["signal"] = "TIMEOUT"
    except Exception as e:
        summary["exit_code"] = -2
        summary["signal"] = str(e)

    # Detect segfault
    if summary["exit_code"] == 139 or summary["exit_code"] == -signal.SIGSEGV:
        summary["signal"] = "SIGSEGV"

    # Detect OOM kill
    if summary["exit_code"] == 137:
        summary["signal"] = "OOM_KILLED"

    summary["end_time"] = datetime.now(timezone.utc).isoformat()

    # Write exit code file
    with open(exitcode_file, "w") as f:
        f.write(str(summary["exit_code"]))

    # Parse JUnit XML if it exists for summary counts
    if results_xml.exists():
        try:
            summary.update(_parse_junit_summary(results_xml))
        except Exception:
            pass  # XML may be truncated if process crashed

    return summary


def _parse_junit_summary(xml_path: Path) -> Dict[str, Any]:
    """Quick parse of JUnit XML for top-level counts."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Handle both <testsuites> and <testsuite> root elements
    if root.tag == "testsuites":
        suites = root.findall("testsuite")
    else:
        suites = [root]

    total = passed = failed = errors = skipped = 0
    duration = 0.0

    for suite in suites:
        total += int(suite.get("tests", 0))
        failed += int(suite.get("failures", 0))
        errors += int(suite.get("errors", 0))
        skipped += int(suite.get("skipped", 0))
        duration += float(suite.get("time", 0))

    passed = total - failed - errors - skipped

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "duration": duration,
    }


def main() -> None:
    """Entry point for worker pods."""
    import argparse

    parser = argparse.ArgumentParser(description="Discovery worker runner")
    parser.add_argument(
        "--index", type=int,
        default=int(os.environ.get("JOB_COMPLETION_INDEX", "0")),
        help="Job completion index (default: $JOB_COMPLETION_INDEX)",
    )
    parser.add_argument("--run-id", required=True, help="Discovery run ID")
    parser.add_argument(
        "--pytorch-root",
        default=os.environ.get("PYTORCH_ROOT", "/workspace/pytorch"),
        help="Path to PyTorch source",
    )
    parser.add_argument(
        "--manifest",
        help="Path to file_manifest.json (default: auto from PVC)",
    )
    args = parser.parse_args()

    run_dir = get_run_dir(args.run_id)
    worker_dir = get_worker_dir(args.run_id, args.index)
    worker_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    manifest_path = args.manifest or str(run_dir / "config" / "file_manifest.json")
    manifest = load_manifest(Path(manifest_path))

    if args.index >= len(manifest):
        print(f"Index {args.index} out of range (manifest has {len(manifest)} entries)")
        sys.exit(1)

    entry = manifest[args.index]
    test_file = entry["file"]
    test_timeout = entry.get("test_timeout", DEFAULT_PER_TEST_TIMEOUT)

    print(f"[Worker {args.index}] Running {test_file}")

    # Setup env
    env = setup_env(args.pytorch_root)

    # Start heartbeat
    heartbeat = HeartbeatWriter(worker_dir)
    heartbeat.start()

    try:
        # Phase 1: Collect test IDs
        heartbeat.update(f"collecting: {test_file}")
        collect_tests(test_file, args.pytorch_root, env, worker_dir)

        # Phase 2: Run tests
        summary = run_tests(
            test_file, args.pytorch_root, env, worker_dir,
            test_timeout=test_timeout, heartbeat=heartbeat,
        )

        # Write summary
        with open(worker_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Write done marker (LAST — signals completion)
        (worker_dir / "done.marker").touch()

        print(f"[Worker {args.index}] Done: {test_file} — "
              f"passed={summary['passed']}, failed={summary['failed']}, "
              f"errors={summary['errors']}, skipped={summary['skipped']}")

    finally:
        heartbeat.stop()

    sys.exit(0)


if __name__ == "__main__":
    main()
