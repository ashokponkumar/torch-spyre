"""
Orchestrator — runs inside the orchestrator pod on the cluster.

Manages the full discovery pipeline:
  1. Generate artifacts (config, manifest)
  2. Create Indexed Job for workers
  3. Monitor Job with dynamic scaling
  4. Aggregate results
  5. Generate daily CI config
  6. Diff against baseline (optional)
  7. Write final status
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader

from discovery.aggregator import aggregate_results, write_results
from discovery.config_generator import generate_config, write_config
from discovery.constants import (
    DEFAULT_BACKOFF_LIMIT,
    DEFAULT_IMAGE,
    DEFAULT_IMAGE_PULL_SECRET,
    DEFAULT_MAX_PARALLELISM,
    DEFAULT_MIN_PARALLELISM,
    DEFAULT_NAMESPACE,
    DEFAULT_PER_POD_TIMEOUT,
    DEFAULT_POLL_INTERVAL,
    DEFAULT_PVC,
    DEFAULT_RESERVE_CARDS,
    DEFAULT_SCHEDULER,
    DEFAULT_SERVICE_ACCOUNT,
    DEFAULT_WORKER_MEMORY_LIMIT,
    DEFAULT_WORKER_MEMORY_REQUEST,
    DISCOVERY_BASE_DIR,
    PHASES,
    PVC_MOUNT_PATH,
    UPSTREAM_TEST_FILES,
)
from discovery.differ import diff_results, generate_diff_report
from discovery.k8s_manager import (
    apply_yaml,
    delete_resource,
    force_delete_pod,
    get_job_progress,
    get_pod_list,
    get_pod_phases,
    get_spyre_capacity,
    is_job_complete,
    is_job_failed,
    patch_job_parallelism,
)
from discovery.manifest import (
    build_discovery_config,
    build_file_manifest,
    write_discovery_config,
    write_manifest,
)

logger = logging.getLogger("discovery.orchestrator")


class Orchestrator:
    """Manages the full discovery run from within the orchestrator pod."""

    def __init__(
        self,
        run_id: str,
        namespace: str = DEFAULT_NAMESPACE,
        pvc: str = DEFAULT_PVC,
        image: str = DEFAULT_IMAGE,
        max_parallelism: int = DEFAULT_MAX_PARALLELISM,
        min_parallelism: int = DEFAULT_MIN_PARALLELISM,
        reserve_cards: int = DEFAULT_RESERVE_CARDS,
        per_pod_timeout: int = DEFAULT_PER_POD_TIMEOUT,
        pytorch_root: str = "/workspace/pytorch",
        baseline: Optional[str] = None,
        files: Optional[List[str]] = None,
    ):
        self.run_id = run_id
        self.namespace = namespace
        self.pvc = pvc
        self.image = image
        self.max_parallelism = max_parallelism
        self.min_parallelism = min_parallelism
        self.reserve_cards = reserve_cards
        self.per_pod_timeout = per_pod_timeout
        self.pytorch_root = pytorch_root
        self.baseline = baseline
        self.files = files or list(UPSTREAM_TEST_FILES)

        self.run_dir = Path(PVC_MOUNT_PATH) / DISCOVERY_BASE_DIR / run_id
        self.state_path = self.run_dir / "state.json"
        self.job_name = f"discovery-workers-{run_id}"

        self._template_env = Environment(
            loader=FileSystemLoader(
                str(Path(__file__).parent / "templates")
            ),
            keep_trailing_newline=True,
        )

    def run(self) -> None:
        """Execute the full orchestrator pipeline."""
        self.run_dir.mkdir(parents=True, exist_ok=True)

        state = self._load_state()
        current_phase = state.get("current_phase", "init")

        logger.info(f"Starting orchestrator for run {self.run_id}, resuming from phase: {current_phase}")

        try:
            phase_idx = PHASES.index(current_phase)
        except ValueError:
            phase_idx = 0

        phases = [
            ("init", self._phase_init),
            ("generate_artifacts", self._phase_generate_artifacts),
            ("create_job", self._phase_create_job),
            ("monitor_job", self._phase_monitor_job),
            ("aggregate", self._phase_aggregate),
            ("generate_config", self._phase_generate_config),
            ("diff", self._phase_diff),
            ("finalize", self._phase_finalize),
        ]

        for i, (name, func) in enumerate(phases):
            if i < phase_idx:
                continue

            logger.info(f"=== Phase: {name} ===")
            self._save_state({"current_phase": name})

            try:
                func()
            except Exception as e:
                logger.error(f"Phase {name} failed: {e}", exc_info=True)
                self._write_status("failed", error=str(e))
                raise

        logger.info(f"Orchestrator complete for run {self.run_id}")

    def _phase_init(self) -> None:
        """Initialize run directory structure."""
        (self.run_dir / "config").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "workers").mkdir(parents=True, exist_ok=True)
        self._write_status("running")

    def _phase_generate_artifacts(self) -> None:
        """Generate discovery_config.yaml and file_manifest.json."""
        manifest = build_file_manifest(self.files)
        write_manifest(manifest, self.run_dir / "config" / "file_manifest.json")

        config = build_discovery_config(
            run_id=self.run_id,
            repo="",  # filled by orchestrator pod env
            branch="",
            pytorch_repo="",
            pytorch_branch="",
            max_parallelism=self.max_parallelism,
            min_parallelism=self.min_parallelism,
            reserve_cards=self.reserve_cards,
            per_pod_timeout=self.per_pod_timeout,
            namespace=self.namespace,
            pvc=self.pvc,
            image=self.image,
            baseline=self.baseline,
        )
        write_discovery_config(config, self.run_dir / "config" / "discovery_config.yaml")

        # Save run params for reproducibility
        params = {
            "run_id": self.run_id,
            "namespace": self.namespace,
            "pvc": self.pvc,
            "image": self.image,
            "max_parallelism": self.max_parallelism,
            "files": self.files,
            "baseline": self.baseline,
        }
        with open(self.run_dir / "config" / "run_params.json", "w") as f:
            json.dump(params, f, indent=2)

        logger.info(f"Generated manifest for {len(manifest)} files")

    def _phase_create_job(self) -> None:
        """Create the Indexed Job for worker pods."""
        # Check which indices are already complete (for resume)
        complete_indices = self._get_complete_indices()
        remaining = len(self.files) - len(complete_indices)

        if remaining == 0:
            logger.info("All indices already complete, skipping job creation")
            return

        # Probe initial capacity
        capacity = get_spyre_capacity(self.namespace)
        available = max(0, capacity["allocatable"] - self.reserve_cards)
        initial_parallelism = max(
            self.min_parallelism,
            min(self.max_parallelism, available, remaining),
        )

        logger.info(
            f"Capacity: {capacity['total']} total, {capacity['used_by_others']} used, "
            f"{self.reserve_cards} reserved → {available} available. "
            f"Initial parallelism: {initial_parallelism}"
        )

        # Render Job template
        template = self._template_env.get_template("discovery-job.yaml.j2")
        job_yaml = template.render(
            run_id=self.run_id,
            namespace=self.namespace,
            num_files=len(self.files),
            parallelism=initial_parallelism,
            backoff_limit=DEFAULT_BACKOFF_LIMIT,
            scheduler=DEFAULT_SCHEDULER,
            service_account=DEFAULT_SERVICE_ACCOUNT,
            image=self.image,
            image_pull_secret=DEFAULT_IMAGE_PULL_SECRET,
            pvc=self.pvc,
            pvc_mount_path=PVC_MOUNT_PATH,
            per_pod_timeout=self.per_pod_timeout,
            default_memory_request=DEFAULT_WORKER_MEMORY_REQUEST,
            default_memory_limit=DEFAULT_WORKER_MEMORY_LIMIT,
            torch_spyre_repo=DEFAULT_NAMESPACE,  # placeholder, set by CLI
            torch_spyre_branch="main",
            pytorch_repo="https://github.com/pytorch/pytorch.git",
            pytorch_branch="main",
        )

        apply_yaml(job_yaml, self.namespace)
        logger.info(f"Created Job {self.job_name}")

        self._save_state({
            "current_phase": "create_job",
            "job_name": self.job_name,
            "initial_parallelism": initial_parallelism,
        })

    def _phase_monitor_job(self) -> None:
        """Monitor Job until completion with dynamic scaling."""
        label_selector = f"discovery-run={self.run_id}"
        current_parallelism = self.max_parallelism
        scaling_check_interval = 60  # seconds
        last_scale_check = 0

        while True:
            # Check completion
            if is_job_complete(self.job_name, self.namespace):
                logger.info("Job completed successfully")
                break

            if is_job_failed(self.job_name, self.namespace):
                logger.warning("Job failed — proceeding with partial results")
                break

            # Progress report
            progress = get_job_progress(self.job_name, self.namespace)
            phases = get_pod_phases(label_selector, self.namespace)
            complete_indices = self._get_complete_indices()

            logger.info(
                f"Progress: {progress['succeeded']}/{progress['completions']} succeeded, "
                f"{progress['active']} active, {progress['failed']} failed | "
                f"PVC done markers: {len(complete_indices)} | "
                f"Pods: {phases}"
            )

            # Dynamic scaling check (every 60s)
            now = time.time()
            if now - last_scale_check >= scaling_check_interval:
                last_scale_check = now
                self._dynamic_scale(progress, phases, label_selector)

            # Check for stuck pods (stale heartbeats)
            self._check_stuck_pods(label_selector)

            time.sleep(DEFAULT_POLL_INTERVAL)

    def _dynamic_scale(
        self,
        progress: Dict[str, int],
        phases: Dict[str, int],
        label_selector: str,
    ) -> None:
        """Adjust Job parallelism based on cluster capacity."""
        capacity = get_spyre_capacity(self.namespace)
        available = max(0, capacity["allocatable"] - self.reserve_cards)
        remaining = progress["completions"] - progress["succeeded"]

        desired = max(
            self.min_parallelism,
            min(self.max_parallelism, available, remaining),
        )

        # Handle pods stuck in Pending >10 min
        pending = phases.get("Pending", 0)
        running = phases.get("Running", 0)
        if pending > 0 and running > 0:
            # Scale down to what's actually running
            desired = min(desired, running + 2)

        current = progress.get("active", 0)
        if desired != current and desired > 0:
            try:
                patch_job_parallelism(self.job_name, desired, self.namespace)
                logger.info(
                    f"Scaling: parallelism {current} → {desired} "
                    f"(capacity: {capacity['total']} total, {capacity['used_by_others']} used, "
                    f"{self.reserve_cards} reserved → {available} available)"
                )
            except Exception as e:
                logger.warning(f"Failed to scale: {e}")

    def _check_stuck_pods(self, label_selector: str) -> None:
        """Check for worker pods with stale heartbeats and force-delete them."""
        from discovery.constants import DEFAULT_HEARTBEAT_STALE_THRESHOLD

        workers_dir = self.run_dir / "workers"
        now = datetime.now(timezone.utc)

        pods = get_pod_list(label_selector, self.namespace)
        for pod in pods:
            phase = pod.get("status", {}).get("phase", "")
            if phase != "Running":
                continue

            pod_name = pod["metadata"]["name"]
            # Try to find index from pod name or annotations
            index_str = pod["metadata"].get("annotations", {}).get(
                "batch.kubernetes.io/job-completion-index", ""
            )
            if not index_str:
                continue

            hb_path = workers_dir / index_str / "heartbeat.json"
            if not hb_path.exists():
                continue

            try:
                with open(hb_path) as f:
                    hb = json.load(f)
                ts = datetime.fromisoformat(hb["timestamp"])
                age = (now - ts).total_seconds()

                if age > DEFAULT_HEARTBEAT_STALE_THRESHOLD:
                    logger.warning(
                        f"Stuck pod {pod_name} (index {index_str}): "
                        f"heartbeat stale for {age:.0f}s, force-deleting"
                    )
                    force_delete_pod(pod_name, self.namespace)
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

    def _phase_aggregate(self) -> None:
        """Aggregate worker results into discovery_results.json."""
        results = aggregate_results(self.run_dir, len(self.files))

        # Add run metadata
        results["metadata"]["run_id"] = self.run_id
        results["metadata"]["total_files"] = len(self.files)

        write_results(results, self.run_dir / "discovery_results.json")

        s = results["summary"]
        logger.info(
            f"Aggregated: {s['total_tests']} tests, "
            f"{s['passed']} passed, {s['failed']} failed, "
            f"{s['errors']} errors, {s['skipped']} skipped, "
            f"{s['incomplete']} incomplete files"
        )

    def _phase_generate_config(self) -> None:
        """Generate test_suite_config_daily.yaml from results."""
        from discovery.aggregator import load_results

        results = load_results(self.run_dir / "discovery_results.json")
        config = generate_config(results, partial=True)
        write_config(config, self.run_dir / "test_suite_config_daily.yaml")
        logger.info("Generated test_suite_config_daily.yaml")

    def _phase_diff(self) -> None:
        """Diff against baseline if provided."""
        if not self.baseline:
            logger.info("No baseline provided, skipping diff")
            return

        baseline_dir = Path(PVC_MOUNT_PATH) / DISCOVERY_BASE_DIR / self.baseline
        baseline_results_path = baseline_dir / "discovery_results.json"

        if not baseline_results_path.exists():
            logger.warning(f"Baseline results not found at {baseline_results_path}")
            return

        from discovery.aggregator import load_results

        new_results = load_results(self.run_dir / "discovery_results.json")
        base_results = load_results(baseline_results_path)

        diff = diff_results(new_results, base_results)
        report = generate_diff_report(diff)

        with open(self.run_dir / "diff_report.md", "w") as f:
            f.write(report)

        with open(self.run_dir / "diff.json", "w") as f:
            json.dump(diff, f, indent=2)

        s = diff["summary"]
        logger.info(
            f"Diff vs {self.baseline}: "
            f"{s['regressions']} regressions, {s['fixes']} fixes, "
            f"{s['new_tests']} new, {s['removed_tests']} removed"
        )

    def _phase_finalize(self) -> None:
        """Write final status."""
        from discovery.aggregator import load_results

        results = load_results(self.run_dir / "discovery_results.json")
        incomplete = results["summary"].get("incomplete", 0)

        if incomplete == 0:
            self._write_status("completed")
        elif incomplete < len(self.files):
            self._write_status("partial")
        else:
            self._write_status("failed")

    def _get_complete_indices(self) -> List[int]:
        """Return list of indices that have a done.marker on PVC."""
        workers_dir = self.run_dir / "workers"
        complete = []
        for idx in range(len(self.files)):
            if (workers_dir / str(idx) / "done.marker").exists():
                complete.append(idx)
        return complete

    def _load_state(self) -> Dict[str, Any]:
        """Load state.json from PVC."""
        if self.state_path.exists():
            with open(self.state_path) as f:
                return json.load(f)
        return {}

    def _save_state(self, updates: Dict[str, Any]) -> None:
        """Merge updates into state.json on PVC."""
        state = self._load_state()
        state.update(updates)
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2)

    def _write_status(self, state: str, error: Optional[str] = None) -> None:
        """Write status.json to PVC."""
        status = {
            "run_id": self.run_id,
            "state": state,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if error:
            status["error"] = error
        with open(self.run_dir / "status.json", "w") as f:
            json.dump(status, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Discovery orchestrator")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--pvc", default=DEFAULT_PVC)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--max-parallelism", type=int, default=DEFAULT_MAX_PARALLELISM)
    parser.add_argument("--min-parallelism", type=int, default=DEFAULT_MIN_PARALLELISM)
    parser.add_argument("--reserve", type=int, default=DEFAULT_RESERVE_CARDS)
    parser.add_argument("--per-pod-timeout", type=int, default=DEFAULT_PER_POD_TIMEOUT)
    parser.add_argument("--pytorch-root", default="/workspace/pytorch")
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--files", nargs="*", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    orch = Orchestrator(
        run_id=args.run_id,
        namespace=args.namespace,
        pvc=args.pvc,
        image=args.image,
        max_parallelism=args.max_parallelism,
        min_parallelism=args.min_parallelism,
        reserve_cards=args.reserve,
        per_pod_timeout=args.per_pod_timeout,
        pytorch_root=args.pytorch_root,
        baseline=args.baseline,
        files=args.files,
    )
    orch.run()


if __name__ == "__main__":
    main()
