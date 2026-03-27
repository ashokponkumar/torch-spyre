"""
CLI for the upstream test discovery system.

Commands (all talk to the cluster via oc/kubectl):
  launch-orchestrator  Start a discovery run
  status              Check orchestrator + job progress
  logs                Tail orchestrator pod logs
  fetch-results       Copy results from PVC to local machine
  diff                Compare two completed runs
  resume              Resume a failed/interrupted run
  clean               Delete K8s resources for a run
  list-runs           List all discovery runs on the PVC
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml
from jinja2 import Environment, FileSystemLoader

from discovery.constants import (
    DEFAULT_IMAGE,
    DEFAULT_IMAGE_PULL_SECRET,
    DEFAULT_MAX_PARALLELISM,
    DEFAULT_MIN_PARALLELISM,
    DEFAULT_NAMESPACE,
    DEFAULT_PER_POD_TIMEOUT,
    DEFAULT_PYTORCH_BRANCH,
    DEFAULT_PYTORCH_REPO,
    DEFAULT_PVC,
    DEFAULT_RESERVE_CARDS,
    DEFAULT_SERVICE_ACCOUNT,
    DEFAULT_TORCH_SPYRE_BRANCH,
    DEFAULT_TORCH_SPYRE_REPO,
    DISCOVERY_BASE_DIR,
    PVC_MOUNT_PATH,
    UPSTREAM_TEST_FILES,
)
from discovery.differ import diff_from_paths
from discovery.k8s_manager import (
    apply_yaml,
    delete_resource,
    get_job_progress,
    get_pod_logs,
    get_pod_phases,
    run_oc,
)
from discovery.manifest import build_file_manifest


def _template_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(Path(__file__).parent / "templates")),
        keep_trailing_newline=True,
    )


def _generate_run_id() -> str:
    return datetime.now().strftime("%Y-%m-%d-%H%M")


def cmd_launch_orchestrator(args: argparse.Namespace) -> None:
    """Launch the orchestrator pod on the cluster."""
    run_id = args.run_id if args.run_id != "auto" else _generate_run_id()
    files = args.files.split(",") if args.files else list(UPSTREAM_TEST_FILES)

    print(f"Launching discovery run: {run_id}")
    print(f"  Files: {len(files)}")
    print(f"  Max parallelism: {args.parallelism}")
    print(f"  Namespace: {args.namespace}")
    print(f"  Image: {args.image}")

    # Generate manifest and config for the ConfigMap
    manifest = build_file_manifest(files)
    manifest_json = json.dumps(manifest, indent=2)

    config = {
        "run_id": run_id,
        "files": files,
        "max_parallelism": args.parallelism,
        "min_parallelism": args.min_parallelism,
        "reserve_cards": args.reserve,
        "per_pod_timeout": args.timeout,
        "baseline": args.baseline,
    }
    config_yaml = yaml.dump(config, default_flow_style=False)

    params = {
        "run_id": run_id,
        "torch_spyre_repo": args.repo,
        "torch_spyre_branch": args.branch,
        "pytorch_repo": args.pytorch_repo,
        "pytorch_branch": args.pytorch_branch,
        "namespace": args.namespace,
        "pvc": args.pvc,
        "image": args.image,
        "parallelism": args.parallelism,
        "files": files,
        "baseline": args.baseline,
    }
    params_json = json.dumps(params, indent=2)

    env = _template_env()

    if args.dry_run:
        print("\n--- DRY RUN: Orchestrator Pod YAML ---")

    # Render and apply ConfigMap
    cm_template = env.get_template("discovery-configmap.yaml.j2")
    cm_yaml = cm_template.render(
        run_id=run_id,
        namespace=args.namespace,
        discovery_config=config_yaml,
        file_manifest=manifest_json,
        run_params=params_json,
    )

    if args.dry_run:
        print(cm_yaml)
    else:
        result = apply_yaml(cm_yaml, args.namespace)
        print(f"  ConfigMap created: {result.strip()}")

    # Render and apply orchestrator pod
    pod_template = env.get_template("orchestrator-pod.yaml.j2")
    pod_yaml = pod_template.render(
        run_id=run_id,
        namespace=args.namespace,
        service_account=DEFAULT_SERVICE_ACCOUNT,
        image=args.image,
        image_pull_secret=DEFAULT_IMAGE_PULL_SECRET,
        pvc=args.pvc,
        pvc_mount_path=PVC_MOUNT_PATH,
        torch_spyre_repo=args.repo,
        torch_spyre_branch=args.branch,
        pytorch_repo=args.pytorch_repo,
        pytorch_branch=args.pytorch_branch,
        max_parallelism=args.parallelism,
        min_parallelism=args.min_parallelism,
        reserve_cards=args.reserve,
        baseline=args.baseline,
    )

    if args.dry_run:
        print("\n--- Orchestrator Pod YAML ---")
        print(pod_yaml)
        print("\n--- DRY RUN COMPLETE ---")
        return

    result = apply_yaml(pod_yaml, args.namespace)
    print(f"  Orchestrator pod created: {result.strip()}")

    print(f"\nDiscovery run {run_id} launched!")
    print(f"  Monitor: python -m discovery status --run-id {run_id}")
    print(f"  Logs:    python -m discovery logs --run-id {run_id} --follow")
    print(f"  Fetch:   python -m discovery fetch-results --run-id {run_id}")


def cmd_status(args: argparse.Namespace) -> None:
    """Check orchestrator + job progress."""
    run_id = args.run_id
    ns = args.namespace

    # Check orchestrator pod
    result = run_oc(
        ["get", "pod", f"discovery-orchestrator-{run_id}",
         "-o", "jsonpath={.status.phase}"],
        namespace=ns, check=False,
    )
    orch_phase = result.stdout.strip() if result.returncode == 0 else "NotFound"
    print(f"Orchestrator pod: {orch_phase}")

    # Check Job
    job_name = f"discovery-workers-{run_id}"
    progress = get_job_progress(job_name, ns)
    if progress["completions"] > 0:
        pct = progress["succeeded"] / progress["completions"] * 100
        print(f"Worker Job: {progress['succeeded']}/{progress['completions']} "
              f"({pct:.0f}%) succeeded, {progress['active']} active, "
              f"{progress['failed']} failed")
    else:
        print(f"Worker Job: not found or not yet created")

    # Pod phases
    phases = get_pod_phases(f"discovery-run={run_id}", ns)
    if phases:
        print(f"Worker pods: {phases}")

    # Check PVC status file (via exec into orchestrator pod)
    result = run_oc(
        ["exec", f"discovery-orchestrator-{run_id}", "--",
         "cat", f"{PVC_MOUNT_PATH}/{DISCOVERY_BASE_DIR}/{run_id}/status.json"],
        namespace=ns, check=False,
    )
    if result.returncode == 0:
        try:
            status = json.loads(result.stdout)
            print(f"Run status: {status.get('state', 'unknown')}")
            if status.get("error"):
                print(f"  Error: {status['error']}")
        except json.JSONDecodeError:
            pass


def cmd_logs(args: argparse.Namespace) -> None:
    """Tail orchestrator pod logs."""
    pod_name = f"discovery-orchestrator-{args.run_id}"
    cmd = ["oc", "-n", args.namespace, "logs", pod_name]
    if args.follow:
        cmd.append("-f")

    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        pass


def cmd_fetch_results(args: argparse.Namespace) -> None:
    """Copy results from PVC to local machine."""
    run_id = args.run_id
    output_dir = Path(args.output_dir or f"./results/{run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use oc rsync from orchestrator pod (which has PVC mounted)
    pod_name = f"discovery-orchestrator-{run_id}"
    src = f"{pod_name}:{PVC_MOUNT_PATH}/{DISCOVERY_BASE_DIR}/{run_id}/"

    print(f"Fetching results to {output_dir}")
    result = subprocess.run(
        ["oc", "-n", args.namespace, "rsync", src, str(output_dir), "--progress"],
        check=False,
    )

    if result.returncode != 0:
        # Fallback: try using a temporary pod for rsync
        print("Direct rsync failed — orchestrator pod may not be running.")
        print("Try fetching individual files with:")
        print(f"  oc -n {args.namespace} exec <running-pod> -- "
              f"cat {PVC_MOUNT_PATH}/{DISCOVERY_BASE_DIR}/{run_id}/discovery_results.json")
    else:
        print(f"Results saved to {output_dir}")

        # Print summary if available
        results_file = output_dir / "discovery_results.json"
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)
            s = data.get("summary", {})
            print(f"\nSummary:")
            print(f"  Total tests: {s.get('total_tests', '?')}")
            print(f"  Passed: {s.get('passed', '?')}")
            print(f"  Failed: {s.get('failed', '?')}")
            print(f"  Errors: {s.get('errors', '?')}")
            print(f"  Skipped: {s.get('skipped', '?')}")
            print(f"  Incomplete: {s.get('incomplete', '?')}")


def cmd_diff(args: argparse.Namespace) -> None:
    """Compare two completed runs."""
    new_path = Path(args.run_id) if Path(args.run_id).exists() else Path(f"./results/{args.run_id}/discovery_results.json")
    base_path = Path(args.baseline) if Path(args.baseline).exists() else Path(f"./results/{args.baseline}/discovery_results.json")

    if not new_path.exists():
        print(f"Results not found at {new_path}")
        print(f"Fetch them first: python -m discovery fetch-results --run-id {args.run_id}")
        sys.exit(1)

    if not base_path.exists():
        print(f"Baseline results not found at {base_path}")
        sys.exit(1)

    diff_dict, report = diff_from_paths(new_path, base_path)
    print(report)

    # Also write to file
    output = Path(f"./results/diff_{args.baseline}_vs_{args.run_id}.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        f.write(report)
    print(f"\nReport saved to {output}")


def cmd_resume(args: argparse.Namespace) -> None:
    """Resume a failed/interrupted run."""
    run_id = args.run_id
    print(f"Resuming run {run_id}")

    # Delete old orchestrator pod if exists
    delete_resource("pod", f"discovery-orchestrator-{run_id}", args.namespace)

    # Re-launch orchestrator — it will read state.json from PVC and resume
    # We need the original params from the ConfigMap
    result = run_oc(
        ["get", "configmap", f"discovery-config-{run_id}",
         "-o", "jsonpath={.data.run_params\\.json}"],
        namespace=args.namespace, check=False,
    )

    if result.returncode != 0:
        print("ConfigMap not found. Cannot resume without original parameters.")
        sys.exit(1)

    params = json.loads(result.stdout)
    env = _template_env()

    pod_template = env.get_template("orchestrator-pod.yaml.j2")
    pod_yaml = pod_template.render(
        run_id=run_id,
        namespace=args.namespace,
        service_account=DEFAULT_SERVICE_ACCOUNT,
        image=params.get("image", DEFAULT_IMAGE),
        image_pull_secret=DEFAULT_IMAGE_PULL_SECRET,
        pvc=params.get("pvc", DEFAULT_PVC),
        pvc_mount_path=PVC_MOUNT_PATH,
        torch_spyre_repo=params.get("torch_spyre_repo", DEFAULT_TORCH_SPYRE_REPO),
        torch_spyre_branch=params.get("torch_spyre_branch", DEFAULT_TORCH_SPYRE_BRANCH),
        pytorch_repo=params.get("pytorch_repo", DEFAULT_PYTORCH_REPO),
        pytorch_branch=params.get("pytorch_branch", DEFAULT_PYTORCH_BRANCH),
        max_parallelism=params.get("parallelism", DEFAULT_MAX_PARALLELISM),
        min_parallelism=params.get("min_parallelism", DEFAULT_MIN_PARALLELISM),
        reserve_cards=params.get("reserve_cards", DEFAULT_RESERVE_CARDS),
        baseline=params.get("baseline"),
    )

    apply_yaml(pod_yaml, args.namespace)
    print(f"Orchestrator pod re-created. It will resume from last checkpoint.")
    print(f"  Logs: python -m discovery logs --run-id {run_id} --follow")


def cmd_clean(args: argparse.Namespace) -> None:
    """Delete K8s resources for a run."""
    run_id = args.run_id
    ns = args.namespace

    print(f"Cleaning up resources for run {run_id}")
    delete_resource("pod", f"discovery-orchestrator-{run_id}", ns)
    delete_resource("job", f"discovery-workers-{run_id}", ns)
    delete_resource("configmap", f"discovery-config-{run_id}", ns)
    delete_resource("configmap", f"discovery-worker-config-{run_id}", ns)

    print("K8s resources deleted. PVC data preserved.")
    print(f"  To also delete PVC data, exec into a pod and: "
          f"rm -rf {PVC_MOUNT_PATH}/{DISCOVERY_BASE_DIR}/{run_id}")


def cmd_list_runs(args: argparse.Namespace) -> None:
    """List all discovery runs on the PVC."""
    ns = args.namespace

    # Find a running pod that has the PVC mounted
    result = run_oc(
        ["get", "pods", "-l", "app=discovery-orchestrator",
         "-o", "jsonpath={.items[0].metadata.name}"],
        namespace=ns, check=False,
    )

    pod_name = result.stdout.strip() if result.returncode == 0 else ""

    if not pod_name:
        # Try any pod with the PVC
        print("No orchestrator pod running. Checking for any pod with PVC...")
        result = run_oc(
            ["get", "pods",
             "-o", "jsonpath={range .items[*]}{.metadata.name}{\"\\n\"}{end}"],
            namespace=ns, check=False,
        )
        if result.returncode != 0:
            print("Cannot list runs — no accessible pods found")
            return
        pods = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
        if not pods:
            print("No pods found in namespace")
            return
        pod_name = pods[0]

    result = run_oc(
        ["exec", pod_name, "--",
         "ls", "-lt", f"{PVC_MOUNT_PATH}/{DISCOVERY_BASE_DIR}/"],
        namespace=ns, check=False,
    )

    if result.returncode == 0:
        print(f"Discovery runs on PVC:")
        print(result.stdout)
    else:
        print("No discovery runs found or PVC not mounted on this pod")

    # Also list K8s resources
    result = run_oc(
        ["get", "pods,jobs", "-l", "app in (discovery-orchestrator,discovery-worker)",
         "--sort-by=.metadata.creationTimestamp"],
        namespace=ns, check=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        print(f"\nActive K8s resources:")
        print(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="discovery",
        description="Upstream test discovery system for Spyre",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # launch-orchestrator
    p_launch = subparsers.add_parser(
        "launch-orchestrator", help="Start a discovery run"
    )
    p_launch.add_argument("--run-id", default="auto")
    p_launch.add_argument("--parallelism", type=int, default=DEFAULT_MAX_PARALLELISM)
    p_launch.add_argument("--min-parallelism", type=int, default=DEFAULT_MIN_PARALLELISM)
    p_launch.add_argument("--reserve", type=int, default=DEFAULT_RESERVE_CARDS)
    p_launch.add_argument("--repo", default=DEFAULT_TORCH_SPYRE_REPO)
    p_launch.add_argument("--branch", default=DEFAULT_TORCH_SPYRE_BRANCH)
    p_launch.add_argument("--pytorch-repo", default=DEFAULT_PYTORCH_REPO)
    p_launch.add_argument("--pytorch-branch", default=DEFAULT_PYTORCH_BRANCH)
    p_launch.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    p_launch.add_argument("--pvc", default=DEFAULT_PVC)
    p_launch.add_argument("--image", default=DEFAULT_IMAGE)
    p_launch.add_argument("--timeout", type=int, default=DEFAULT_PER_POD_TIMEOUT)
    p_launch.add_argument("--baseline", default=None)
    p_launch.add_argument("--files", default=None, help="Comma-separated list of test files")
    p_launch.add_argument("--dry-run", action="store_true")
    p_launch.set_defaults(func=cmd_launch_orchestrator)

    # status
    p_status = subparsers.add_parser("status", help="Check progress")
    p_status.add_argument("--run-id", required=True)
    p_status.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    p_status.set_defaults(func=cmd_status)

    # logs
    p_logs = subparsers.add_parser("logs", help="Tail orchestrator logs")
    p_logs.add_argument("--run-id", required=True)
    p_logs.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    p_logs.add_argument("--follow", "-f", action="store_true")
    p_logs.set_defaults(func=cmd_logs)

    # fetch-results
    p_fetch = subparsers.add_parser("fetch-results", help="Copy results locally")
    p_fetch.add_argument("--run-id", required=True)
    p_fetch.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    p_fetch.add_argument("--output-dir", default=None)
    p_fetch.set_defaults(func=cmd_fetch_results)

    # diff
    p_diff = subparsers.add_parser("diff", help="Compare two runs")
    p_diff.add_argument("--run-id", required=True)
    p_diff.add_argument("--baseline", required=True)
    p_diff.set_defaults(func=cmd_diff)

    # resume
    p_resume = subparsers.add_parser("resume", help="Resume a failed run")
    p_resume.add_argument("--run-id", required=True)
    p_resume.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    p_resume.set_defaults(func=cmd_resume)

    # clean
    p_clean = subparsers.add_parser("clean", help="Delete K8s resources")
    p_clean.add_argument("--run-id", required=True)
    p_clean.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    p_clean.set_defaults(func=cmd_clean)

    # list-runs
    p_list = subparsers.add_parser("list-runs", help="List all runs")
    p_list.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    p_list.set_defaults(func=cmd_list_runs)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
