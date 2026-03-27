"""
Kubernetes resource management for discovery runs.

Used by both the CLI (to launch the orchestrator pod) and the orchestrator
(to create/monitor the Indexed Job). Uses `oc` CLI subprocess calls.
"""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from discovery.constants import (
    DEFAULT_HEARTBEAT_STALE_THRESHOLD,
    DEFAULT_NAMESPACE,
    DEFAULT_POLL_INTERVAL,
)


def run_oc(
    args: List[str],
    namespace: Optional[str] = None,
    capture: bool = True,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run an oc command."""
    cmd = ["oc"]
    if namespace:
        cmd.extend(["-n", namespace])
    cmd.extend(args)

    return subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        check=check,
    )


def apply_yaml(yaml_content: str, namespace: str, dry_run: bool = False) -> str:
    """Apply a YAML manifest to the cluster."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        tmp_path = f.name

    try:
        cmd = ["apply", "-f", tmp_path]
        if dry_run:
            cmd.append("--dry-run=server")
        result = run_oc(cmd, namespace=namespace)
        return result.stdout
    finally:
        os.unlink(tmp_path)


def delete_resource(
    kind: str, name: str, namespace: str, ignore_not_found: bool = True
) -> None:
    """Delete a K8s resource."""
    cmd = ["delete", kind, name]
    if ignore_not_found:
        cmd.append("--ignore-not-found=true")
    run_oc(cmd, namespace=namespace, check=not ignore_not_found)


def get_job_status(
    job_name: str, namespace: str
) -> Optional[Dict[str, Any]]:
    """Get the status of a Job as a dict."""
    result = run_oc(
        ["get", "job", job_name, "-o", "json"],
        namespace=namespace,
        check=False,
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)


def get_job_progress(job_name: str, namespace: str) -> Dict[str, int]:
    """Get active/succeeded/failed counts for a Job."""
    status = get_job_status(job_name, namespace)
    if not status:
        return {"active": 0, "succeeded": 0, "failed": 0, "completions": 0}

    job_status = status.get("status", {})
    spec = status.get("spec", {})
    return {
        "active": job_status.get("active", 0),
        "succeeded": job_status.get("succeeded", 0),
        "failed": job_status.get("failed", 0),
        "completions": spec.get("completions", 0),
    }


def is_job_complete(job_name: str, namespace: str) -> bool:
    """Check if a Job has completed (all indices done)."""
    progress = get_job_progress(job_name, namespace)
    completions = progress["completions"]
    return completions > 0 and progress["succeeded"] >= completions


def is_job_failed(job_name: str, namespace: str) -> bool:
    """Check if a Job has failed beyond backoff limit."""
    status = get_job_status(job_name, namespace)
    if not status:
        return True
    conditions = status.get("status", {}).get("conditions", [])
    for c in conditions:
        if c.get("type") == "Failed" and c.get("status") == "True":
            return True
    return False


def get_pod_list(
    label_selector: str, namespace: str
) -> List[Dict[str, Any]]:
    """Get list of pods matching a label selector."""
    result = run_oc(
        ["get", "pods", "-l", label_selector, "-o", "json"],
        namespace=namespace,
        check=False,
    )
    if result.returncode != 0:
        return []
    data = json.loads(result.stdout)
    return data.get("items", [])


def get_pod_phases(label_selector: str, namespace: str) -> Dict[str, int]:
    """Count pods in each phase (Running, Pending, Succeeded, Failed)."""
    pods = get_pod_list(label_selector, namespace)
    phases: Dict[str, int] = {}
    for pod in pods:
        phase = pod.get("status", {}).get("phase", "Unknown")
        phases[phase] = phases.get(phase, 0) + 1
    return phases


def patch_job_parallelism(job_name: str, parallelism: int, namespace: str) -> None:
    """Patch a Job's parallelism (live scaling)."""
    patch = json.dumps({"spec": {"parallelism": parallelism}})
    run_oc(
        ["patch", "job", job_name, "-p", patch, "--type=merge"],
        namespace=namespace,
    )


def get_spyre_capacity(namespace: str) -> Dict[str, int]:
    """Query cluster for total and available Spyre cards.

    Returns {"total": N, "used_by_others": M, "allocatable": N-M}
    """
    # Get total allocatable Spyre PF cards across AIU nodes
    result = run_oc(
        ["get", "nodes", "-l", "node-role.kubernetes.io/aiu",
         "-o", "jsonpath={range .items[*]}{.status.allocatable.ibm\\.com/spyre_pf}{\"\\n\"}{end}"],
        check=False,
    )
    total = 0
    if result.returncode == 0:
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if line and line.isdigit():
                total += int(line)

    # Count cards used by non-discovery pods
    result = run_oc(
        ["get", "pods", "--all-namespaces",
         "-o", "jsonpath={range .items[*]}{.spec.containers[*].resources.requests.ibm\\.com/spyre_pf}{\"\\n\"}{end}"],
        check=False,
    )
    used = 0
    if result.returncode == 0:
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if line and line.isdigit():
                used += int(line)

    return {
        "total": total,
        "used_by_others": used,
        "allocatable": max(0, total - used),
    }


def force_delete_pod(pod_name: str, namespace: str) -> None:
    """Force-delete a stuck pod."""
    run_oc(
        ["delete", "pod", pod_name, "--grace-period=0", "--force"],
        namespace=namespace,
        check=False,
    )


def get_pod_logs(pod_name: str, namespace: str, follow: bool = False) -> str:
    """Get logs from a pod."""
    cmd = ["logs", pod_name]
    if follow:
        cmd.append("-f")
    result = run_oc(cmd, namespace=namespace, check=False, capture=not follow)
    return result.stdout if result.stdout else ""


def wait_for_pod_ready(
    pod_name: str, namespace: str, timeout: int = 300
) -> bool:
    """Wait for a pod to reach Running state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = run_oc(
            ["get", "pod", pod_name, "-o", "jsonpath={.status.phase}"],
            namespace=namespace,
            check=False,
        )
        phase = result.stdout.strip()
        if phase == "Running":
            return True
        if phase in ("Failed", "Error"):
            return False
        time.sleep(5)
    return False


def create_configmap(
    name: str,
    namespace: str,
    data: Dict[str, str],
    labels: Optional[Dict[str, str]] = None,
) -> str:
    """Create a ConfigMap from a data dict."""
    cm: Dict[str, Any] = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": name,
            "namespace": namespace,
        },
        "data": data,
    }
    if labels:
        cm["metadata"]["labels"] = labels

    import yaml
    return apply_yaml(yaml.dump(cm, default_flow_style=False), namespace)
