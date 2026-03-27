"""
Generate discovery_config.yaml and file_manifest.json for a discovery run.

The file manifest maps each Job completion index to a test file,
along with any per-file overrides (memory, timeout).
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from discovery.constants import (
    DEFAULT_PER_TEST_TIMEOUT,
    DEFAULT_WORKER_MEMORY_LIMIT,
    DEFAULT_WORKER_MEMORY_REQUEST,
    HIGH_MEMORY_FILES,
    HIGH_MEMORY_LIMIT,
    UPSTREAM_TEST_FILES,
)


def build_file_manifest(
    files: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Build a list of per-index file entries for the Indexed Job.

    Args:
        files: Specific test files to run. If None, uses all UPSTREAM_TEST_FILES.

    Returns:
        List of dicts, one per completion index:
        [{"index": 0, "file": "test_complex.py", "memory_request": "8Gi", ...}, ...]
    """
    file_list = files if files else list(UPSTREAM_TEST_FILES)
    manifest = []
    for idx, filename in enumerate(file_list):
        base = Path(filename).name
        is_heavy = base in HIGH_MEMORY_FILES
        entry = {
            "index": idx,
            "file": filename,
            "memory_request": DEFAULT_WORKER_MEMORY_REQUEST,
            "memory_limit": HIGH_MEMORY_LIMIT if is_heavy else DEFAULT_WORKER_MEMORY_LIMIT,
            "test_timeout": DEFAULT_PER_TEST_TIMEOUT,
        }
        manifest.append(entry)
    return manifest


def build_discovery_config(
    run_id: str,
    repo: str,
    branch: str,
    pytorch_repo: str,
    pytorch_branch: str,
    max_parallelism: int,
    min_parallelism: int,
    reserve_cards: int,
    per_pod_timeout: int,
    namespace: str,
    pvc: str,
    image: str,
    baseline: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the discovery_config.yaml content."""
    return {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repos": {
            "torch_spyre": {"url": repo, "branch": branch},
            "pytorch": {"url": pytorch_repo, "branch": pytorch_branch},
        },
        "cluster": {
            "namespace": namespace,
            "pvc": pvc,
            "image": image,
        },
        "parallelism": {
            "max": max_parallelism,
            "min": min_parallelism,
            "reserve_cards": reserve_cards,
        },
        "timeouts": {
            "per_pod": per_pod_timeout,
            "per_test": DEFAULT_PER_TEST_TIMEOUT,
        },
        "baseline": baseline,
    }


def write_manifest(manifest: List[Dict[str, Any]], output_path: Path) -> None:
    """Write file_manifest.json to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)


def write_discovery_config(config: Dict[str, Any], output_path: Path) -> None:
    """Write discovery_config.yaml to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    """Load file_manifest.json from disk."""
    with open(path) as f:
        return json.load(f)


def load_discovery_config(path: Path) -> Dict[str, Any]:
    """Load discovery_config.yaml from disk."""
    with open(path) as f:
        return yaml.safe_load(f)
