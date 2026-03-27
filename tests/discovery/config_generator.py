"""
Generate test_suite_config_daily.yaml from discovery_results.json.

Assigns test modes based on pass ratios:
  >80% pass → mandatory_success
  20-80% pass → xfail
  <20% pass → skip

Groups tests by (mode, error_category) for compact YAML.
Auto-detects passing dtypes per op from test names.
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

from discovery.aggregator import load_results
from discovery.constants import ERROR_CATEGORIES


def _extract_dtypes_from_test_name(test_name: str) -> Set[str]:
    """Extract dtype names from a test ID like 'TestOps::test_add_float32_spyre'."""
    known_dtypes = {
        "float16", "float32", "float64", "bfloat16",
        "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64",
        "complex32", "complex64", "complex128", "bool",
    }
    found = set()
    for dt in known_dtypes:
        if dt in test_name:
            found.add(dt)
    return found


def _extract_op_from_test_name(test_name: str) -> Optional[str]:
    """Try to extract an op name from test_id like 'TestOps::test_add_float32'."""
    # Common patterns: test_{op}_{dtype}, test_{op}_{variant}_{dtype}
    match = re.match(r".*::test_(\w+?)(?:_(?:float|int|uint|bfloat|complex|bool)\w*)?(?:_spyre|_privateuse1)?$", test_name)
    if match:
        return match.group(1)
    return None


def _compute_mode(counts: Dict[str, int]) -> str:
    """Determine unlisted_test_mode based on pass ratio."""
    total = counts.get("total", 0)
    if total == 0:
        return "skip"

    passed = counts.get("passed", 0)
    ratio = passed / total

    if ratio > 0.8:
        return "mandatory_success"
    elif ratio >= 0.2:
        return "xfail"
    else:
        return "skip"


def _dominant_error_category(error_categories: Dict[str, int]) -> str:
    """Return the most common error category for a file."""
    if not error_categories:
        return "unknown"
    return max(error_categories, key=error_categories.get)


def generate_config(
    results: Dict[str, Any],
    partial: bool = False,
) -> Dict[str, Any]:
    """Generate test_suite_config_daily.yaml from discovery results.

    Args:
        results: Parsed discovery_results.json
        partial: If True, include incomplete files as skip

    Returns:
        Dict representing the YAML structure (SpyreTestConfig-compatible)
    """
    files_config: List[Dict[str, Any]] = []
    all_passing_dtypes: Set[str] = set()
    all_passing_ops: Set[str] = set()
    op_dtype_map: Dict[str, Set[str]] = defaultdict(set)

    for file_entry in results.get("files", []):
        filename = file_entry["file"]
        completed = file_entry.get("completed", False)
        counts = file_entry.get("counts", {})
        error_cats = file_entry.get("error_categories", {})

        if not completed:
            if partial:
                files_config.append({
                    "path": f"${{PYTORCH}}/test/{filename}",
                    "unlisted_test_mode": "skip",
                    "description": "discovery incomplete",
                    "tests": [],
                })
            continue

        mode = _compute_mode(counts)
        dom_cat = _dominant_error_category(error_cats)
        cat_desc = ERROR_CATEGORIES.get(dom_cat, {}).get("description", "")

        file_cfg: Dict[str, Any] = {
            "path": f"${{PYTORCH}}/test/{filename}",
            "unlisted_test_mode": mode,
            "tests": [],
        }

        if mode != "mandatory_success" and cat_desc:
            file_cfg["description"] = f"Primary failure: {cat_desc}"

        # Group tests by status for compact representation
        passing_tests: List[str] = []
        failing_tests: Dict[str, List[str]] = defaultdict(list)  # category → test_ids
        skipped_tests: List[str] = []

        for test in file_entry.get("tests", []):
            test_id = test.get("test_id", "")
            status = test.get("status", "unknown")

            # Collect dtype/op info from passing tests
            if status == "passed":
                passing_tests.append(test_id)
                dtypes = _extract_dtypes_from_test_name(test_id)
                all_passing_dtypes.update(dtypes)
                op = _extract_op_from_test_name(test_id)
                if op:
                    all_passing_ops.add(op)
                    op_dtype_map[op].update(dtypes)

            elif status in ("failed", "error"):
                cat = test.get("error_category", "unknown")
                failing_tests[cat].append(test_id)

            elif status == "skipped":
                skipped_tests.append(test_id)

        # Add explicit test entries for non-default modes
        # For files with unlisted_test_mode=skip, list passing tests as mandatory_success
        if mode == "skip" and passing_tests:
            # Compact: group by class
            _add_grouped_entries(file_cfg["tests"], passing_tests, "mandatory_success")

        # For files with unlisted_test_mode=mandatory_success, list failing tests
        elif mode == "mandatory_success":
            for cat, test_ids in failing_tests.items():
                cat_desc_inner = ERROR_CATEGORIES.get(cat, {}).get("description", cat)
                _add_grouped_entries(
                    file_cfg["tests"], test_ids, "xfail",
                    description=cat_desc_inner,
                )

        # For xfail files, list both passing (mandatory_success) and skipped tests
        elif mode == "xfail":
            if passing_tests:
                _add_grouped_entries(file_cfg["tests"], passing_tests, "mandatory_success")
            for cat, test_ids in failing_tests.items():
                if cat in ("segfault", "oom", "timeout"):
                    _add_grouped_entries(
                        file_cfg["tests"], test_ids, "skip",
                        description=ERROR_CATEGORIES.get(cat, {}).get("description", cat),
                    )

        files_config.append(file_cfg)

    # Build global config
    global_config: Dict[str, Any] = {}

    if all_passing_dtypes:
        global_config["supported_dtypes"] = [
            {"name": dt} for dt in sorted(all_passing_dtypes)
        ]

    if all_passing_ops:
        supported_ops = []
        for op in sorted(all_passing_ops):
            op_entry: Dict[str, Any] = {"name": op}
            dtypes = op_dtype_map.get(op, set())
            if dtypes:
                op_entry["dtypes"] = [{"name": dt} for dt in sorted(dtypes)]
            supported_ops.append(op_entry)
        global_config["supported_ops"] = supported_ops

    config = {
        "test_suite_config": {
            "files": files_config,
        }
    }
    if global_config:
        config["test_suite_config"]["global"] = global_config

    return config


def _add_grouped_entries(
    tests_list: List[Dict[str, Any]],
    test_ids: List[str],
    mode: str,
    description: Optional[str] = None,
) -> None:
    """Add test entries grouped by class for compact YAML."""
    # Group by class name
    by_class: Dict[str, List[str]] = defaultdict(list)
    for tid in test_ids:
        parts = tid.split("::", 1)
        if len(parts) == 2:
            by_class[parts[0]].append(tid)
        else:
            by_class[""].append(tid)

    for _cls, ids in by_class.items():
        entry: Dict[str, Any] = {
            "names": ids,
            "mode": mode,
        }
        if description:
            entry["description"] = description
        tests_list.append(entry)


def write_config(config: Dict[str, Any], output_path: Path) -> None:
    """Write test_suite_config_daily.yaml."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, width=120)


def generate_from_results_file(
    results_path: Path,
    output_path: Path,
    partial: bool = False,
) -> Dict[str, Any]:
    """Load results, generate config, write to file."""
    results = load_results(results_path)
    config = generate_config(results, partial=partial)
    write_config(config, output_path)
    return config
