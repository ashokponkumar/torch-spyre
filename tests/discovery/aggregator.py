"""
Aggregate per-worker JUnit XML results into a unified discovery_results.json.

Parses each worker's results.xml and summary.json from the PVC,
categorizes failures, and produces a single JSON report.
"""

import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from discovery.constants import ERROR_CATEGORIES


def categorize_error(message: str) -> str:
    """Match an error message against known patterns and return a category."""
    if not message:
        return "unknown"

    for category, info in ERROR_CATEGORIES.items():
        if category == "unknown":
            continue
        for pattern in info["patterns"]:
            if re.search(pattern, message, re.IGNORECASE):
                return category
    return "unknown"


def parse_junit_xml(xml_path: Path) -> List[Dict[str, Any]]:
    """Parse a JUnit XML file into a list of test result dicts."""
    results = []

    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return results

    root = tree.getroot()
    if root.tag == "testsuites":
        suites = root.findall("testsuite")
    else:
        suites = [root]

    for suite in suites:
        for tc in suite.findall("testcase"):
            classname = tc.get("classname", "")
            name = tc.get("name", "")
            time_s = float(tc.get("time", 0))

            entry: Dict[str, Any] = {
                "classname": classname,
                "name": name,
                "test_id": f"{classname}::{name}" if classname else name,
                "duration": time_s,
                "status": "passed",
                "error_category": None,
                "error_message": None,
            }

            failure = tc.find("failure")
            if failure is not None:
                msg = failure.get("message", "") or failure.text or ""
                entry["status"] = "failed"
                entry["error_message"] = msg[:500]  # truncate long messages
                entry["error_category"] = categorize_error(msg)

            error = tc.find("error")
            if error is not None:
                msg = error.get("message", "") or error.text or ""
                entry["status"] = "error"
                entry["error_message"] = msg[:500]
                entry["error_category"] = categorize_error(msg)

            skipped = tc.find("skipped")
            if skipped is not None:
                entry["status"] = "skipped"
                entry["error_message"] = skipped.get("message", "")

            results.append(entry)

    return results


def load_worker_summary(worker_dir: Path) -> Optional[Dict[str, Any]]:
    """Load the summary.json from a worker directory."""
    summary_path = worker_dir / "summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        return json.load(f)


def aggregate_results(
    run_dir: Path,
    num_files: int,
) -> Dict[str, Any]:
    """Aggregate all worker results from a discovery run.

    Args:
        run_dir: Path to /mnt/devwork/discovery/<run-id>/
        num_files: Expected number of test files

    Returns:
        Unified results dict ready to write as discovery_results.json
    """
    workers_dir = run_dir / "workers"
    files_results: List[Dict[str, Any]] = []
    total_stats = {
        "total_tests": 0,
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "incomplete": 0,
        "file_segfaults": 0,
    }

    for idx in range(num_files):
        worker_dir = workers_dir / str(idx)
        done = (worker_dir / "done.marker").exists()
        summary = load_worker_summary(worker_dir)

        file_entry: Dict[str, Any] = {
            "index": idx,
            "file": summary["file"] if summary else f"index_{idx}",
            "completed": done,
            "exit_code": summary.get("exit_code") if summary else None,
            "signal": summary.get("signal") if summary else None,
            "duration": summary.get("duration", 0) if summary else 0,
            "counts": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "errors": 0,
                "skipped": 0,
            },
            "tests": [],
            "error_categories": {},
        }

        if not done:
            total_stats["incomplete"] += 1
            files_results.append(file_entry)
            continue

        # Check for file-level segfault
        if summary and summary.get("signal") == "SIGSEGV":
            total_stats["file_segfaults"] += 1
            file_entry["error_categories"]["segfault"] = file_entry["counts"]["total"] or 1

        # Parse JUnit XML
        xml_path = worker_dir / "results.xml"
        tests = []
        if xml_path.exists():
            tests = parse_junit_xml(xml_path)

        # Tally
        cat_counts: Dict[str, int] = {}
        for t in tests:
            file_entry["counts"]["total"] += 1
            status = t["status"]
            if status == "passed":
                file_entry["counts"]["passed"] += 1
            elif status == "failed":
                file_entry["counts"]["failed"] += 1
                cat = t.get("error_category", "unknown")
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
            elif status == "error":
                file_entry["counts"]["errors"] += 1
                cat = t.get("error_category", "unknown")
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
            elif status == "skipped":
                file_entry["counts"]["skipped"] += 1

        file_entry["tests"] = tests
        file_entry["error_categories"] = cat_counts

        # Update totals
        for k in ["total", "passed", "failed", "errors", "skipped"]:
            total_stats[f"total_tests" if k == "total" else k] += file_entry["counts"][k]

        files_results.append(file_entry)

    # Build final results
    results = {
        "metadata": {
            "aggregated_at": datetime.now(timezone.utc).isoformat(),
            "total_files": num_files,
        },
        "summary": total_stats,
        "files": files_results,
    }

    return results


def write_results(results: Dict[str, Any], output_path: Path) -> None:
    """Write discovery_results.json."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def load_results(path: Path) -> Dict[str, Any]:
    """Load discovery_results.json."""
    with open(path) as f:
        return json.load(f)
