"""
Compare two discovery_results.json files and produce a diff report.

Identifies regressions (pass→fail), fixes (fail→pass),
new tests, and removed tests.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

from discovery.aggregator import load_results


def _build_test_map(results: Dict[str, Any]) -> Dict[str, str]:
    """Build a map of test_id → status from discovery_results.json."""
    test_map = {}
    for file_entry in results.get("files", []):
        for test in file_entry.get("tests", []):
            test_id = test.get("test_id", "")
            status = test.get("status", "unknown")
            test_map[test_id] = status
    return test_map


def _build_file_map(results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build a map of file → counts from discovery_results.json."""
    file_map = {}
    for file_entry in results.get("files", []):
        file_map[file_entry["file"]] = {
            "completed": file_entry.get("completed", False),
            "counts": file_entry.get("counts", {}),
            "exit_code": file_entry.get("exit_code"),
            "signal": file_entry.get("signal"),
        }
    return file_map


def diff_results(
    new_results: Dict[str, Any],
    baseline_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare two result sets and return a diff report.

    Returns a dict with regressions, fixes, new_tests, removed_tests,
    and file-level changes.
    """
    new_tests = _build_test_map(new_results)
    base_tests = _build_test_map(baseline_results)

    new_files = _build_file_map(new_results)
    base_files = _build_file_map(baseline_results)

    all_test_ids = set(new_tests.keys()) | set(base_tests.keys())

    regressions: List[Dict[str, str]] = []
    fixes: List[Dict[str, str]] = []
    new_test_ids: List[str] = []
    removed_test_ids: List[str] = []

    for tid in sorted(all_test_ids):
        old_status = base_tests.get(tid)
        new_status = new_tests.get(tid)

        if old_status is None:
            new_test_ids.append(tid)
            continue
        if new_status is None:
            removed_test_ids.append(tid)
            continue

        # Regression: was passing, now failing
        if old_status == "passed" and new_status in ("failed", "error"):
            regressions.append({
                "test_id": tid,
                "old_status": old_status,
                "new_status": new_status,
            })

        # Fix: was failing, now passing
        if old_status in ("failed", "error") and new_status == "passed":
            fixes.append({
                "test_id": tid,
                "old_status": old_status,
                "new_status": new_status,
            })

    # File-level changes
    file_changes: List[Dict[str, Any]] = []
    all_files = set(new_files.keys()) | set(base_files.keys())
    for fname in sorted(all_files):
        nf = new_files.get(fname, {})
        bf = base_files.get(fname, {})
        nc = nf.get("counts", {})
        bc = bf.get("counts", {})

        old_pass = bc.get("passed", 0)
        new_pass = nc.get("passed", 0)
        old_fail = bc.get("failed", 0) + bc.get("errors", 0)
        new_fail = nc.get("failed", 0) + nc.get("errors", 0)

        if old_pass != new_pass or old_fail != new_fail:
            file_changes.append({
                "file": fname,
                "old_passed": old_pass,
                "new_passed": new_pass,
                "old_failed": old_fail,
                "new_failed": new_fail,
                "pass_delta": new_pass - old_pass,
                "fail_delta": new_fail - old_fail,
            })

    return {
        "summary": {
            "regressions": len(regressions),
            "fixes": len(fixes),
            "new_tests": len(new_test_ids),
            "removed_tests": len(removed_test_ids),
            "files_changed": len(file_changes),
        },
        "regressions": regressions,
        "fixes": fixes,
        "new_tests": new_test_ids[:100],  # cap for readability
        "removed_tests": removed_test_ids[:100],
        "file_changes": file_changes,
    }


def generate_diff_report(diff: Dict[str, Any]) -> str:
    """Generate a markdown diff report from a diff dict."""
    lines = []
    s = diff["summary"]

    lines.append("# Discovery Diff Report\n")
    lines.append(f"| Metric | Count |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Regressions (pass→fail) | {s['regressions']} |")
    lines.append(f"| Fixes (fail→pass) | {s['fixes']} |")
    lines.append(f"| New tests | {s['new_tests']} |")
    lines.append(f"| Removed tests | {s['removed_tests']} |")
    lines.append(f"| Files changed | {s['files_changed']} |")
    lines.append("")

    if diff["regressions"]:
        lines.append("## Regressions\n")
        for r in diff["regressions"]:
            lines.append(f"- `{r['test_id']}`: {r['old_status']} → {r['new_status']}")
        lines.append("")

    if diff["fixes"]:
        lines.append("## Fixes\n")
        for f in diff["fixes"]:
            lines.append(f"- `{f['test_id']}`: {f['old_status']} → {f['new_status']}")
        lines.append("")

    if diff["file_changes"]:
        lines.append("## File-Level Changes\n")
        lines.append("| File | Pass Δ | Fail Δ |")
        lines.append("|------|--------|--------|")
        for fc in diff["file_changes"]:
            pd = fc["pass_delta"]
            fd = fc["fail_delta"]
            ps = f"+{pd}" if pd > 0 else str(pd)
            fs = f"+{fd}" if fd > 0 else str(fd)
            lines.append(f"| {fc['file']} | {ps} | {fs} |")
        lines.append("")

    if diff["new_tests"]:
        lines.append(f"## New Tests ({len(diff['new_tests'])} shown)\n")
        for tid in diff["new_tests"][:20]:
            lines.append(f"- `{tid}`")
        if len(diff["new_tests"]) > 20:
            lines.append(f"- ... and {len(diff['new_tests']) - 20} more")
        lines.append("")

    if diff["removed_tests"]:
        lines.append(f"## Removed Tests ({len(diff['removed_tests'])} shown)\n")
        for tid in diff["removed_tests"][:20]:
            lines.append(f"- `{tid}`")
        if len(diff["removed_tests"]) > 20:
            lines.append(f"- ... and {len(diff['removed_tests']) - 20} more")
        lines.append("")

    return "\n".join(lines)


def diff_from_paths(
    new_path: Path, baseline_path: Path
) -> Tuple[Dict[str, Any], str]:
    """Load two result files, diff them, and return (diff_dict, markdown_report)."""
    new_results = load_results(new_path)
    baseline_results = load_results(baseline_path)
    diff = diff_results(new_results, baseline_results)
    report = generate_diff_report(diff)
    return diff, report
