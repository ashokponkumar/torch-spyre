#!/usr/bin/env python3
"""Classification tests for parse_hw_failures.

Log excerpts are trimmed from real GHA job logs, keeping the GHA timestamp
prefix and the `=== Attempt ... ===` banners the parser slices on.

Run: pytest .github/scripts/tests/test_parse_hw_failures_classification.py
"""

import importlib.util
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "parse_hw_failures.py"
_spec = importlib.util.spec_from_file_location("parse_hw_failures", _SRC)
phf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(phf)


def _parse(body: str, suite: str = "Some Suite") -> list[dict]:
    return phf.parse_log(body, run_id="1", suite_hint=suite)


def _only(body: str) -> dict:
    recs = _parse(body)
    assert len(recs) == 1, f"expected 1 attempt, got {len(recs)}"
    return recs[0]


TS = "2026-08-31T10:13:09.2822545Z "


def _attempt(inner: str, exit_code: int | None = 1) -> str:
    """Wrap log body in the attempt banners the parser slices on."""
    head = f"{TS}=== Attempt 1/2: Some Suite ===\n"
    tail = f"{TS}=== Attempt 1 FAILED (exit={exit_code}) ===\n" if exit_code else ""
    return head + inner + tail


INVENTORY_1PF = f"{TS}---- --> Detected:   1 AIU PFs,   0 AIU VFs,   0 NICs,   2 NVMEs\n"


def test_setup_abort_missing_pytorch_root():
    """The dominant real-world residual: harness dies before pytest collects."""
    rec = _only(
        _attempt(
            INVENTORY_1PF
            + f"{TS}[spyre_run] Resolving TORCH_ROOT...\n"
            + f"{TS}ERROR: Could not locate PyTorch source root.\n"
        )
    )
    assert rec["outcome"] == "failed"
    assert rec["failure_reason"] == "setup_abort"
    assert "PyTorch source root" in rec["failure_reason_detail"]["signal"]


def test_real_test_failure_from_pytest_summary():
    rec = _only(
        _attempt(
            INVENTORY_1PF
            + f"{TS}collecting ... collected 48 items\n"
            + f"{TS}============= 1 failed, 47 passed, 2 warnings in 91.71s =============\n"
        )
    )
    assert rec["failure_reason"] == "test_failure"
    assert rec["tests_failed"] == 1
    assert rec["failure_reason_detail"]["tests_failed"] == 1


def test_distributed_crash():
    rec = _only(
        _attempt(
            INVENTORY_1PF
            + f"{TS}E0831 api.py:1002] failed (exitcode: 255) local_rank: 1\n"
            + f"{TS}torch.distributed.elastic.multiprocessing.errors.ChildFailedError:\n"
            + f"{TS}collecting ... Fatal Python error: Aborted\n"
        )
    )
    assert rec["failure_reason"] == "distributed_crash"


def test_no_device_detected_is_hardware():
    rec = _only(
        _attempt(
            f"{TS}---- --> Detected:   0 AIU PFs,   0 AIU VFs,   0 NICs,   2 NVMEs\n"
            + f"{TS}some later failure\n"
        )
    )
    assert rec["failure_reason"] == "hardware_no_device"


def test_infra_fault_beats_setup_abort():
    """Runner-level faults are reported first: the setup error is a symptom."""
    rec = _only(
        _attempt(
            f"{TS}no space left on device\n"
            + f"{TS}ERROR: Could not locate PyTorch source root.\n"
        )
    )
    assert rec["failure_reason"] == "infra_fault"


def test_harness_failure_when_tests_ran_clean_but_attempt_failed():
    rec = _only(
        _attempt(
            INVENTORY_1PF
            + f"{TS}collecting ... collected 12 items\n"
            + f"{TS}============= 12 passed in 8.10s =============\n"
        )
    )
    assert rec["failure_reason"] == "harness_failure"


def test_prose_failed_count_is_not_a_test_failure():
    """Regression: 'Downloaded 2 failed-suite descriptor(s)' is not a tally.

    The retry-collector job succeeds and logs that line; the unanchored
    `\\d+ failed` pattern made it a failed attempt with 2 failed tests.
    """
    body = (
        f"{TS}Downloaded 2 failed-suite descriptor(s).\n"
        f"{TS}Single-card suites to retry: ['Inductor / Test Misc Shape C']\n"
    )
    rec = _only(body)
    assert rec["tests_failed"] == 0
    assert rec["outcome"] != "failed"
    assert rec["failure_reason"] == "none"


def test_ras_event_still_wins_over_residual_classes():
    """Pre-existing hardware classification must not regress.

    Uses the `RuntimeError: {...}` form, which is prefix-tolerant. The
    `ras_base.hpp` form is matched with `.match()` against the stripped line, so
    it only parses in logs captured without GHA's ISO-8601 line prefix.
    """
    blob = (
        '{"code":"0x1234","name":"RAS::CBRB::ResponseTimeout","category":"hardware",'
        '"severity":"fatal","description":"timeout","action":"retry","message":"m"}'
    )
    rec = _only(
        _attempt(
            f"{TS}RuntimeError: {blob}\n"
            + f"{TS}ERROR: Could not locate PyTorch source root.\n"
        )
    )
    assert rec["failure_reason"] == "hardware_ras_timeout"
    assert rec["ras_category"] == "hardware"


def test_passing_attempt_is_untouched():
    rec = _only(
        f"{TS}=== Attempt 1/2: Some Suite ===\n"
        + INVENTORY_1PF
        + f"{TS}collecting ... collected 364 items\n"
        + f"{TS}= 284 passed, 4 skipped, 10 xfailed in 424.12s =\n"
        + f"{TS}=== Attempt 1 PASSED ===\n"
    )
    assert rec["outcome"] == "passed"
    assert rec["failure_reason"] == "none"


@pytest.mark.parametrize(
    "line,expected",
    [
        ("= 1 failed, 47 passed, 2 warnings in 91.71s =", "1"),
        ("= 12 failed, 3 error in 4.00s =", "12"),
        ("Downloaded 2 failed-suite descriptor(s).", None),
        ("wrote 5 failed_suites.json entries", None),
    ],
)
def test_summary_failed_regex_anchoring(line, expected):
    m = phf.RE_SUMMARY_FAILED.search(line)
    assert (m.group("n") if m else None) == expected
