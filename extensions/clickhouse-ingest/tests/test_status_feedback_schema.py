"""status_*, external_* and feedback_* tables: models match the DDL, bad rows are refused, views avoid the known pitfalls."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
import regex as re

from spyre_clickhouse_ingest import schema
from spyre_clickhouse_ingest.schema import (
    EXTERNAL_EVENTS,
    EXTERNAL_FEED_FETCHES,
    EXTERNAL_REMINDERS,
    FEEDBACK_SIGNALS,
    STATUS_ANNOTATIONS,
    STATUS_SAMPLES,
    SchemaError,
    TABLES,
    status_sample_id,
)

SCHEMA = Path(__file__).resolve().parents[1] / "schema"
FILES = {
    "status_samples": "31-status.sql",
    "status_annotations": "31-status.sql",
    "external_feed_fetches": "32-external-feeds.sql",
    "external_events": "32-external-feeds.sql",
    "external_reminders": "32-external-feeds.sql",
    "feedback_signals": "33-feedback.sql",
}
VIEW_FILES = (
    "65-status-views.sql",
    "66-external-feed-views.sql",
    "67-feedback-views.sql",
)
VIEWS = {f: (SCHEMA / f).read_text() for f in VIEW_FILES}
MODELS = [
    STATUS_SAMPLES,
    STATUS_ANNOTATIONS,
    EXTERNAL_FEED_FETCHES,
    EXTERNAL_EVENTS,
    EXTERNAL_REMINDERS,
    FEEDBACK_SIGNALS,
]
T0 = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)


def _block(table: str) -> str:
    text = (SCHEMA / FILES[table]).read_text()
    block = re.search(
        rf"CREATE TABLE IF NOT EXISTS {table}\s*\((.*?)\n\)\s*\nENGINE", text, re.S
    ).group(1)
    return block


def _ddl_columns(table: str) -> list[str]:
    cols, in_constraint = [], False
    for line in _block(table).splitlines():
        line = line.split("--")[0].strip()
        if not line:
            continue
        if line.startswith("CONSTRAINT"):
            in_constraint = not line.rstrip(",").endswith(")")
            continue
        if in_constraint:
            in_constraint = not line.rstrip(",").endswith(")")
            continue
        cols.append(line.split()[0])
    return cols


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.name)
def test_model_columns_are_the_ddl_columns(model):
    # `bucket` is MATERIALIZED from ts, so it is not inserted.
    server = {"audit_uuid", "audit_timestamp", "bucket"}
    assert [c for c in _ddl_columns(model.name) if c not in server] == list(
        model.columns
    )
    assert model.name in TABLES


def _fill(model, base, over):
    row = {c: "" for c in model.columns}
    row.update(props={}, **base)
    row.update(over)
    return row


def _sample(**over):
    return _fill(
        STATUS_SAMPLES,
        dict(
            sample_id="vm|jenkins|http_ping|1",
            ts=T0,
            vantage="aiplatformwh-vm",
            component="jenkins",
            probe="http_ping",
            state="ok",
            http_status=200,
            latency_ms=12,
            ingested_at=T0,
            upload_path="primary",
        ),
        over,
    )


def _note(**over):
    return _fill(
        STATUS_ANNOTATIONS,
        dict(
            annotation_id="a1",
            ts=T0,
            component="jenkins",
            kind="human",
            author="ashok@example.com",
            text="restarting the controller",
            expires_at=None,
            supersedes="",
        ),
        over,
    )


def _fetch(**over):
    return _fill(
        EXTERNAL_FEED_FETCHES,
        dict(
            fetched_at=T0,
            vantage="laptop",
            source="githubstatus",
            feed_url="https://www.githubstatus.com/api/v2/summary.json",
            outcome="ok",
            page_indicator="none",
            http_status=200,
            latency_ms=100,
            bytes=1000,
        ),
        over,
    )


def _event(**over):
    return _fill(
        EXTERNAL_EVENTS,
        dict(
            observed_at=T0,
            event_id="githubstatus:abc",
            vantage="laptop",
            source="githubstatus",
            component="github-com",
            kind="maintenance",
            impact="maintenance",
            status="scheduled",
            starts_at=T0,
            ends_at=T0,
            title="Planned maintenance",
            upstream_updated_at=None,
        ),
        over,
    )


def _reminder(**over):
    return _fill(
        EXTERNAL_REMINDERS,
        dict(
            sent_at=T0,
            reminder_key="githubstatus:abc|1d|1",
            event_id="githubstatus:abc",
            threshold="1d",
            window_starts_at=T0,
            recipients=["ci-team@example.com"],
        ),
        over,
    )


def _signal(**over):
    return _fill(
        FEEDBACK_SIGNALS,
        dict(
            created_at=T0,
            target_type="signature_match",
            target_id="Spyre/orchestrator#12",
            signal="up",
            source="explicit",
            actor="ashok@example.com",
            weight_hint=1.0,
            signature_id="artifactory-401",
            signature_version="1",
        ),
        over,
    )


GOOD = [
    (STATUS_SAMPLES, _sample()),
    (STATUS_ANNOTATIONS, _note()),
    (EXTERNAL_FEED_FETCHES, _fetch()),
    (EXTERNAL_EVENTS, _event()),
    (EXTERNAL_REMINDERS, _reminder()),
    (FEEDBACK_SIGNALS, _signal()),
]


@pytest.mark.parametrize("model,row", GOOD, ids=lambda x: getattr(x, "name", ""))
def test_good_rows_are_ordered_by_columns(model, row):
    out = model.row(row)
    assert len(out) == len(model.columns)
    assert out == [row[c] for c in model.columns]


BAD = [
    (STATUS_SAMPLES, _sample(state="up"), "state"),
    (STATUS_SAMPLES, _sample(upload_path="later"), "upload_path"),
    (STATUS_SAMPLES, _sample(detail="x" * 501), "500"),
    (STATUS_SAMPLES, _sample(sample_id=""), "sample_id"),
    (STATUS_ANNOTATIONS, _note(kind="bot"), "kind"),
    (STATUS_ANNOTATIONS, _note(text="x" * 2001), "2000"),
    (STATUS_ANNOTATIONS, _note(supersedes="a1"), "supersede"),
    (EXTERNAL_FEED_FETCHES, _fetch(outcome="timeout"), "outcome"),
    (EXTERNAL_FEED_FETCHES, _fetch(page_indicator="severe"), "page_indicator"),
    (EXTERNAL_EVENTS, _event(kind="outage"), "kind"),
    (EXTERNAL_EVENTS, _event(status="over"), "status"),
    (EXTERNAL_EVENTS, _event(impact="huge"), "impact"),
    (EXTERNAL_EVENTS, _event(title="t" * 201), "200"),
    (EXTERNAL_EVENTS, _event(title=""), "title"),
    (
        EXTERNAL_EVENTS,
        _event(starts_at=T0, ends_at=datetime(2026, 10, 24, 11, tzinfo=timezone.utc)),
        "ends_at",
    ),
    (EXTERNAL_REMINDERS, _reminder(threshold="2h"), "threshold"),
    (EXTERNAL_REMINDERS, _reminder(reminder_key=""), "reminder_key"),
    (FEEDBACK_SIGNALS, _signal(target_type="repo"), "target_type"),
    (FEEDBACK_SIGNALS, _signal(signal="love"), "signal"),
    (FEEDBACK_SIGNALS, _signal(source="bot"), "source"),
    (FEEDBACK_SIGNALS, _signal(detail="d" * 501), "500"),
    (FEEDBACK_SIGNALS, _signal(weight_hint=-1.0), "weight"),
    (FEEDBACK_SIGNALS, _signal(signature_id=""), "signature"),
    (FEEDBACK_SIGNALS, _signal(actor=""), "actor"),
]


@pytest.mark.parametrize("model,row,why", BAD, ids=lambda x: str(x)[:24])
def test_rows_outside_the_ddl_checks_are_refused(model, row, why):
    with pytest.raises(SchemaError, match=why):
        model.row(row)


def test_a_signal_on_something_other_than_a_signature_needs_no_signature_id():
    FEEDBACK_SIGNALS.row(
        _signal(target_type="component", signature_id="", signal="affected")
    )


def test_an_open_ended_maintenance_window_is_allowed():
    EXTERNAL_EVENTS.row(_event(starts_at=T0, ends_at=None))
    EXTERNAL_EVENTS.row(_event(starts_at=None, ends_at=None))


def test_ddl_vocabularies_equal_the_models():
    checks = [
        ("status_samples", "state", schema.STATUS_STATE_VALUES),
        ("status_samples", "upload_path", schema.STATUS_UPLOAD_PATH_VALUES),
        ("status_annotations", "kind", schema.ANNOTATION_KIND_VALUES),
        ("external_feed_fetches", "outcome", schema.FETCH_OUTCOME_VALUES),
        ("external_feed_fetches", "page_indicator", schema.FETCH_INDICATOR_VALUES),
        ("external_events", "kind", schema.EXTERNAL_KIND_VALUES),
        ("external_events", "impact", schema.EXTERNAL_IMPACT_VALUES),
        ("external_events", "status", schema.EXTERNAL_STATUS_VALUES),
        ("external_reminders", "threshold", schema.REMINDER_THRESHOLD_VALUES),
        ("feedback_signals", "target_type", schema.FEEDBACK_TARGET_VALUES),
        ("feedback_signals", "signal", schema.FEEDBACK_SIGNAL_VALUES),
        ("feedback_signals", "source", schema.FEEDBACK_SOURCE_VALUES),
    ]
    for table, col, allowed in checks:
        listed = re.search(rf"CHECK {col} IN\s*\(([^)]*)\)", _block(table)).group(1)
        assert set(re.findall(r"'([^']*)'", listed)) == set(allowed), (table, col)


def test_length_caps_in_the_ddl_equal_the_model_caps():
    pairs = [
        ("status_samples", "detail", schema.MAX_STATUS_DETAIL),
        ("status_annotations", "text", schema.MAX_ANNOTATION_TEXT),
        ("external_events", "title", schema.MAX_EXTERNAL_TITLE),
        ("feedback_signals", "detail", schema.MAX_FEEDBACK_DETAIL),
    ]
    for table, col, cap in pairs:
        assert f"length({col}) <= {cap}" in _block(table), (table, col)


def test_sample_id_repeats_inside_a_bucket_and_changes_across_buckets():
    a = status_sample_id(
        "vm",
        "jenkins",
        "http_ping",
        datetime(2026, 10, 24, 12, 1, 5, tzinfo=timezone.utc),
    )
    b = status_sample_id(
        "vm",
        "jenkins",
        "http_ping",
        datetime(2026, 10, 24, 12, 4, 59, tzinfo=timezone.utc),
    )
    c = status_sample_id(
        "vm",
        "jenkins",
        "http_ping",
        datetime(2026, 10, 24, 12, 5, 0, tzinfo=timezone.utc),
    )
    assert a == b != c
    assert a.startswith("vm|jenkins|http_ping|")
    assert status_sample_id("laptop", "jenkins", "http_ping", T0) != status_sample_id(
        "vm", "jenkins", "http_ping", T0
    )


def test_samples_dedupe_by_id_keep_the_original_time_and_expire():
    text = (SCHEMA / "31-status.sql").read_text()
    block = text[
        text.index("EXISTS status_samples") : text.index("EXISTS status_annotations")
    ]
    assert "ReplacingMergeTree(ingested_at)" in block
    assert "ORDER BY (component, vantage, bucket, sample_id)" in block
    assert "bucket       DateTime('UTC') MATERIALIZED toStartOfFiveMinutes(ts)" in block
    assert "TTL toDateTime(ts) + INTERVAL 400 DAY" in block


def test_the_append_only_tables_are_plain_mergetree_and_only_the_log_like_ones_expire():
    external = (SCHEMA / "32-external-feeds.sql").read_text()
    assert external.count("TTL") == 3
    assert (SCHEMA / "33-feedback.sql").read_text().count("TTL") == 0
    assert "ReplacingMergeTree" not in external


@pytest.mark.parametrize("name", VIEW_FILES)
def test_view_aggregates_are_not_aliased_to_their_source_columns(name):
    # An alias equal to a source column is substituted into the other aggregates (ILLEGAL_AGGREGATION).
    text = VIEWS[name]
    for src in (
        "ts",
        "state",
        "status",
        "impact",
        "kind",
        "starts_at",
        "ends_at",
        "title",
        "signal",
        "actor",
        "created_at",
    ):
        assert not re.search(
            rf"\b(?:argMax|max|min|any|anyLast)\w*\([^)]*\)\s+AS {src}\b", text
        ), (name, src)


def test_unset_times_are_null_not_the_epoch():
    health = VIEWS["66-external-feed-views.sql"]
    assert "toDateTime64(0, 3, 'UTC'), NULL" in health
    assert "toDateTime64(0, 3, 'UTC'), NULL" in VIEWS["65-status-views.sql"]


def test_ratios_are_guarded_against_division_by_zero():
    assert "if(known_buckets = 0, NULL" in VIEWS["65-status-views.sql"]
    assert "if(labelled_matches = 0, NULL" in VIEWS["67-feedback-views.sql"]


def test_quorum_order_unknown_then_operational_then_the_failure_cases():
    block = VIEWS["65-status-views.sql"]
    block = block[block.index("multiIf(") : block.index(") AS state")]
    labels = re.findall(r"'(\w+)',?\s*$", block, re.M)
    assert labels[:2] == ["unknown", "operational"]
    assert set(labels) == {
        "unknown",
        "operational",
        "degraded",
        "major_outage",
        "partial_outage",
    }
    # two vantages failing together is only a major outage after two buckets in a row
    assert "both_failed_buckets, 0) >= 2, 'major_outage', 'partial_outage'" in block


def test_only_the_view_files_define_views_and_only_the_table_files_define_tables():
    for name in VIEW_FILES:
        assert "CREATE TABLE" not in VIEWS[name]
    for f in set(FILES.values()):
        assert "CREATE VIEW" not in (SCHEMA / f).read_text()


def test_file_numbers_are_unique_and_tables_sort_before_their_views():
    numbers = [p.name.split("-")[0] for p in SCHEMA.glob("[0-9]*.sql")]
    assert len(numbers) == len(set(numbers))
    assert "31" < "65" and "32" < "66" and "33" < "67"
