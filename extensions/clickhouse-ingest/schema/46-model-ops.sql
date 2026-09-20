-- Model-operator enablement coverage: which torch operations a model exercises, and whether each
-- ran on Spyre, fell back to CPU, or is not implemented.
--
-- NOT test results, despite the name. status is XPASS / XFAIL / FALLBACK -- an enablement verdict,
-- not a pass/fail one. These rows deliberately do NOT go into test_case_runs: run_case_counters_mv
-- buckets on status names, so folding an XFAIL-dominated set in there would corrupt every
-- pass_rate the functional views report.
--
-- SIX v1 TABLES COLLAPSE TO TWO. v1 had model_ops_{suites,variants} x {,_p,_z} -- the same schema
-- three times, one per architecture, each holding exactly one distinct workflow. `arch` is a
-- column here, which is how every other v2 table already handles this.
--
-- The join to the artifact side, which is the point of the v2 shape:
--   v2_run_id -> artifact_results.run_id -> .artifact_id -> artifacts -> artifact_tags
-- v1 had no such path: measured 0 of 33 commit_shas joining spyre_v2.artifacts, and 0 gha_run_ids
-- appearing in artifact_results. suite_id/variant_id stay as the per-row content discriminators
-- they already are, but they are NO LONGER an identity scheme -- v2_run_id is, derived by the same
-- two-case rule as every other writer (threaded uuid, else the coordinate hash), which retires
-- the fifth parallel identity these FixedString(64) hashes represented.
--
-- FORWARD-ONLY. No backfill: v1 rows cannot produce a v2_run_id (the hash inputs were never
-- recorded), and the populations do not even overlap -- v1 model_ops last wrote 2026-09-10,
-- spyre_v2.artifacts first wrote 2026-09-16.
--
-- ReplacingMergeTree(ingested_at) and the monthly partition carry over from v1 unchanged: a
-- re-ingest of the same suite must collapse, not duplicate.
--
-- No audit_uuid/audit_timestamp. v1 has both and they are non-nil on every row, but only because
-- of their DEFAULT generateUUIDv7()/now64(3) -- no writer sets either, so they record when a row
-- was inserted, which ingested_at already does. Same call as v2 hw_failure_diagnostics and
-- jenkins_agents.

CREATE TABLE IF NOT EXISTS model_ops_suites
(
    `suite_id` FixedString(64),
    -- Raw producer coordinate, kept for traceability; v2_run_id is the join key.
    `gha_run_id` UInt64,
    `run_id` String,
    -- Nil UUID / '' on an un-updated writer: a nil UUID joins nothing, which reads as
    -- "not linked". A defaulted hash would join everything, which is worse than blank.
    `v2_run_id` UUID DEFAULT toUUID('00000000-0000-0000-0000-000000000000'),
    `v2_artifact_id` UUID DEFAULT toUUID('00000000-0000-0000-0000-000000000000'),
    `component` LowCardinality(String) DEFAULT '',
    -- Replaces the _p / _z table suffixes. Canonical spelling (amd64 folds to x86_64).
    `arch` LowCardinality(String) DEFAULT '',
    `workflow` LowCardinality(String) DEFAULT '',
    `branch` LowCardinality(String) DEFAULT '',
    `commit_sha` String DEFAULT '',
    `suite_name` String,
    `model_name` LowCardinality(String) DEFAULT '',
    `yaml_file` String DEFAULT '',
    `total_tests` UInt32 DEFAULT 0,
    `spyre_enabled_count` UInt32 DEFAULT 0,
    `not_implemented_count` UInt32 DEFAULT 0,
    `cpu_fallback_count` UInt32 DEFAULT 0,
    `spyre_failed_count` UInt32 DEFAULT 0,
    `suite_outcome` LowCardinality(String) DEFAULT 'unknown',
    `suite_exit_code` Nullable(Int32),
    `tests_total` UInt32 DEFAULT 0,
    `tests_passed` UInt32 DEFAULT 0,
    `tests_failed` UInt32 DEFAULT 0,
    `tests_skipped` UInt32 DEFAULT 0,
    `tests_error` UInt32 DEFAULT 0,
    `tests_xfail` UInt32 DEFAULT 0,
    `tests_xpass` UInt32 DEFAULT 0,
    `duration_s` Float32 DEFAULT 0,
    `triggered_at` DateTime64(3, 'UTC'),
    `ingested_at` DateTime64(3, 'UTC')
)
ENGINE = ReplacingMergeTree(ingested_at)
PARTITION BY toYYYYMM(triggered_at)
ORDER BY (component, arch, suite_id);

CREATE TABLE IF NOT EXISTS model_ops_variants
(
    `variant_id` FixedString(64),
    `suite_id` FixedString(64),
    `gha_run_id` UInt64,
    `run_id` String,
    `v2_run_id` UUID DEFAULT toUUID('00000000-0000-0000-0000-000000000000'),
    `v2_artifact_id` UUID DEFAULT toUUID('00000000-0000-0000-0000-000000000000'),
    `component` LowCardinality(String) DEFAULT '',
    `arch` LowCardinality(String) DEFAULT '',
    `workflow` LowCardinality(String) DEFAULT '',
    `branch` LowCardinality(String) DEFAULT '',
    `commit_sha` String DEFAULT '',
    `suite_name` String,
    `model_name` LowCardinality(String) DEFAULT '',
    `yaml_file` String DEFAULT '',
    `operation` LowCardinality(String),
    `classification` LowCardinality(String),
    `test_name` String,
    -- XPASS | XFAIL | FALLBACK -- an enablement verdict, not a test outcome.
    `status` LowCardinality(String),
    `input_shapes` String DEFAULT '[]',
    `input_strides` String DEFAULT '[]',
    `input_dtypes` String DEFAULT '[]',
    `triggered_at` DateTime64(3, 'UTC'),
    `ingested_at` DateTime64(3, 'UTC'),
    `tags` String DEFAULT ''
)
ENGINE = ReplacingMergeTree(ingested_at)
PARTITION BY toYYYYMM(triggered_at)
ORDER BY (component, arch, operation, variant_id);
