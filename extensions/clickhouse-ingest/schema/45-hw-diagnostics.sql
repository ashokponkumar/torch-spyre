-- Hardware-failure diagnostics: one row per (workflow, run, suite, attempt).
--
-- The parse/ingest logic lives in spyre_clickhouse_ingest.hw_parse / .hw_diagnostics and is
-- shared by torch-spyre, hf-adapters and spyre-inference. This file is the table those writers
-- assume; until now the table had no checked-in DDL at all, and its only migration path was
-- hw_schema.ensure_extra_columns' ALTER ... ADD COLUMN IF NOT EXISTS on every ingest. That still
-- runs (older deployments need it) but is no longer the definition.
--
-- ORDER BY is (workflow, run_id, suite_name, attempt) and deliberately NOT tuple(). The live
-- table was created unsorted, so hw_schema.already_ingested -- which runs once per ingest and
-- filters exactly (run_id, workflow) -- reads every row: measured 840,032 of 840,032 on prod.
-- A sort key is immutable, so fixing the live table means DROP+CREATE or a rewrite through a
-- temp table; this file is the shape a fresh database gets.
--
-- attempt is in the key because retries are the point: a flaky card shows up as attempt 2+ of
-- the same (workflow, run, suite), and the retry_trigger/pod_level_retry columns only make sense
-- read alongside the attempt they belong to.
--
-- No audit_uuid/audit_timestamp: the live table carries both, but no writer sets them, so they
-- are unwritten defaults rather than data. Same call as v2 jenkins_agents.
--
-- run_id is a String, not a UUID: it is the GHA run id (or the Jenkins build key), what the
-- producer actually holds. It does NOT join test_case_runs.run_id -- that is a uuid5 over
-- (source, external_run_id, arch, test_type) -- so a reader wanting both must derive the uuid5
-- from this id, not equate them.
CREATE TABLE IF NOT EXISTS hw_failure_diagnostics
(
    `run_id` String,
    `workflow` LowCardinality(String) DEFAULT '',
    `branch` LowCardinality(String) DEFAULT '',
    `commit_sha` String DEFAULT '',
    `suite_name` String,
    `attempt` UInt8,
    `total_attempts` UInt8,
    `ingested_at` DateTime64(6, 'UTC'),
    `outcome` LowCardinality(String),
    `exit_code` Nullable(Int32),
    `failure_reason` LowCardinality(String),
    `failure_phase` LowCardinality(String),
    `retry_trigger` String,
    `failure_reason_detail` String DEFAULT '{}',
    `ras_code` LowCardinality(String),
    `ras_name` String,
    `ras_description` String,
    `ras_action` LowCardinality(String),
    `ras_category` LowCardinality(String),
    `ras_severity` LowCardinality(String),
    `ras_message` String,
    `ras_events_json` String DEFAULT '[]',
    `node_name` LowCardinality(String),
    `pci_device` LowCardinality(String),
    `aiu_world_rank0` LowCardinality(String),
    `card_serial` String,
    `chip_ecid_raw` String,
    `chip_wafer_id` LowCardinality(String),
    `chip_mfg_x` String,
    `chip_mfg_y` String,
    `chip_chipy` String,
    `chip_chipx` String,
    `first_error_ts` Nullable(DateTime64(6, 'UTC')),
    `attempt_start_ts` Nullable(DateTime64(6, 'UTC')),
    `tests_collected` UInt32,
    `tests_passed` UInt32,
    `tests_failed` UInt32,
    `tests_error` UInt32,
    `stall_max_secs` UInt32,
    `run_link` String DEFAULT '',
    `pod_level_retry` Bool DEFAULT false
)
ENGINE = MergeTree
ORDER BY (workflow, run_id, suite_name, attempt)
