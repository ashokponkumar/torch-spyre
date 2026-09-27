-- Adds audit_uuid / audit_timestamp to tables created before they were part of the DDL.
-- An added column's DEFAULT is evaluated at READ time for older parts, so the UPDATE writes
-- real values: the row's own insert time, and one stable v7 uuid per existing row.

ALTER TABLE test_cases
    ADD COLUMN IF NOT EXISTS audit_uuid UUID DEFAULT generateUUIDv7(),
    ADD COLUMN IF NOT EXISTS audit_timestamp DateTime64(3) DEFAULT now64(3);
ALTER TABLE test_cases UPDATE audit_timestamp = toDateTime64(ts, 3), audit_uuid = generateUUIDv7()
    WHERE 1 SETTINGS mutations_sync = 2, allow_nondeterministic_mutations = 1;

ALTER TABLE test_case_runs
    ADD COLUMN IF NOT EXISTS audit_uuid UUID DEFAULT generateUUIDv7(),
    ADD COLUMN IF NOT EXISTS audit_timestamp DateTime64(3) DEFAULT now64(3);
ALTER TABLE test_case_runs UPDATE audit_timestamp = toDateTime64(ts, 3), audit_uuid = generateUUIDv7()
    WHERE 1 SETTINGS mutations_sync = 2, allow_nondeterministic_mutations = 1;

ALTER TABLE artifacts
    ADD COLUMN IF NOT EXISTS audit_uuid UUID DEFAULT generateUUIDv7(),
    ADD COLUMN IF NOT EXISTS audit_timestamp DateTime64(3) DEFAULT now64(3);
ALTER TABLE artifacts UPDATE audit_timestamp = toDateTime64(ts, 3), audit_uuid = generateUUIDv7()
    WHERE 1 SETTINGS mutations_sync = 2, allow_nondeterministic_mutations = 1;

ALTER TABLE artifact_refs
    ADD COLUMN IF NOT EXISTS audit_uuid UUID DEFAULT generateUUIDv7(),
    ADD COLUMN IF NOT EXISTS audit_timestamp DateTime64(3) DEFAULT now64(3);
ALTER TABLE artifact_refs UPDATE audit_timestamp = toDateTime64(ts, 3), audit_uuid = generateUUIDv7()
    WHERE 1 SETTINGS mutations_sync = 2, allow_nondeterministic_mutations = 1;

ALTER TABLE artifact_tags
    ADD COLUMN IF NOT EXISTS audit_uuid UUID DEFAULT generateUUIDv7(),
    ADD COLUMN IF NOT EXISTS audit_timestamp DateTime64(3) DEFAULT now64(3);
ALTER TABLE artifact_tags UPDATE audit_timestamp = toDateTime64(ts, 3), audit_uuid = generateUUIDv7()
    WHERE 1 SETTINGS mutations_sync = 2, allow_nondeterministic_mutations = 1;

ALTER TABLE artifact_results
    ADD COLUMN IF NOT EXISTS audit_uuid UUID DEFAULT generateUUIDv7(),
    ADD COLUMN IF NOT EXISTS audit_timestamp DateTime64(3) DEFAULT now64(3);
ALTER TABLE artifact_results UPDATE audit_timestamp = toDateTime64(ts, 3), audit_uuid = generateUUIDv7()
    WHERE 1 SETTINGS mutations_sync = 2, allow_nondeterministic_mutations = 1;

ALTER TABLE benchmarks
    ADD COLUMN IF NOT EXISTS audit_uuid UUID DEFAULT generateUUIDv7(),
    ADD COLUMN IF NOT EXISTS audit_timestamp DateTime64(3) DEFAULT now64(3);
ALTER TABLE benchmarks UPDATE audit_timestamp = toDateTime64(ts, 3), audit_uuid = generateUUIDv7()
    WHERE 1 SETTINGS mutations_sync = 2, allow_nondeterministic_mutations = 1;

ALTER TABLE benchmark_runs
    ADD COLUMN IF NOT EXISTS audit_uuid UUID DEFAULT generateUUIDv7(),
    ADD COLUMN IF NOT EXISTS audit_timestamp DateTime64(3) DEFAULT now64(3);
ALTER TABLE benchmark_runs UPDATE audit_timestamp = toDateTime64(ts, 3), audit_uuid = generateUUIDv7()
    WHERE 1 SETTINGS mutations_sync = 2, allow_nondeterministic_mutations = 1;

ALTER TABLE jenkins_agents
    ADD COLUMN IF NOT EXISTS audit_uuid UUID DEFAULT generateUUIDv7(),
    ADD COLUMN IF NOT EXISTS audit_timestamp DateTime64(3) DEFAULT now64(3);
ALTER TABLE jenkins_agents UPDATE audit_timestamp = toDateTime64(ts, 3), audit_uuid = generateUUIDv7()
    WHERE 1 SETTINGS mutations_sync = 2, allow_nondeterministic_mutations = 1;

ALTER TABLE hw_failure_diagnostics
    ADD COLUMN IF NOT EXISTS audit_uuid UUID DEFAULT generateUUIDv7(),
    ADD COLUMN IF NOT EXISTS audit_timestamp DateTime64(3) DEFAULT now64(3);
ALTER TABLE hw_failure_diagnostics UPDATE audit_timestamp = toDateTime64(ingested_at, 3), audit_uuid = generateUUIDv7()
    WHERE 1 SETTINGS mutations_sync = 2, allow_nondeterministic_mutations = 1;

ALTER TABLE capabilities
    ADD COLUMN IF NOT EXISTS audit_uuid UUID DEFAULT generateUUIDv7(),
    ADD COLUMN IF NOT EXISTS audit_timestamp DateTime64(3) DEFAULT now64(3);
ALTER TABLE capabilities UPDATE audit_timestamp = toDateTime64(ts, 3), audit_uuid = generateUUIDv7()
    WHERE 1 SETTINGS mutations_sync = 2, allow_nondeterministic_mutations = 1;

ALTER TABLE capability_runs
    ADD COLUMN IF NOT EXISTS audit_uuid UUID DEFAULT generateUUIDv7(),
    ADD COLUMN IF NOT EXISTS audit_timestamp DateTime64(3) DEFAULT now64(3);
ALTER TABLE capability_runs UPDATE audit_timestamp = toDateTime64(ts, 3), audit_uuid = generateUUIDv7()
    WHERE 1 SETTINGS mutations_sync = 2, allow_nondeterministic_mutations = 1;
