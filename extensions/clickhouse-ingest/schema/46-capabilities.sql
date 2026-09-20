-- Capability analysis: for a subject, is a capability supported on a backend.
--
-- NOT test results, and deliberately not in test_cases/test_case_runs. A model that cannot load
-- is an unsupported model, not a failing test, and run_case_counters_mv buckets on status names
-- -- so folding these in would report every unimplemented operation as a functional failure and
-- corrupt every pass_rate the functional views serve.
--
-- EIGHT v1 TABLES COLLAPSE HERE:
--   model_ops_{suites,variants} x {,_p,_z}          (torch-spyre, arch in the table name)
--   {embedding,generative}_model_spyre_support      (hf-adapters, model type in the table name)
-- Both answer one question -- "for subject S, is capability C supported on backend B" -- with
-- different axes, and both encoded a dimension in the table NAME instead of a column. The v1
-- suites table is gone entirely: its counters are derivable from the variants and already
-- disagreed with them (measured 24 stored spyre_enabled_count vs 27 distinct XPASS operations),
-- so 60-capability-views.sql aggregates instead.
--
-- SPLIT INTO IDENTITY + OBSERVATION, matching test_cases/test_case_runs and
-- benchmarks/benchmark_runs. Measured: 246,292 v1 rows carry only 46,607 distinct identities --
-- each re-observed 4-13 times -- so a flat table repeats the subject, the operation and the
-- input signature on every row.
--
-- FORWARD-ONLY. No backfill: v1 rows cannot produce a run_id (the hash inputs were never
-- recorded), and v1 model_ops last wrote 2026-09-10 while spyre_v2.artifacts began 2026-09-16.


-- WHAT can be supported: the stable identity of one (subject, capability) pair.
CREATE TABLE IF NOT EXISTS capabilities
(
    ts            DateTime DEFAULT now(),

    -- uuid5 over (component, kind, subject, name, disc) -- DERIVED, never minted, so two
    -- writers reach the same id for the same pair with nothing threaded between them. v1's
    -- variant_id was a per-row surrogate instead: 51,356 distinct ids for 51,356 rows, so it
    -- identified nothing and no two runs of one operation ever reconciled.
    capability_id UUID,

    component     LowCardinality(String),
    -- Which analysis this is. The axis v1 put in the table name.
    kind          LowCardinality(String),

    -- The thing analysed (a model), and the capability asked of it (a torch operation for
    -- model_ops, an adapter for model_support). v1 spelled `subject` three ways --
    -- suite_name, model_name, yaml_file -- for only 10 distinct triples; the yaml path is an
    -- incidental and lives in props.
    subject       String,
    name          String,

    tags          Array(LowCardinality(String)),

    -- The signature distinguishing two variants of one operation: 2,788 (operation, test)
    -- pairs expand to 3,707 once input shapes and dtypes are counted. Hashed into
    -- capability_id, so it cannot drift from the identity it defines.
    props         Map(LowCardinality(String), String),

    CONSTRAINT chk_component CHECK component != '',
    CONSTRAINT chk_kind      CHECK kind != '',
    CONSTRAINT chk_name      CHECK name != ''
)
ENGINE = MergeTree()
ORDER BY (component, kind, subject, capability_id);


-- WHETHER it was supported, per run: one row per (capability, run, backend).
CREATE TABLE IF NOT EXISTS capability_runs
(
    ts            DateTime DEFAULT now(),

    -- The only two foreign keys, as in test_case_runs. artifact_id is deliberately NOT stored:
    -- run_id determines it (measured 0 of 1,441 run_ids carrying more than one artifact, which
    -- is the invariant artifact_results' own sort key rests on), so a second copy could only
    -- drift from it.
    --   run_id -> artifact_results.run_id -> .artifact_id -> artifacts -> artifact_tags
    run_id        UUID,
    capability_id UUID,

    -- Denormalized only because it leads the sort key and is a capability_id hash input, so it
    -- cannot disagree. Same call as test_case_runs.component.
    component     LowCardinality(String),
    -- Replaces the _p / _z table suffixes. Canonical spelling (amd64 folds to x86_64).
    arch          LowCardinality(String),

    -- Two INDEPENDENT facts, deliberately two columns. v1 had one: XPASS / XFAIL / FALLBACK,
    -- where FALLBACK is a PASS that ran on the CPU instead. Conflated, every
    -- countIf(status='XPASS') silently undercounted the operations that actually work, which is
    -- the same class of bug that keeps these rows out of test_case_runs.
    status        LowCardinality(String),
    -- Same axis and vocabulary as benchmark_runs.backend, so a capability verdict and a
    -- measurement of the same operation segment alike.
    backend       LowCardinality(String),

    -- Why, when status is not passed. hf-adapters emits a closed 13-value vocabulary
    -- (cpu_load_failed, model_too_large, moe, worker_timeout, ...) worth grouping by, hence
    -- LowCardinality rather than test_case_runs' free-text fail_message. Empty for model_ops,
    -- which emits no reason today.
    fail_reason   LowCardinality(String) DEFAULT '',

    props         Map(LowCardinality(String), String),

    CONSTRAINT chk_status CHECK status IN ('passed','failed','not_implemented')
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(ts)
-- run_id is second, not last: the same artifact re-analysed is a NEW run and both are kept and
-- traceable (an artifact-keyed sort would have silently overwritten the earlier verdict). A
-- re-INGEST of one run collapses, because the key is then identical.
ORDER BY (component, run_id, capability_id, backend);
