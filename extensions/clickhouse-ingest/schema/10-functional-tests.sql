-- Functional-test results, schema v2. Zero backward compatibility with the v1
-- test_runs/test_cases/run_properties set (and their hf_/si_ mirrors): v2 is a
-- clean break on fresh data, read by a new dashboard layer.
--
-- Rationale for every decision below: docs/clickhouse_v2_functional_tests_schema.md
--
-- The organising invariant: a column lives on the fact only if it is an input to
-- the fact's own key hash (so it CANNOT disagree with its source of truth) AND a
-- leading ORDER BY column (so it earns index pruning). `component` passes both.
-- Everything else run-scoped resolves through run_id, and everything else
-- test-scoped through test_case_id -- so there is exactly one home per fact.
--
-- One table pair serves all products; `component` replaces the hf_/si_ prefixed
-- copies. artifact_results (artifacts_v2.sql) is the run record: run_id, arch,
-- test_type, state and the run-level counters live there, not here.
--
-- Bag-column convention, uniform across artifacts_v2.sql and this file:
--   `props` = Map(LowCardinality(String), String), open-ended, NEVER in a key --
--             extend it freely, no DDL and no identity churn.
--   `tags`  = Array(LowCardinality(String)), a SET, and IN the identity hash --
--             writing to it mints a new id, so it must be sorted before hashing.
-- The names differ because the guarantees differ; do not converge them.

CREATE TABLE IF NOT EXISTS test_cases
(
    ts           DateTime DEFAULT now(),

    -- Content hash: uuid5 over (component, classname, name, sorted(tags)).
    -- Sorting is REQUIRED: the source order is incidental, so two writers emitting
    -- the same tags in a different order must not mint different identities.
    -- Deriving identity rather than minting uuid4 per ingest is what lets the same
    -- test reconcile across runs. Because `tags` is IN the hash, re-tagging a test
    -- mints a NEW identity -- so trend queries group on
    -- (component, classname, name), never on test_case_id.
    test_case_id UUID,

    component    LowCardinality(String),
    classname    String,
    name         String,

    -- Replaces the run_properties EAV table (misnamed: it was keyed on case_id, not
    -- v1's run_id, and its one prop_name was literally 'tag'). Array, NOT Map: these are
    -- pytest tags in `namespace__value` form and a namespace repeats -- testtype
    -- carries up to 5 values on 91.7% of cases, so a Map would silently keep one and
    -- drop the rest. Filter with has()/hasAny().
    -- Named `tags`, not `props`: this is a SET that feeds test_case_id, so writing to
    -- it mints a new identity. `props` (every other table) is a Map outside all keys.
    tags         Array(LowCardinality(String)),

    CONSTRAINT chk_component CHECK component != '',
    CONSTRAINT chk_name      CHECK name != ''
)
ENGINE = MergeTree()
ORDER BY (component, test_case_id);


CREATE TABLE IF NOT EXISTS test_case_runs
(
    ts           DateTime DEFAULT now(),

    -- The only two foreign keys. run_id is uuid5 over
    -- (source, external_run_id, arch, test_type) -- DERIVED by every writer from values it
    -- already holds, with no cross-job threading contract, which is what retires v1's
    -- minted run_id and runner_run_id. Same name as v1's column, deliberately: v2 is a
    -- separate database, so there is nothing to disambiguate from, and <thing>_id is the
    -- rule everywhere else here (artifact_id, test_case_id, benchmark_id).
    run_id      UUID,
    test_case_id UUID,

    -- Denormalized ONLY because it is a test_case_id hash input (cannot disagree)
    -- and the leading sort key: per-component queries read 1.7% of the table
    -- instead of all of it, and leading-key grouping makes the table smaller.
    component    LowCardinality(String),

    status       LowCardinality(String),
    duration_s   Float32,
    fail_message String DEFAULT '',

    -- Per-execution incidentals only. Anything run-scoped belongs on
    -- artifact_results; anything test-scoped on test_cases.tags.
    props        Map(LowCardinality(String), String),

    CONSTRAINT chk_status CHECK status IN
        ('passed','failed','error','skipped','xfail','xpass')
)
ENGINE = MergeTree()
-- Monthly parts exist for retention (cheap DROP PARTITION), NOT for query pruning:
-- the ORDER BY prefix already prunes, and PARTITION BY component measured slower.
PARTITION BY toYYYYMM(ts)
ORDER BY (component, run_id, test_case_id);
