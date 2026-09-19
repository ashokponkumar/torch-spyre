-- Views spanning BOTH families, which is why they are neither in 50- nor 51-: v_run_coverage
-- joins artifact_results to test_case_runs to answer "did the leg that produced this artifact
-- actually report cases". Apply last -- it needs every table above.

-- Coverage honesty. artifact_results is a sparse junction (measured 7.3% of runs on prod),
-- so a chart drawn from it is a sample, not a census. This view lets the UI LABEL that
-- rather than hide it: a page showing 8% coverage must say so.
-- Grain is (day, arch) because coverage is not uniform -- it varies by platform and over
-- time as writers are rolled out, and a single global percentage would misdescribe both.
-- Both gaps are reported, because they mean different things: a run with cases but no
-- junction row is invisible to every artifact page, while a junction row with no cases is
-- a suite that never reported (or never ran) and would chart as a silent zero.
CREATE VIEW IF NOT EXISTS v_run_coverage AS
SELECT
    day,
    arch,
    uniqExact(run_id)                              AS runs,
    uniqExactIf(run_id, in_results AND in_cases)   AS runs_linked,
    uniqExactIf(run_id, NOT in_results)            AS runs_missing_result,
    uniqExactIf(run_id, NOT in_cases)              AS runs_missing_cases,
    if(uniqExact(run_id) > 0,
       uniqExactIf(run_id, in_results AND in_cases) / uniqExact(run_id),
       NULL)                                        AS coverage
FROM
(
    -- Every run known from EITHER side. A run present in only one table is exactly the gap
    -- being measured, so neither table alone can enumerate the denominator.
    SELECT
        min(day)   AS day,
        argMax(arch, arch != '') AS arch,
        run_id,
        max(in_results) AS in_results,
        max(in_cases)   AS in_cases
    FROM
    (
        -- state='running' excluded: a crashed run's stale row would otherwise count as
        -- covered, overstating exactly the number this view exists to report honestly.
        SELECT toDate(ts) AS day, if(arch IN ('amd64', 'x86'), 'x86_64', arch) AS arch,
               run_id, 1 AS in_results, 0 AS in_cases
        FROM artifact_results
        WHERE state != 'running'
        UNION ALL
        SELECT toDate(min(ts)) AS day, '' AS arch, run_id, 0 AS in_results, 1 AS in_cases
        FROM test_case_runs
        GROUP BY run_id
    )
    GROUP BY run_id
)
GROUP BY day, arch;

-- ---------------------------------------------------------------------------
-- Indexes. One only, and it is measured; see docs/clickhouse_v2_views.md.
-- ---------------------------------------------------------------------------

-- test_case_runs is ORDER BY (component, run_id, test_case_id), so a run_id-only lookup
-- prunes to parts but NOT to granules: run_id is the second sort-key column and its values
-- scatter across every component prefix. Measured at 53,623 rows and again at 22.5M, the
-- index halves-to-sixths the rows read (5.0x and 5.3-6.0x respectively).
-- It buys NO wall-time improvement: measured a statistical tie at 22.5M rows and ~1 ms WORSE
-- at 20M, because the bloom probe is per-granule CPU on the critical path. Under concurrency
-- it is worse at the TAIL (p90 5.14 ms vs 3.83 ms at 24 threads) -- the probe does not
-- amortise, so "it will pay off once the system is busy" is measurably false, not merely
-- unproven. Retained ONLY for the 5-6x rows-read reduction at 0.007% of table size; do not
-- cite it as a latency optimisation. Whether rows-read eventually dominates is an
-- EXTRAPOLATION to re-measure at 100M+ rows, not something the data shows. If tail latency
-- under load matters more than scan volume here, dropping this index is defensible.
-- PRECONDITION: the index only prunes if a run_id's rows are CONTIGUOUS in the sort order,
-- which requires one run to belong to one component. True by construction (run_id is uuid5
-- over one product's run) and verified on prod: all 338,142 runs are single-component. If it
-- is ever violated the index still materializes to a plausible size and prunes NOTHING --
-- measured on a 400k-row probe, 8,192 rows read single-component vs 40,960 across 5, same
-- index and same volume. Worth an ingest-side assertion, since the failure is invisible here.
-- Do NOT add `component` to a run_id predicate as a tuning step: a run_id belongs to
-- exactly one component, so it carries no information the index has not already used.
-- Rejected as unjustified: any index on artifact_results (1,184 rows) or test_cases
-- (2,400 rows) -- both scan fully in 2-3 ms, so an index costs more than it saves.
-- ADD INDEX covers only parts written AFTER it, so on an existing or rebuilt table the index
-- exists but indexes nothing until materialized. Deployers must follow this with:
--   ALTER TABLE test_case_runs MATERIALIZE INDEX idx_run_id SETTINGS mutations_sync = 2;
-- mutations_sync=2 waits for completion, without which a benchmark measures a half-built index.
ALTER TABLE test_case_runs
    ADD INDEX IF NOT EXISTS idx_run_id run_id TYPE bloom_filter(0.01) GRANULARITY 1;
