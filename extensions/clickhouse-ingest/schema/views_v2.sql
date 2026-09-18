-- NOTE: CREATE VIEW IF NOT EXISTS, never CREATE OR REPLACE VIEW. This server rejects
-- OR REPLACE with `System call renameat2() is not supported` (verified on the live
-- instance), so a file using it cannot be applied at all. Re-pointing a view at a new
-- definition therefore needs an explicit DROP VIEW first -- and note that dropping a
-- BASE table silently drops the views over it, so recreate them after any table rebuild.
-- Query interface over schema v2 (artifacts_v2.sql, functional_tests_v2.sql), read by
-- the dashboard UI. Nothing else may read the base tables directly: the two correctness
-- traps below are easy to reintroduce in an ad-hoc query and impossible to see in its
-- output.
--
-- Rationale for every decision below: docs/clickhouse_v2_views.md
--
-- Trap 1 -- tag resolution. A tag MOVES. Joining artifact_results to artifact_tags on
-- tag alone blends every artifact the tag ever pointed at, so a two-week-old `passed`
-- reports today's nightly as green. Every tag-scoped view therefore resolves the tag to
-- ONE artifact_id per (component, arch) with argMax(artifact_id, ts) BEFORE touching
-- results. v_tag_results is the only sanctioned tag->results path.
--
-- Trap 2 -- tier filtering. Tier is a TAG on the test, not a property of the run, so
-- per-case tier filtering is has(tags,'testtype__<tier>') on test_cases;
-- artifact_results.test_type records only what the run was LAUNCHED as, and filtering
-- cases by it silently hides the subsets already present in the rows.
-- The tier relation is NOT transitive: measured on prod, 616 integration cases are not
-- tagged trunk. So NO view here infers one tier from another -- every tier is filtered by
-- its own tag independently. Never add a ladder, tier rank, or tier_satisfies concept:
-- it would claim coverage that does not exist, and tag filtering is correct either way.
--
-- Trap 3 -- arch spelling. artifact_id is the producer key
-- "{component}|{artifact_name}|{id12}|{arch}", so it EMBEDS an arch -- and that segment uses
-- the producer's spelling while the arch COLUMN is canonical. Never parse arch out of the
-- key: the embedded token matches the column on all live rows today, but 400 arch='multi'
-- rows already exist, so the first multi manifest tested on one platform separates them. The
-- column records what actually happened; the key cannot be invalidated by a producer rename.
-- x86 is also written BOTH ways in prod, split by table family:
-- artifacts/artifact_results carry 'amd64' (measured 4,206 rows) while images,
-- pipeline_runs, jenkins_builds and test_runs.platform carry 'x86_64' (273,327). A
-- platform-comparison view that passes the raw value through therefore silently splits x86
-- into two series, one of which reads as "no data". Every arch column these views EXPOSE is
-- normalized through archName(); the stored values are left untouched.
--
-- All views are plain (non-materialized) views: at v2 volumes the base scans are
-- already sub-100ms, and a materialized view would add a second copy that can diverge
-- from a corrected backfill.


-- Tag -> the artifact it points at NOW, one row per (tag, component, arch).
-- The only correct resolution primitive; `is_rolling` is emergent (a tag is rolling iff
-- it has ever pointed at more than one artifact), never stored, so it cannot contradict
-- the rows it summarises.
CREATE VIEW IF NOT EXISTS v_tag_resolution AS
SELECT
    t.tag                              AS tag,
    any(t.tag_family)                  AS tag_family,
    a.component                        AS component,
    if(a.arch IN ('amd64', 'x86'), 'x86_64', a.arch) AS arch,
    argMax(t.artifact_id, t.ts)        AS artifact_id,
    max(t.ts)                          AS resolved_ts,
    count()                            AS promotion_count,
    uniqExact(t.artifact_id) > 1       AS is_rolling  -- within this (component, arch) slot
FROM artifact_tags AS t
INNER JOIN artifacts AS a ON a.artifact_id = t.artifact_id
GROUP BY tag, component, arch;


-- The tag picker. One row per tag, so the UI can list channels without resolving each.
-- arch_list is an array because a dated tag spans all three platforms and the picker
-- shows that span before a platform is chosen.
-- is_rolling is computed PER (component, arch) slot and then OR-ed, not as
-- uniqExact(artifact_id) over the whole tag: a dated tag legitimately holds one artifact
-- per component, so the tag-wide count marks every bundle tag rolling.
CREATE VIEW IF NOT EXISTS v_tag_list AS
SELECT
    t.tag                          AS tag,
    any(t.tag_family)              AS tag_family,
    min(t.ts)                      AS first_ts,
    max(t.ts)                      AS last_ts,
    count()                        AS promotion_count,
    uniqExact(t.artifact_id)       AS artifact_count,
    uniqExact(t.component)         AS component_count,
    arraySort(groupUniqArray(if(t.arch IN ('amd64', 'x86'), 'x86_64', t.arch))) AS arch_list,
    max(slot_artifacts) > 1        AS is_rolling
FROM
(
    SELECT at.tag AS tag, at.tag_family AS tag_family, at.artifact_id AS artifact_id,
           at.ts AS ts, a.component AS component, a.arch AS arch,
           uniqExact(at.artifact_id) OVER (PARTITION BY at.tag, a.component, a.arch)
               AS slot_artifacts
    FROM artifact_tags AS at
    LEFT JOIN artifacts AS a ON a.artifact_id = at.artifact_id
) AS t
GROUP BY tag;


-- Base results view: every run verdict with its artifact's identity attached. The input
-- to the trend and tag views, and the artifact drill-down's own source.
-- artifact_arch vs run_arch are deliberately separate: a 'multi' manifest is tested on
-- one platform, so only run_arch answers "which platform did this pass on".
-- The counters are DERIVED from test_case_runs rather than read off artifact_results, which
-- no longer stores them: a stored copy is a second source of truth that drifts the moment a
-- delta run copies a covering run's cases in. Counted over the run's WHOLE row set (executed
-- plus copied), which is the point of the copy -- a delta run reports its whole tier. Use
-- props['ran_in'] = run_id to count only what this run itself executed.
-- suite_ran distinguishes "the suite never executed" (no case rows at all) from "it ran and
-- regressed" -- state alone cannot, and a 0/0 row would otherwise chart as 0% pass.
CREATE VIEW IF NOT EXISTS v_artifact_results_enriched AS
SELECT
    r.ts             AS ts,
    r.artifact_id    AS artifact_id,
    r.run_id        AS run_id,
    a.component      AS component,
    a.kind           AS kind,
    a.artifact_name  AS artifact_name,
    if(a.arch IN ('amd64', 'x86'), 'x86_64', a.arch) AS artifact_arch,
    if(r.arch IN ('amd64', 'x86'), 'x86_64', r.arch) AS run_arch,
    a.origin         AS origin,
    r.result_kind    AS result_kind,
    r.test_type      AS test_type,
    r.state          AS state,
    c.total_tests    AS total_tests,
    c.passed         AS passed,
    c.failed         AS failed,
    c.errors         AS errors,
    c.skipped        AS skipped,
    r.duration_s     AS duration_s,
    if(c.total_tests > 0, c.passed / c.total_tests, NULL) AS pass_rate,
    c.total_tests > 0 AS suite_ran,
    -- 'running' is advisory only: a crashed run keeps this row until the 90-day TTL. Kept
    -- VISIBLE here (the drill-down should show a live run) but callers that aggregate must
    -- exclude it, as v_tier_trend and v_run_coverage do.
    CAST(r.state = 'running' AS UInt8) AS is_advisory
FROM artifact_results AS r
LEFT JOIN artifacts AS a ON a.artifact_id = r.artifact_id
-- LEFT JOIN, not INNER: a run with no case rows must still appear, with total_tests = 0.
-- That is exactly the "suite never executed" signal, and an INNER JOIN would delete it.
LEFT JOIN (
    SELECT
        run_id,
        count()                                 AS total_tests,
        countIf(status = 'passed')              AS passed,
        countIf(status = 'failed')              AS failed,
        countIf(status = 'error')               AS errors,
        countIf(status = 'skipped')             AS skipped
    FROM test_case_runs
    GROUP BY run_id
) AS c ON c.run_id = r.run_id;


-- Combined functional + performance results for every artifact in one tag.
-- The INNER JOIN is on (tag-resolved artifact_id) -- resolving first is what makes this
-- safe where a direct join on tag is not (Trap 1). One row per run of a member artifact.
CREATE VIEW IF NOT EXISTS v_tag_results AS
SELECT
    tr.tag         AS tag,
    tr.tag_family  AS tag_family,
    tr.component   AS component,
    tr.arch        AS artifact_arch,   -- already canonical via v_tag_resolution
    tr.resolved_ts AS resolved_ts,
    e.artifact_id  AS artifact_id,
    e.artifact_name AS artifact_name,
    e.run_id      AS run_id,
    e.run_arch     AS run_arch,
    e.result_kind  AS result_kind,
    e.test_type    AS test_type,
    e.state        AS state,
    e.total_tests  AS total_tests,
    e.passed       AS passed,
    e.failed       AS failed,
    e.errors       AS errors,
    e.skipped      AS skipped,
    e.duration_s   AS duration_s,
    e.pass_rate    AS pass_rate,
    e.suite_ran    AS suite_ran,
    e.ts           AS ts
FROM v_tag_resolution AS tr
INNER JOIN v_artifact_results_enriched AS e ON e.artifact_id = tr.artifact_id;


-- The membership list behind a tag: which artifacts are in it, with their addresses.
-- Separate from v_tag_results because a tag member with no test run must still be
-- listed -- an inner join to results would hide exactly the untested artifacts the
-- page exists to surface.
CREATE VIEW IF NOT EXISTS v_tag_artifacts AS
SELECT
    tr.tag          AS tag,
    tr.tag_family   AS tag_family,
    tr.component    AS component,
    tr.arch         AS arch,           -- already canonical via v_tag_resolution
    tr.artifact_id  AS artifact_id,
    tr.resolved_ts  AS resolved_ts,
    a.artifact_name AS artifact_name,
    a.kind          AS kind,
    a.origin        AS origin,
    a.props['id12'] AS id12,
    a.sources       AS sources,
    groupArray(f.ref) AS refs
FROM v_tag_resolution AS tr
INNER JOIN artifacts AS a ON a.artifact_id = tr.artifact_id
LEFT JOIN artifact_refs AS f ON f.artifact_id = tr.artifact_id
GROUP BY tag, tag_family, component, arch, artifact_id, resolved_ts,
         artifact_name, kind, origin, id12, sources;


-- Daily trend for the overview page, split by platform so the three arches chart
-- side by side. Grain is one row per (day, tag_family, result_kind, test_type,
-- run_arch, component); the UI aggregates upward, because summing a day's rows is
-- correct but re-deriving a per-component split from a rolled-up row is not.
-- pass_rate is computed from the summed counters, not averaged over runs: averaging
-- rates weights a 3-test run equally with a 3,000-test one.
-- Excludes state='running': see the WHERE clause.
-- One row per TAG, so a rolling channel and its dated alias (nightly + nightly-2026-09-07,
-- same family, same artifact) each contribute a row: summing `runs` double-counts (measured
-- 1,169 distinct runs -> 1,201). Because both duplicate tags sit in ONE family, grouping by
-- tag_family does NOT fix this -- nightly alone sums to 1,063 against 1,039 distinct. Counts
-- here are exact only within a single `tag`; any exact total is uniqExact(run_id) from
-- v_artifact_results_enriched. Never max() over tag_family: it assumes families describe the
-- same runs and silently drops the disjoint ones. pass_rate is unaffected (both sides scale).
CREATE VIEW IF NOT EXISTS v_tier_trend AS
-- One row per (day, tag_family, kind, test_type, arch, component).
-- The tag join is deduped to ONE row per artifact first: rolling and dated tags coexist by
-- design (an artifact carries both `weekly` and `weekly-2026-09-05`), so joining
-- v_tag_resolution directly fans out and counts the same run once per tag. Measured on dev:
-- 1266 resolution rows over 1248 artifacts inflated cases 56638 vs a true 53623.
-- `runs` counts DISTINCT run_id, not result rows -- one run can carry several result rows
-- (15 of 1184 on dev), and a run-count label must not follow the row count.
SELECT
    toDate(d.rts)           AS day,
    d.fam                   AS tag_family,
    e.result_kind           AS result_kind,
    e.test_type             AS test_type,
    e.run_arch              AS run_arch,
    e.component             AS component,
    uniqExact(e.run_id)     AS runs,
    uniqExact(e.artifact_id) AS artifacts,
    uniqExactIf(e.run_id, e.state != 'passed') AS failed_runs,
    sum(e.total_tests)      AS total_tests,
    sum(e.passed)           AS passed,
    sum(e.failed)           AS failed,
    sum(e.errors)           AS errors,
    sum(e.skipped)          AS skipped,
    if(sum(e.total_tests) > 0, sum(e.passed) / sum(e.total_tests), NULL) AS pass_rate,
    avg(e.duration_s)       AS mean_duration_s
FROM
(
    -- Latest resolution per artifact: collapses the rolling/dated pair to one row.
    SELECT artifact_id,
           -- Aliases deliberately differ from the source column names: an output alias equal
           -- to a source column makes ClickHouse resolve the alias inside WHERE and inside the
           -- enclosing aggregate, which fails as ILLEGAL_AGGREGATION.
           argMax(tag_family, resolved_ts) AS fam,
           max(resolved_ts)                AS rts
    FROM v_tag_resolution
    WHERE tag_family IN ('nightly', 'weekly')
    GROUP BY artifact_id
) AS d
INNER JOIN v_artifact_results_enriched AS e ON e.artifact_id = d.artifact_id
-- state='running' is advisory display only and a crashed run leaves a stale row until the
-- TTL reaps it, so it must never contribute to a trend.
WHERE e.state != 'running'
GROUP BY day, tag_family, result_kind, test_type, run_arch, component;


-- Per-case detail for one run, the artifact drill-down's expanded row.
-- ALWAYS filter by run_id; unfiltered this joins both tables in full.
-- `tags` is carried through so the UI can derive tier subsets client-side
-- (has(tags,'testtype__integration')) without a second query -- see Trap 2.
-- When selecting runs by artifact rather than by a known run_id, filter
-- `run_id IN (SELECT ... FROM artifact_results ...)`: the IN form pushes the values down
-- as a key predicate on the (component, run_id, ...) sort key, whereas an INNER JOIN to
-- artifact_results streams the entire table through a hash join. Measured 108x more rows
-- read at 22.5M rows, and the gap grows linearly with the table.
CREATE VIEW IF NOT EXISTS v_case_results AS
SELECT
    cr.run_id      AS run_id,
    cr.test_case_id AS test_case_id,
    cr.component    AS component,
    c.classname     AS classname,
    c.name          AS name,
    c.tags          AS tags,
    cr.status       AS status,
    cr.duration_s   AS duration_s,
    cr.fail_message AS fail_message,
    cr.ts           AS ts
FROM test_case_runs AS cr
LEFT JOIN test_cases AS c
       ON c.test_case_id = cr.test_case_id AND c.component = cr.component;


-- Per-case pass history across runs, for the flaky/regressing panel on the drill-down.
-- Grouped on (component, classname, name) and NOT on test_case_id: `tags` is in the
-- identity hash, so re-tagging a test mints a new id and would split its own history.
CREATE VIEW IF NOT EXISTS v_case_trend AS
SELECT
    cr.component  AS component,
    c.classname   AS classname,
    c.name        AS name,
    toDate(cr.ts) AS day,
    count()       AS runs,
    countIf(cr.status = 'passed') AS passed,
    countIf(cr.status IN ('failed', 'error')) AS failed,
    countIf(cr.status = 'skipped') AS skipped,
    passed / runs AS pass_rate,
    avg(cr.duration_s) AS mean_duration_s,
    anyIf(cr.fail_message, cr.fail_message != '') AS sample_fail_message
FROM test_case_runs AS cr
INNER JOIN test_cases AS c
        ON c.test_case_id = cr.test_case_id AND c.component = cr.component
GROUP BY component, classname, name, day;


-- Per-tier counters for one run, so the UI renders integration/regression/trunk from a
-- single execution. Each tier is counted by ITS OWN tag, independently -- no tier is
-- inferred from another, because the tier relation is not transitive (Trap 2).
-- The tier list is fixed here rather than derived from the tags present: a tier with zero
-- matching cases must still return a row saying zero, or the UI cannot distinguish
-- "this tier did not run" from "this tier is absent from the picker".
CREATE VIEW IF NOT EXISTS v_run_tier_counters AS
SELECT
    cr.run_id AS run_id,
    cr.component AS component,
    tier,
    countIf(cr.status = 'passed')  AS passed,
    countIf(cr.status = 'failed')  AS failed,
    countIf(cr.status = 'error')   AS errors,
    countIf(cr.status = 'skipped') AS skipped,
    countIf(cr.status = 'xfail')   AS xfail,
    countIf(cr.status = 'xpass')   AS xpass,
    -- Kept alongside the split columns: existing callers address xfailed, and an
    -- expected-fail total is the more meaningful figure for a pass-rate label.
    countIf(cr.status IN ('xfail', 'xpass')) AS xfailed,
    count()  AS total,
    if(count() > 0, countIf(cr.status = 'passed') / count(), NULL) AS pass_rate,
    sum(cr.duration_s) AS duration_s
FROM test_case_runs AS cr
INNER JOIN test_cases AS c
        ON c.test_case_id = cr.test_case_id AND c.component = cr.component
ARRAY JOIN ['integration', 'regression', 'trunk', 'unit', 'smoke'] AS tier
WHERE has(c.tags, concat('testtype__', tier))
GROUP BY run_id, component, tier;


-- Completeness of a derived tier report: of the cases tagged for the tier the UI is SHOWING,
-- how many did THIS run actually execute. Required because the tier relation is not
-- transitive -- a trunk run does not necessarily cover every integration case (measured: 616
-- integration cases are not tagged trunk) -- so a derived report can silently claim coverage
-- it does not have. If not_covered > 0 the report is PARTIAL and must be labelled so.
-- Grain is (run_id, component, tier); `tier` is the tier being REPORTED, independent of
-- what the run was launched as. Filter by run_id.
CREATE VIEW IF NOT EXISTS v_tier_report_completeness AS
SELECT
    ran.run_id   AS run_id,
    c.component   AS component,
    tier,
    uniqExact(c.test_case_id)                            AS want_total,
    uniqExactIf(c.test_case_id, has(ran.ids, c.test_case_id)) AS ran_total,
    want_total - ran_total                               AS not_covered,
    if(want_total > 0, ran_total / want_total, NULL)      AS completeness
FROM test_cases AS c
ARRAY JOIN ['integration', 'regression', 'trunk', 'unit', 'smoke'] AS tier
-- The cases each run actually executed, folded to one row per run so the per-tier
-- comparison below is a set membership test rather than a second pass over the fact table.
CROSS JOIN
(
    SELECT run_id, component, groupUniqArray(test_case_id) AS ids
    FROM test_case_runs
    GROUP BY run_id, component
) AS ran
WHERE has(c.tags, concat('testtype__', tier))
  AND c.component = ran.component
GROUP BY run_id, component, tier;

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
