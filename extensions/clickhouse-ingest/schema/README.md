# v2 schema DDL

The ClickHouse DDL for the v2 tables this package writes and reads.

| file | declares |
|---|---|
| `functional_tests_v2.sql` | `test_cases`, `test_case_runs` |
| `benchmarks_v2.sql` | `benchmarks`, `benchmark_runs` |
| `artifacts_v2.sql` | `artifacts`, `artifact_refs`, `artifact_tags`, `artifact_results` |
| `vllm_results_v3.sql` | `oss_ci_benchmark_v3`, `oss_ci_benchmark_metadata` + their MVs (upstream-aligned vLLM perf, projected from `benchmark_runs`) |
| `ci_events_v2.sql` | `jenkins_agents` |
| `views_v2.sql` | 11 `v_*` views over the functional and artifact tables |
| `benchmark_views_v2.sql` | 5 `v_benchmark_*` views |

## Why it lives here

`schema.py` models these tables as data — columns, order, CHECK-constraint vocabularies — and
every row this package inserts is ordered through that model. Until now the DDL itself lived in
another repo (`spyre-frameworks/pipelines/clickhouse/`), so the *shape* was declared in one place
and *modelled* here, with nothing but prose comments tying them together.

That split is what let prod drift: `spyre_v2.benchmarks` was missing the `component` column
`schema.py` requires and that leads the `benchmark_id` hash, and nothing failed — both tables were
0 rows, so the gap only surfaced when a writer was finally pointed at them. Co-locating the DDL
with the model means a column added to one is reviewed beside the other.

## Applying it

**Nothing applies these files automatically.** There is no migration runner and no
`schema_migrations` ledger for the v2 databases — every statement here has been applied by hand.
So before writing to a v2 table, `DESCRIBE TABLE` it on the target server rather than trusting
this directory or `schema.py`.

Three mechanical gotchas, all hit while applying the benchmark pair:

- A `MergeTree` `ORDER BY` is fixed at creation, so adding a column to a sort key means
  DROP+CREATE, not `ALTER`.
- `MODIFY COLUMN` cannot convert `Float64` to `Array(Float64)` (Code 53, "same-dimensional Array,
  Map or String types") — that change is also a DROP+CREATE, or a rewrite through a temp table on
  a populated one.
- Views must be dropped and recreated, not `CREATE OR REPLACE`d (the server rejects it:
  `renameat2() is not supported`). Dropping a base table silently drops its views, so recreate
  them explicitly, base view first — `v_benchmark_results_enriched` before the four that select
  from it.

Several comments in these files cite row counts and percentages measured when the statement was
written. They are evidence for a design decision, not live figures; re-measure before relying on
one.

## State

On the **prod** server, `spyre_v2` matches these files for every table except one, verified
column-for-column (including view signatures for the benchmark views):

- The `oss_ci_benchmark_*` pair **does not exist in prod** — the file is the intended shape, not
  a deployed one. It was verified on the dev server: applying it and inserting one
  `benchmark_runs` row propagated through both materialized views, and a live HUD read the
  result with upstream's own queries unmodified.
- Staging `spyre_v2_next` has `component` but **not** the widened `measurements` or the `samples`
  column, so staging and prod differ on the benchmark pair.
