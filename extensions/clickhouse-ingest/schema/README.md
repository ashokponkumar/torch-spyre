# v2 schema DDL

The ClickHouse DDL for the v2 tables this package writes.

## Why it lives here

`schema.py` models these tables as data — columns, order, CHECK-constraint vocabularies — and
every row this package inserts is ordered through that model. Until now the DDL itself lived in
another repo (`spyre-frameworks/pipelines/clickhouse/`), so the *shape* was declared in one place
and *modelled* in another, with nothing but prose comments tying them together.

That split is what let prod drift: `spyre_v2.benchmarks` was missing the `component` column that
`schema.py` requires and that leads the `benchmark_id` hash, and nothing failed — both tables were
0 rows, so the gap only surfaced when a writer was finally pointed at them. Co-locating the DDL
with the model means a column added to one is reviewed beside the other.

Only the benchmark pair has moved so far. `functional_tests_v2.sql`, `artifacts_v2.sql`,
`views_v2.sql`, `ci_events_v2.sql` and `vllm_results_v3.sql` are still in spyre-frameworks; the
`Source of truth:` comments in `schema.py` name them.

## Applying it

**Nothing applies these files automatically.** There is no migration runner and no
`schema_migrations` ledger for the v2 databases — every statement here has been applied by hand.
So before writing to a v2 table, `DESCRIBE TABLE` it on the target server rather than trusting
this directory or `schema.py`.

Two mechanical gotchas, both hit while applying the benchmark pair:

- A `MergeTree` `ORDER BY` is fixed at creation, so adding a column to a sort key means
  DROP+CREATE, not `ALTER`.
- `MODIFY COLUMN` cannot convert `Float64` to `Array(Float64)` (Code 53, "same-dimensional Array,
  Map or String types") — that change is also a DROP+CREATE, or a rewrite through a temp table on
  a populated one.

The 5 `v_benchmark_*` views read the pair, and dropping a base table silently drops its views.
Drop and recreate them explicitly, `v_benchmark_results_enriched` first, since the other four
select from it.

## State

`spyre_v2` on the prod server matches these files, verified column-for-column including the view
signatures. Staging `spyre_v2_next` does **not**: it has `component` but still has the
pre-widening `Map(..., Float64)` measurements and no `samples` column.
