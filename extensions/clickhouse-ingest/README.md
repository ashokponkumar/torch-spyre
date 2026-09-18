# spyre-clickhouse-ingest

The schema-v2 ClickHouse schema, derived identity and write path, shared by the Spyre CI ingests.

## Why it is a library

Every id here is **derived, never minted**: the product ingests and the Jenkins-side writer must
reach the same uuid for the same run without coordinating. A second copy that drifts by one
normalisation step produces ids that silently never join — no error, just missing data.

## Install

No PyPI or Artifactory publish. Every consumer installs it straight from the repo, pinned:

```
uv run --no-project \
  --with "git+https://github.com/torch-spyre/torch-spyre@<tag>#subdirectory=extensions/clickhouse-ingest" \
  ...
```

Verified on a build node with the same `uv run --no-project --with` form the baked-image ingest
uses. Pin a tag, not `@main`.

## Layout

| module | contents |
|---|---|
| `schema.py` | the table model: columns, order, `qualified()` |
| `identity.py` | `v2_run_id`, `v2_test_case_id`, `v2_component`, `v2_canonical_arch` |
| `client.py` | `get_client`, `v2_database`, `v2_tables_present` |
| `v2_writer.py` | `insert_v2`, `v2_already_ingested` |
| `junit.py` | JUnit helpers + CI run-coordinate resolution |
| `hw_parse.py` | GHA log → `hw_failure_diagnostics` records (RAS events, phases, pytest counts) |
| `hw_schema.py` | `hw_failure_diagnostics` columns + its `ADD COLUMN IF NOT EXISTS` migration |
| `hw_diagnostics.py` | `build_row`/`insert_rows` for `hw_failure_diagnostics` |
| `gha_logs.py` | fetching GHA job logs via `gh`, with transient-5xx retry |

## What is deliberately NOT here

- **v1 write paths whose TARGET TABLE differs per repo.** `hf_test_runs` vs `test_runs` cannot share
  a writer, so those stay per-repo until v1 is retired. Name divergence is the blocker, not the v1
  generation as such: `hw_failure_diagnostics` is the same table with the same columns in every repo,
  which is why its parse/ingest lives here despite being v1.
- **A dependency on `torch_spyre`.** Installing that to obtain a schema module would pull
  torch/numpy/ortools, and ortools has no ppc64le/s390x wheel — the ingest would break on p/z.

## Changing an identity function

Don't, without a migration. Every id ever written derives from these functions and the namespace
`cb0af9bf-2858-5eab-9211-f51190531bf3`. `tests/test_identity_golden.py` pins the contract with
golden values; if a change makes those fail, it invalidates historical rows.
