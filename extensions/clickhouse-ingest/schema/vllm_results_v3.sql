-- ============================================================================
-- vLLM benchmark results, aligned with upstream pytorch/test-infra
-- benchmark.oss_ci_benchmark_v3 (schema fetched verbatim, 2026-09-10).
--
-- WHY THIS SHAPE. vLLM perf reaches us through vLLM's own bench harness, so the
-- record shape is an UPSTREAM CONTRACT we do not control. Matching it is what lets
-- the PyTorch HUD read our numbers unmodified: its mounted query uses ~16 nested
-- accessors, and the flat fork breaks every one of them with Code 43 tupleElement.
--
-- WHAT WE ADD, AND WHY IT IS THE POINT. `run_id` is OURS, not upstream's. Upstream has
-- no artifact concept -- its `dependencies` Map is a provenance hint, not a content
-- identity -- so an upstream-shaped table alone cannot join to artifact_results and the
-- measured join stays 0 of 133 workflow_ids. This one column is the difference between
-- HUD-compatible and HUD-compatible-AND-joinable. It carries the same v2 uuid5 as
-- artifact_results.run_id, so the artifact/tag pages light up for perf.
--
-- WHAT WE DROP FROM UPSTREAM, DELIBERATELY. The 11-field `runners` tuple is built for
-- GPU fleets (gpu_info, gpu_count, avail_gpu_mem_in_gb) and the servicelab_* ids are
-- Meta-internal. Carrying columns nothing ever writes makes the schema lie about what
-- is known. `inputs`/`dependencies`/`_meta` go the same way until something populates
-- them.
--
-- WHY IT COEXISTS WITH benchmarks/benchmark_runs, WHICH ALSO HOLD PERF. Two shapes for two
-- readers, deliberately, not a migration half-done. This table exists to be read by the
-- PyTorch HUD, so it keeps upstream's nested Tuples verbatim; benchmark_runs is the generic
-- v2 fact (measurements Map, one row per benchmark per run) that the Spyre dashboard's own
-- perf surfaces read, and it serves torch-spyre's kernel benchmarks too, which have no
-- upstream shape at all. The vLLM writer populates BOTH from one parse, keyed by the same
-- run_id, so the duplication costs a second insert and no second source of truth: neither is
-- derived from the other, and a reader picking the wrong one gets fewer columns, never
-- different numbers. Collapsing them would mean either denying the HUD its nested accessors
-- or forcing kernel benchmarks into a GPU-fleet record shape.
--
-- SORT KEY IS OURS, NOT UPSTREAM'S. Upstream orders by
-- (timestamp, head_branch, head_sha, workflow_id, job_id, servicelab_*) -- tuned for
-- HUD's time-series scan. We read by run and by benchmark, so run_id leads. Adopting a
-- schema's COLUMNS does not oblige adopting its access pattern.
-- ============================================================================

CREATE TABLE IF NOT EXISTS vllm_results_v3
(
    -- ── our join key: not upstream, and the reason this table exists ──────────
    -- uuid5(NS, "{source}|{external_run_id}|{arch}|{test_type}"), the SAME value
    -- artifact_results.run_id carries. On the Jenkins path params.RUN_ID is ALREADY
    -- this uuid -- pass it VERBATIM; re-hashing an already-hashed id mints a third
    -- identity that joins to nothing.
    run_id       UUID,

    -- ── upstream metadata ────────────────────────────────────────────────────
    timestamp      Int64,                                  -- epoch millis, as upstream
    schema_version LowCardinality(String) DEFAULT 'v3',
    -- The BENCHMARK name, not a constant. The fork wrote 'spyre_e2e_benchmark' on all
    -- 6,611 rows, collapsing every benchmark into one name and making per-benchmark
    -- trends impossible.
    name         String,

    -- ── about the change ─────────────────────────────────────────────────────
    repo         LowCardinality(String) DEFAULT 'torch-spyre/spyre-inference',
    head_branch  String,
    -- A real value. The fork left this blank on 6,611/6,611 rows and hid the sha in
    -- `extra`, which is why the HUD's commit picker shows nothing.
    head_sha     String,
    workflow_id  Int64,
    run_attempt  UInt32 DEFAULT 0,
    job_id       Int64 DEFAULT 0,

    -- ── upstream nested records ──────────────────────────────────────────────
    -- Kept as Tuples so the HUD's nested accessors resolve unchanged.
    runners      Array(Tuple(name String, type String)),
    benchmark    Tuple(name String, mode String, dtype String, extra_info Map(String, String)),
    model        Tuple(name String, type String, backend String, origins Array(String)),
    -- benchmark_values is an ARRAY: vLLM reports every iteration. The fork collapsed it
    -- to a single `actual`, destroying the samples at ingest -- so no variance, no
    -- percentile recomputation, no outlier detection was ever possible downstream.
    -- target_value is Float32 as upstream types it, NOT String (the earlier nested
    -- attempt used String, which breaks any HUD query that casts).
    metric       Tuple(name String, benchmark_values Array(Float32), target_value Float32,
                       extra_info Map(String, String)),
    -- Test-shape discriminators: tp / input len / output len. Present in the fork only
    -- inside the test_name string, so two shapes of the same benchmark were
    -- indistinguishable.
    inputs       Map(String, Tuple(dtype String, extra_info Map(String, String))),
    -- Which torch-spyre produced the number. Upstream's shape; blank until wired.
    dependencies Map(String, Tuple(repo String, branch String, sha String, version String)),

    -- ── ours ─────────────────────────────────────────────────────────────────
    -- The same CONCEPT as benchmarks/benchmark_runs.props -- an open key/value bag -- under the
    -- name upstream already uses, and kept rather than renamed for the same reason the Tuples
    -- above are kept: the HUD reads `extra` by name. Contents overlap by design (both carry
    -- model, device, arch, head_sha), because the vLLM writer fills both from one parse; the
    -- v2 pair is not the source of this and neither is derived from the other.
    -- Map, not a JSON String: the fork's `extra` was a String, so every read paid a
    -- JSONExtract and no key could ever be indexed.
    extra        Map(LowCardinality(String), String),

    -- A blank run_id is unjoinable, which downstream reads as "no perf ran" -- refuse it
    -- at write time rather than discover it in a chart.
    CONSTRAINT chk_run_id CHECK run_id != toUUID('00000000-0000-0000-0000-000000000000'),
    CONSTRAINT chk_name   CHECK name != ''
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(toDateTime(intDiv(timestamp, 1000)))
-- run_id leads: every read starts from a run or an artifact that resolves to one.
-- `name` second so a single benchmark's history prunes within a run.
ORDER BY (run_id, name, timestamp);
