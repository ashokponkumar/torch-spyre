-- Benchmark schema v2: a dimension + fact pair, the same split test_cases /
-- test_case_runs uses. Read functional_tests_v2.sql first -- the identity rules,
-- the props/tags convention and the run_id contract are stated there and are not
-- repeated here.
--
-- Scope: torch-spyre op / kernel / model benchmarks (spyre-perf-suite, plus the
-- sendnn and CPU baselines it measures against -- see `backend`). vLLM /
-- spyre-inference perf deliberately stays in results_v3 + run_metadata: those are
-- the schema the upstream PyTorch HUD queries read, and reshaping them would fork
-- SQL we re-mount from upstream on every bump. It reaches this layer by carrying
-- run_id, joining through artifact_results like everything else.
--
-- Replaces six v1 tables: the v1 benchmark_runs + perf_benchmarks + perf_kernels,
-- sendnn_runs + sendnn_benchmarks, and loz_system_performance_vllm (41 wide
-- columns, 0 rows, never written -- the clearest evidence against a wide fact).

CREATE TABLE IF NOT EXISTS benchmarks
(
    ts           DateTime DEFAULT now(),

    -- uuid5(NAMESPACE, "name|sorted(tags)|record_type,config_name,input_shapes,
    -- run_mode,kernel_name,is_total"). tags are IN the hash, so they MUST be sorted
    -- before hashing or the same benchmark mints two identities.
    -- The discriminators are in the hash, not merely in props, because one
    -- operation_name occurs at several record_types in prod (granite as model AND
    -- op; matmul, attention likewise) and config_name separates the granite
    -- variants -- hashing name+tags alone merges different benchmarks into one.
    benchmark_id UUID,
    -- Which producer's suite this benchmark belongs to -- the same axis test_cases uses,
    -- and for the same reason. torch-spyre's op harness and vLLM's bench are separate
    -- suites sharing one namespace: they collide the day either names a benchmark
    -- `latency` or `matmul`, and because benchmark_id is a CONTENT hash that collision
    -- silently merges two unrelated benchmarks into one trend line. In the hash, so the
    -- same name under two components is two identities.
    component    LowCardinality(String),

    name         String,
    tags         Array(LowCardinality(String)),

    -- What distinguishes one benchmark from another, per producer: record_type
    -- (op|model|kernel), config_name, input_shapes, run_mode, kernel_name,
    -- is_total, batch_size, prompt_length, custom_op_file.
    -- These are CONFIGURATION, not measurement: batch_size is set on 40/40 model
    -- rows and 0/297 op rows, i.e. an identity discriminator, not a number
    -- measured. A Map keeps one dimension serving producers whose identity tuples
    -- disagree.
    props        Map(LowCardinality(String), String),

    CONSTRAINT chk_component CHECK component != '',
    CONSTRAINT chk_name      CHECK name != ''
)
ENGINE = MergeTree()
-- name leads, not benchmark_id: every read picks a benchmark BY NAME (the perf trend and
-- comparison surfaces), and benchmark_id is only ever arrived at through it or through a
-- join. Measured 12 names over 1,819 ids, so name-first prunes and also clusters a
-- benchmark's variants together. benchmark_id stays in the key to keep the row unique.
-- component leads, matching test_cases: every read scopes to a producer first, and the
-- LowCardinality prefix prunes before the name range scan.
ORDER BY (component, name, benchmark_id);

-- benchmark_runs, matching test_case_runs: <dimension>_runs is the convention for the
-- observation half of a pair. The v1 table of this name lives in the `spyre` database and
-- does not collide -- v2 is a separate database, which is what frees the clean name.
CREATE TABLE IF NOT EXISTS benchmark_runs
(
    ts           DateTime DEFAULT now(),

    -- The only two foreign keys. run_id carries ALL run context (arch, branch,
    -- commit, pr, tag) through artifact_results; benchmark_id carries ALL benchmark
    -- identity. Nothing run-scoped or benchmark-scoped is copied here: a column
    -- earns a place on a fact only if it is an input to that fact's own key hash
    -- (so it cannot disagree) AND a leading ORDER BY column (so it earns pruning).
    run_id      UUID,
    benchmark_id UUID,

    -- Which implementation produced these numbers. This is the axis that makes
    -- perf_kernels coherent: its torch_spyre_ms and sendnn_ms are co-populated in
    -- 0 of 5,722 rows, because they were never one measurement -- they are the same
    -- kernel on two backends. As rows they compare by self-join on benchmark_id,
    -- and the stored `ratio` becomes derived instead of a third column that can
    -- disagree with the two it divides.
    -- Carried on the fact table too, exactly as test_case_runs carries it, and leading
    -- the sort key so a per-producer read prunes instead of scanning every run.
    component    LowCardinality(String),
    backend      LowCardinality(String),

    -- Metric key -> its SAMPLES, keys verbatim from the producer (total_duration_ms,
    -- cpu_ms, spyre_ms, kernel_mean_ms, memory_transfer_mean_ms, compile_ms,
    -- runtime_ms, mem_size_mb, pt_util_percent, duration_ms, ratio).
    -- A Map, not columns, because the sparsity is per record_type and no column set
    -- fits: mem_size_mb is set on 152/297 op rows and 0/40 model rows; batch_size
    -- the exact inverse. A wide fact is majority-NULL by construction and needs a
    -- DDL change per new metric. Units stay encoded in the key suffix.
    --
    -- An ARRAY per key, not one Float64: a metric measured n times IS n values. Holding
    -- one froze every statistic at ingest -- a geometric mean over a single stored value
    -- equals the arithmetic mean by construction, which is what the upstream PyTorch HUD
    -- computes from metric.benchmark_values. Variance, percentiles and a real geomean are
    -- only recomputable if the samples survive. Readers wanting one number take the mean
    -- via v_benchmark_results_enriched, which reduces this to a scalar Map under the same
    -- column name.
    measurements Map(LowCardinality(String), Array(Float64)),

    -- n behind each mean, as the PRODUCER reported it. Redundant with
    -- length(measurements[k]) whenever the producer sends its samples, and still the only
    -- source when it sends a pre-averaged number instead -- which is why it stays a column
    -- rather than being derived. Without a sample count a delta cannot be separated from
    -- noise. 0 = the producer did not say.
    iterations   UInt32 DEFAULT 0,

    props        Map(LowCardinality(String), String),

    -- regression_status is deliberately absent (255/297 op rows carry one in v1):
    -- a stored verdict with no recorded baseline cannot be checked against the data
    -- it summarises. Derived in v_benchmark_regression against an explicit baseline
    -- run_id; the view keeps the column name so the dashboard contract holds.
    CONSTRAINT chk_measurements CHECK length(measurements) > 0
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(ts)
ORDER BY (component, run_id, benchmark_id, backend);
-- Deliberately NO skip index on benchmark_id, even though every trend and regression view
-- groups by it across runs. A bloom filter only prunes when the matching rows are
-- CONTIGUOUS in sort order, and here they are the opposite: all 1,819 benchmark_ids recur
-- in many runs (32 for a sampled one), so a granule almost always holds some row for any
-- given benchmark. Measured: 6/6 granules read with the index in place -- it costs storage
-- and prunes nothing. If per-benchmark history becomes a hot path, the fix is a projection
-- or a benchmark_id-first ORDER BY, not an index.
