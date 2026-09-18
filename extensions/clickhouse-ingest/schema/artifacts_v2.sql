-- Artifact registry, schema v2. Zero backward compatibility with the v1
-- artifacts/artifact_tags/artifact_results/artifact_metadata set: v2 is a clean
-- break on fresh data, read by a new dashboard layer.
--
-- Rationale for every decision below: docs/clickhouse_v2_artifacts_schema.md
--
-- The organising invariant: `artifacts`, `artifact_refs` and `artifact_results`
-- hold facts fixed at production time and keyed by artifact_id. `artifact_tags`
-- is the only mutable layer, and it points DOWN at the others -- nothing in the
-- immutable layer references a tag. That is what keeps `artifacts` from needing
-- an ALTER every time a channel, consume method or test tier is added.
--
-- Bag-column convention, uniform across this file and functional_tests_v2.sql:
--   `props` = Map(LowCardinality(String), String), open-ended, NEVER in a key --
--             extend it freely, no DDL and no identity churn. All 4 tables here.
--   `tags`  = Array(LowCardinality(String)), a SET, and IN an identity hash --
--             only test_cases.tags today. Sort before hashing.
-- The names differ because the guarantees differ; do not converge them.

CREATE TABLE IF NOT EXISTS artifacts
(
    ts            DateTime DEFAULT now(),
    -- uuid5 DERIVED from (component, artifact_name, id12, arch) -- see artifactId() in
    -- pushToClickhouse.groovy and v2_artifact_id() in the ingest scripts, which must agree.
    -- Derived so a consumer holding only those four fields computes the same id without
    -- anything threaded to it; not every source produces an id12, so a GHA leg hashes
    -- (base image + installed set) into that slot instead. arch is required -- one id12
    -- exists per arch plus a multi pointer, so dropping it collided 1,043 rows.
    -- The four inputs stay in `props`: a uuid cannot be read back, and every human-facing
    -- "which artifact is this" lookup needs them.
    artifact_id   UUID,
    component     LowCardinality(String),
    -- amd64 | ppc64le | s390x, plus 'multi' for manifest-join refs. NOTE prod carries two
    -- spellings by table family (this lineage amd64; test_runs/images x86_64), and
    -- run_id's hash folds to x86_64 -- so read the normalizing views for anything
    -- cross-family, and never parse an arch out of artifact_id or a run_id.
    arch          LowCardinality(String),
    kind          LowCardinality(String),   -- image | rpm | wheel | generic
    artifact_name String,                   -- e.g. ibm-flex-devel; display only, NOT identity

    origin        LowCardinality(String),   -- how it came to exist, as history not plan:
                                            -- built | copied | promoted | upstream

    -- Deps split by identity participation, because config.yaml's `identity: false`
    -- affects build order but must NOT feed the reuse hash.
    -- identity_deps: everything whose content IS an input to this artifact's identity -- the
    --   source repos and pinned upstream artifacts it was compiled against, e.g.
    --   ['flex@<id12>', 'deeptools@<id12>'].
    -- context_deps: the `identity: false` deps -- present at build time, deliberately outside
    --   the hash. Chiefly the BUILDER image: a from-source producer compiles inside it, but the
    --   artifact is a pure function of its own source+recipe, so folding the builder in would
    --   churn a cold rebuild on every builder bump for byte-identical output. Also the cicd
    --   repo where a config declares it with roles [context]. See resolve_deps.py `identity`.
    identity_deps Array(String),
    context_deps  Array(String),

    -- Multi-repo builds carry N sources with no natural ordering, so this is an
    -- array rather than a table: immutable, written once with the artifact.
    sources       Array(Tuple(
                      repo    LowCardinality(String),
                      git_ref String,
                      git_sha String
                  )),

    -- REQUIRED here, not optional metadata: id12 and artifact_name are hash inputs to
    -- artifact_id, so with the id opaque this is the only place they remain readable.
    --
    -- ONE url key across the whole schema: `run_url`, the CI run that produced or reported this
    -- row, whatever system ran it -- a Jenkins build url or a GitHub Actions run url. v1 spread
    -- the same fact over job_url / build_url / gha_url / run_urls / artifact_url plus
    -- orch_run_key / test_job_key / job_key, so a reader had to know which producer wrote a row
    -- before knowing which field to look in. Nothing ever filtered or joined on any of them --
    -- they are display, and one key serves display.
    props         Map(LowCardinality(String), String),  -- id12, content hash + alg, size,
                                                        -- labels, run_url, build_number

    CONSTRAINT chk_kind   CHECK kind   IN ('image','rpm','wheel','generic'),
    -- Each value is a DISTINCT way a row comes to exist, and each does get its own row --
    -- because the artifact_id differs in every case except 'promoted':
    --   built    -- this pipeline compiled it. The overwhelming majority.
    --   copied   -- the same content republished at a second address (an arch-specific image
    --               copied under a multi-arch manifest): new arch, so a new artifact_id.
    --   promoted -- the SAME artifact_id gaining a channel tag (nightly -> 2.0). The row is
    --               the promotion EVENT; the tag itself lives in artifact_tags.
    --   upstream -- not built here at all: a third-party wheel or base image we pin and test.
    --               It has no build of ours, but verdicts still hang off it, so it needs a row.
    -- No 'reused' value: reuse is not a property of the ARTIFACT, which exists exactly once.
    -- A reusing job records itself against the same artifact_id (prod: 285 artifacts carry
    -- 3,349 reuse events, one of them reused by 978 jobs), so reuse is an edge, not an origin.
    CONSTRAINT chk_origin CHECK origin IN ('built','copied','promoted','upstream')
)
ENGINE = MergeTree()
ORDER BY (component, arch, artifact_id);
-- MergeTree, not Replacing: an artifact is produced once and never restated, so a
-- duplicate artifact_id is a producer bug that must stay VISIBLE rather than be
-- silently collapsed. No PARTITION BY: the hot query (the reuse/tier gate) filters
-- identity with no time predicate, so a monthly partition prunes nothing, and at
-- this table's size partitioning only fragments parts.


CREATE TABLE IF NOT EXISTS artifact_refs
(
    ts             DateTime DEFAULT now(),
    artifact_id    UUID,

    -- method is HOW a consumer obtains it; ref_kind is what shape `ref` therefore takes.
    -- They move together, so one row's worth of each, by kind:
    --   image:   method 'container-pull', ref_kind 'pullspec'
    --            index_uri 'icr.io/ai_sw_accel/2.0'
    --            ref       'icr.io/ai_sw_accel/2.0/next/builds/x86_64/torch-spyre:<id12>'
    --   rpm:     method 'dnf', ref_kind 'glob'
    --            index_uri 'https://na.artifactory.swg-devops.com/artifactory/
    --                       sys-ai-sw-accel-team-rpm-local/next/x86_64'
    --            ref       'ibm-flex-*h<sha7>*.el10.x86_64.rpm'
    --   wheel:   method 'pip', ref_kind 'url'
    --            index_uri the team PyPI index; ref the versioned wheel filename
    --   generic: method 'download', ref_kind 'url'
    --            ref the full Artifactory generic-local path
    method         LowCardinality(String),  -- container-pull | dnf | pip | download
    ref_kind       LowCardinality(String),  -- pullspec | glob | url
    index_uri      String,                  -- registry host, yum repo base, PyPI index
    -- A GLOB for RPMs: the NEVRA's version/build segments are unpredictable, and the
    -- gitversion sha7 in the filename is NOT the PR-head sha -- the producer embeds the
    -- PR-head separately as an `h<sha7>` token, which is what the glob matches on.
    ref            String,
    -- e.g. 'sha256:ab12..' for an image, '' for an RPM resolved by glob.
    content_digest String DEFAULT '',       -- '' rather than Nullable: absence is the
                                            -- empty string, no per-row null mask
    props          Map(LowCardinality(String), String),

    CONSTRAINT chk_method   CHECK method   IN ('container-pull','dnf','pip','download'),
    CONSTRAINT chk_ref_kind CHECK ref_kind IN ('pullspec','glob','url')
)
ENGINE = ReplacingMergeTree(ts)
ORDER BY (artifact_id, method, ref);
-- Holds only addresses that never move. Re-publishing the same artifact to the same
-- index by the same method is the SAME fact, so dedupe is correct here.
-- Routing rule: a ref carrying any discriminator that pins it to one artifact or one
-- point in time (an id12 or a date) is immutable and belongs HERE -- including the
-- dated promotion aliases. Only a bare channel address goes on the tag.


CREATE TABLE IF NOT EXISTS artifact_tags
(
    ts             DateTime DEFAULT now(),

    tag            String,                  -- THE resolution key: 'nightly', 'nightly-2026-08-31'
    -- Reporting dimension only: the CHANNEL a tag belongs to, never the tag itself. The
    -- writer validates against exactly these four (pushToClickhouse.groovy: WARN on anything
    -- else, so a stray value still lands and stays visible rather than losing the row).
    -- nightly/weekly come from the orchestrator's TIER_ALIAS_NAME dated-retag flow, main from
    -- a flat promote-to-2.0, pr from a per-PR tag. There is no 'dev'.
    tag_family     LowCardinality(String),  -- nightly | weekly | main | pr
    artifact_id    UUID,                      -- what the tag pointed to as of ts

    -- Inline, NOT a reference into artifact_refs, because these are a different KIND of
    -- address. artifact_refs holds IMMUTABLE addresses -- an id12-bearing pullspec resolves to
    -- the same bytes forever, so it is a property of the artifact and is keyed by artifact_id.
    -- These are MOVING addresses that exist only because of this tag (':nightly', no id12, no
    -- date); the same string points at a different artifact next week. Storing them under
    -- artifact_id would assert something false -- that the artifact owns an address that will
    -- outlive its claim on it -- and a ReplacingMergeTree keyed on (artifact_id, method, ref)
    -- would collapse successive tag holders onto one row, destroying the history this table
    -- exists to keep. Frozen at ts, they answer "what did ':nightly' mean on 08-31".
    refs           Array(Tuple(
                       method    LowCardinality(String),
                       ref_kind  LowCardinality(String),
                       index_uri String,
                       ref       String
                   )),
    -- The other direction, and where a reference IS right: a promotion that also publishes an
    -- immutable address writes that address to artifact_refs and records only its key here,
    -- so the string is not owned twice.
    published_refs Array(String),
    props          Map(LowCardinality(String), String)     -- promoted_by, run_url, actor,
)
ENGINE = MergeTree()
ORDER BY (tag, ts);
-- Deliberately NO skip index on artifact_id for the reverse lookup ("which tags point at
-- THIS artifact"). A bloom filter prunes only when the matching rows are CONTIGUOUS, and a
-- tag-led sort scatters them: 37.5% of artifacts carry more than one tag, which is the
-- design's intent (a rolling tag coexists with its dated aliases). The table is also small
-- by construction -- one row per promotion, 1,716 rows in two granules -- so the reverse
-- scan is cheap and an index would cost storage to prune nothing.
-- The row's payload is the membership fact (tag, artifact_id, ts); both arrays are
-- optional annotations describing what the promotion happened to do, and BOTH may be
-- empty (a re-promotion of an already-published artifact publishes no new address and
-- moves nothing). Never constrain them to be non-empty.
-- Plain MergeTree: every promotion is a NEW fact, not a new version of one. Dedupe
-- would collapse the rolling tag's history (measured: 28 rows, 28 distinct artifacts).
-- Rolling vs pinned is EMERGENT, not stored: uniqExact(artifact_id) > 1 per tag.
-- ORDER BY leads with `tag` because that is the resolution key; tag_family is a
-- grouping dimension and is deliberately not part of the sort key.


CREATE TABLE IF NOT EXISTS artifact_results
(
    ts          DateTime DEFAULT now(),
    artifact_id UUID,                         -- WHAT was tested: immutable identity, never a
                                            -- tag (a tag moves, and a moved tag would make
                                            -- an old verdict describe a new artifact)
    run_id     UUID,                       -- joins the test/benchmark run for per-case detail

    result_kind LowCardinality(String),     -- functional | performance | image
    test_type   LowCardinality(String),     -- the tier ladder, constrained below
    state       LowCardinality(String),     -- passed | failed | error | running
    arch        LowCardinality(String),     -- where it RAN; may differ from artifacts.arch
                                            -- (a 'multi' manifest tested on amd64)

    -- No stored total_tests/passed/failed/errors/skipped. They are exactly derivable by
    -- counting test_case_runs for this run_id (measured equal on all 158,202 v1 shards), and
    -- a stored copy is a second source of truth that drifts the moment a delta run copies a
    -- covering run's cases in. The live v1 table never carried these columns either, so
    -- nothing is lost. `state` still distinguishes a suite that never ran ('error'/'running'
    -- with no case rows) from one that ran and regressed.
    duration_s  Float32,                    -- suite wall clock; measured 94.1% != sum(cases)

    -- `run_url` is THE link: the CI run that produced this verdict, Jenkins or GitHub Actions,
    -- under one key so a reader never has to know which system ran it. Emitted even when it may
    -- have aged out -- Jenkins prunes under numToKeep, and a 404 that says "it ran here" is
    -- worth more than no link. There is deliberately no separate run_key/job_key: the key form
    -- (JOB_NAME#BUILD_NUMBER) is recoverable from the url, nothing ever filtered or joined on
    -- it, and it is already a run_id hash input -- so storing it again bought only ambiguity.
    props       Map(LowCardinality(String), String),  -- run_url, source, plus per-producer keys

    -- Closes a live v1 defect: 401 rows currently carry IMAGE NAMES in test_type
    -- (spyre-inference-dev, hf-adapters-dev). A CHECK makes that unwritable.
    -- Deliberately NOT an Enum: an unknown value would throw on insert, so adding a
    -- tier would need an ALTER before the writer could emit it, and Enum ordering is
    -- by declaration -- an inserted-in-the-middle tier would silently reorder the
    -- ladder comparisons that tier_satisfies() depends on.
    CONSTRAINT chk_test_type   CHECK test_type   IN
        ('smoke','unit','integration','regression','trunk','perf'),
    CONSTRAINT chk_state       CHECK state       IN ('passed','failed','error','running'),
    CONSTRAINT chk_result_kind CHECK result_kind IN ('functional','performance','image'),

    -- run_id is the join key the rest of the schema hangs off (the UI joins on it more
    -- often than on artifact_id) but it cannot lead the sort key: reads are overwhelmingly
    -- "this artifact's verdicts", which needs artifact_id first. A bloom filter covers the
    -- other direction. It pays only because a run's rows are CONTIGUOUS here -- every run
    -- maps to exactly one artifact (0 of 1,169 dev and 0 of 1,927 prod runs span two), so a
    -- granule either holds the run or does not. Measured: granules 3/3 -> 1/3, index 1.52KiB
    -- on a 54KiB table.
    -- Inline, not ALTER ... ADD INDEX: an ADD on an existing table registers the index but
    -- builds nothing until MATERIALIZE INDEX, and a registered-but-empty index prunes zero.
    INDEX idx_run_id run_id TYPE bloom_filter(0.01) GRANULARITY 1
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(ts)
ORDER BY (artifact_id, result_kind, test_type, ts)
TTL ts + INTERVAL 90 DAY DELETE WHERE state = 'running';
-- A sparse junction, not a spine: measured 1,163 rows against 15,999 runs (7.3%), so
-- run metadata must NOT live here -- 93% of runs would have nowhere to go.
-- state='running' is ADVISORY DISPLAY ONLY. Mutual exclusion stays in Jenkins lock()
-- and liveness in Jenkins build state; ClickHouse has neither row updates nor locks,
-- and a crashed run's stale 'running' row must never gate anything. The TTL reaps
-- orphans. The tier gate accepts ONLY 'passed'.
