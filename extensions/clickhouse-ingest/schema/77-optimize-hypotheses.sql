-- Optimize: the hypothesis ledger behind the dashboard's Perf Lab flow. A hypothesis is written by a person
-- (What-If "Turn into hypothesis", chat) or an agent round, then tested, promoted, parked or linked to an
-- optimization. Every change is a new row; the highest version per hypothesis_id is the current state.

CREATE TABLE IF NOT EXISTS optimize_hypotheses
(
    hypothesis_id   String,                      -- H-<6 hex>
    version         UInt64,                      -- strictly increasing per id; writers read-then-write with the next value
    updated_at      DateTime64(3, 'UTC') DEFAULT now64(3),
    state           LowCardinality(String),
    origin          LowCardinality(String),      -- chat | auto | link | import
    statement       String,
    mechanism       String DEFAULT '',
    lever           LowCardinality(String) DEFAULT '',
    level           LowCardinality(String) DEFAULT '',
    prediction      String DEFAULT '',
    test            String DEFAULT '{}',         -- JSON {kind, how, refutes_if}
    tests           String DEFAULT '[]',         -- JSON list of {verdict, observed, evidence[], size_pct, size_source}
    seed            String DEFAULT '',
    metric          LowCardinality(String) DEFAULT '',
    scenario        String DEFAULT '',
    source_url      String DEFAULT '',
    code            String DEFAULT '{}',         -- JSON {repo, path, line}
    links           String DEFAULT '{}',         -- JSON {optimization, whatif_link, dashboard_link}
    session         String DEFAULT '',           -- Opt Studio session that wrote this version
    updated_by      String DEFAULT '',           -- the login that wrote this version
    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_id     CHECK match(hypothesis_id, '^H-[0-9a-f]{6}$'),
    CONSTRAINT chk_state  CHECK state IN ('draft', 'testing', 'survived', 'refuted', 'inconclusive', 'promoted', 'parked'),
    CONSTRAINT chk_origin CHECK origin IN ('chat', 'auto', 'link', 'import'),
    CONSTRAINT chk_lever  CHECK lever IN ('', 'less-work', 'faster', 'overlap', 'fixed-cost'),
    CONSTRAINT chk_stmt   CHECK statement != ''
)
ENGINE = ReplacingMergeTree(version)
ORDER BY hypothesis_id;

-- Current state per hypothesis. Output names differ from the columns: an alias equal to an aggregated
-- column (argMax(version, version) AS version) is rejected as a nested aggregate.
CREATE VIEW IF NOT EXISTS v_optimize_hypotheses AS
SELECT
    hypothesis_id,
    max(h.version)                       AS latest_version,
    argMax(h.updated_at, h.version)      AS last_updated,
    argMax(h.state, h.version)           AS current_state,
    argMax(h.origin, h.version)          AS origin_kind,
    argMax(h.statement, h.version)       AS hypothesis,
    argMax(h.mechanism, h.version)       AS mechanism_text,
    argMax(h.lever, h.version)           AS lever_kind,
    argMax(h.level, h.version)           AS subject_level,
    argMax(h.prediction, h.version)      AS prediction_text,
    argMax(h.test, h.version)            AS test_json,
    argMax(h.tests, h.version)           AS tests_json,
    argMax(h.seed, h.version)            AS seed_text,
    argMax(h.metric, h.version)          AS metric_name,
    argMax(h.scenario, h.version)        AS scenario_id,
    argMax(h.source_url, h.version)      AS source,
    argMax(h.code, h.version)            AS code_json,
    argMax(h.links, h.version)           AS links_json,
    argMax(h.session, h.version)         AS studio_session,
    argMax(h.updated_by, h.version)      AS last_updated_by,
    min(h.updated_at)                    AS created_at
FROM optimize_hypotheses AS h
GROUP BY hypothesis_id;
