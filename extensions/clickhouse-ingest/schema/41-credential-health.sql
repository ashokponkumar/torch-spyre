-- CI credential health: what credentials the pipelines bind, and whether each still works.
-- Infrastructure telemetry, not an artifact fact (no artifact_id). Ids and metadata only: a
-- secret value never reaches either table, and `fingerprint` is a short digest so a rotation is
-- visible without exposing the value.

-- What discovery found, one row per credential; the latest observed_at wins. A writer registers a
-- credential here before probing it, because v_credential_health starts from this table.
CREATE TABLE IF NOT EXISTS credential_inventory
(
    system            LowCardinality(String),   -- jenkins | gha | argocd | dashboard | cluster
    scope             String,                   -- where the id resolves, e.g. system, folder:Spyre, org:torch-spyre, ns:openshift-gitops
    credential_id     String,
    kind              LowCardinality(String) DEFAULT '',  -- the store's own type name
    description       String DEFAULT '',
    owner             String DEFAULT '',
    consumers         Array(String),            -- jobs / workflows / workloads that bind it
    state             LowCardinality(String),   -- bound | unbound | gone
    source_created_at Nullable(DateTime64(3, 'UTC')),  -- first use (Jenkins) or creation (k8s)
    source_updated_at Nullable(DateTime64(3, 'UTC')),  -- last rotation the store reports (GHA, k8s)
    last_bound_at     Nullable(DateTime64(3, 'UTC')),
    observed_at       DateTime64(3, 'UTC'),
    job_run           String DEFAULT '',

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_credential_state CHECK state IN ('bound', 'unbound', 'gone')
)
ENGINE = ReplacingMergeTree(observed_at)
ORDER BY (system, scope, credential_id);

-- One row per probe of one credential. The site that holds the secret probes it: Jenkins for the
-- CI store, a workflow for GHA secrets, the dashboard backend for its own env, the cluster for
-- k8s secrets and certificates.
CREATE TABLE IF NOT EXISTS credential_probes
(
    ts            DateTime('UTC') DEFAULT now(),
    system        LowCardinality(String),
    scope         String,
    credential_id String,
    probe_site    LowCardinality(String),       -- jenkins | gha | dashboard | cluster
    probe         LowCardinality(String),       -- what was called, e.g. whoami, ping, tls
    status        LowCardinality(String),       -- ok | failing | unknown
    error_class   LowCardinality(String) DEFAULT '',  -- auth | forbidden | network | timeout | other
    http_status   UInt16 DEFAULT 0,
    latency_ms    UInt32 DEFAULT 0,
    expires_at    Nullable(DateTime64(3, 'UTC')),  -- NULL when the credential exposes none
    fingerprint   String DEFAULT '',
    detail        String DEFAULT '',            -- redacted error text
    job_run       String DEFAULT '',

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_probe_status CHECK status IN ('ok', 'failing', 'unknown'),
    CONSTRAINT chk_probe_site CHECK probe_site IN ('jenkins', 'gha', 'dashboard', 'cluster')
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(ts)
ORDER BY (system, scope, credential_id, ts)
-- Samples are hourly at most; a year shows rotation cadence.
TTL ts + INTERVAL 365 DAY;
