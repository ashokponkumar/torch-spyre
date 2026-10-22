-- Credential health for the CI/CD maintenance page: each inventoried credential with its latest
-- probe and a single health class.

-- `expired` and `expiring` come from a probe's expires_at; `failing` wins over both, so a 401
-- reads red whatever the date says. A credential no probe has reached is `unprobed`.
CREATE VIEW IF NOT EXISTS v_credential_health AS
WITH probes AS
(
    SELECT
        system, scope, credential_id,
        argMax(status, ts)       AS probe_status,
        argMax(error_class, ts)  AS error_class,
        argMax(probe_site, ts)   AS probe_site,
        argMax(probe, ts)        AS probe,
        argMax(expires_at, ts)   AS expires_at,
        argMax(fingerprint, ts)  AS fingerprint,
        argMax(detail, ts)       AS detail,
        max(ts)                  AS last_probe_at,
        maxIf(ts, status = 'ok') AS last_ok_at
    FROM credential_probes
    WHERE ts >= now() - INTERVAL 90 DAY
    GROUP BY system, scope, credential_id
)
SELECT
    i.system AS system, i.scope AS scope, i.credential_id AS credential_id,
    i.kind AS kind, i.description AS description, i.owner AS owner,
    i.consumers AS consumers, i.state AS state,
    i.source_created_at AS source_created_at, i.source_updated_at AS source_updated_at,
    i.last_bound_at AS last_bound_at, i.observed_at AS observed_at,
    p.probe_site AS probe_site, p.probe AS probe,
    p.probe_status AS probe_status, p.error_class AS error_class,
    p.expires_at AS expires_at, p.fingerprint AS fingerprint, p.detail AS detail,
    p.last_probe_at AS last_probe_at, p.last_ok_at AS last_ok_at,
    dateDiff('day', now(), p.expires_at) AS days_to_expiry,
    multiIf(
        p.last_probe_at = toDateTime(0),                        'unprobed',
        p.probe_status = 'failing',                             'failing',
        p.expires_at IS NOT NULL AND p.expires_at <= now(),     'expired',
        p.expires_at IS NOT NULL AND p.expires_at <= now() + INTERVAL 30 DAY, 'expiring',
        p.probe_status = 'ok',                                  'ok',
                                                                'unknown'
    ) AS health
FROM credential_inventory AS i FINAL
LEFT JOIN probes AS p USING (system, scope, credential_id)
WHERE i.state != 'gone';
