-- Incidents with their current state and response times, for the maintenance page and Ask Spyre.

-- Timings are measured from the first log row; an unset milestone is NULL, not zero.
-- `signals` counts events attached to the incident, so one cause behind 40 builds reads as 40.
CREATE VIEW IF NOT EXISTS v_incidents AS
WITH log AS
(
    SELECT
        incident_id,
        min(updated_at)                                  AS first_logged_at,
        max(updated_at)                                  AS last_logged_at,
        minIf(updated_at, state = 'acknowledged')        AS first_ack_at,
        minIf(updated_at, state IN ('mitigated', 'resolved', 'dismissed')) AS first_stop_at,
        argMax(state, updated_at)                        AS state,
        argMax(severity, updated_at)                     AS severity,
        argMax(title, updated_at)                        AS title,
        argMax(cause_class, updated_at)                  AS cause_class,
        argMax(signature, updated_at)                    AS signature,
        argMax(owner, updated_at)                        AS owner,
        argMax(owner_email, updated_at)                  AS owner_email,
        argMax(thread_url, updated_at)                   AS thread_url,
        argMax(evidence_url, updated_at)                 AS evidence_url,
        argMax(updated_by, updated_at)                   AS updated_by,
        count()                                          AS log_rows
    FROM incident_log
    GROUP BY incident_id
),
events AS
(
    SELECT incident_id, count() AS signals, min(ts) AS first_signal_at,
           groupUniqArray(source) AS sources, groupUniqArray(subject) AS subjects
    FROM incident_events
    WHERE incident_id != ''
    GROUP BY incident_id
)
SELECT
    l.incident_id AS incident_id, l.state AS state, l.severity AS severity, l.title AS title,
    l.cause_class AS cause_class, l.signature AS signature,
    l.owner AS owner, l.owner_email AS owner_email,
    l.thread_url AS thread_url, l.evidence_url AS evidence_url, l.updated_by AS updated_by,
    l.first_logged_at AS opened_at, l.last_logged_at AS updated_at, l.log_rows AS log_rows,
    e.signals AS signals, e.sources AS sources, e.subjects AS subjects,
    if(e.first_signal_at = toDateTime64(0, 3, 'UTC'), NULL, e.first_signal_at) AS first_signal_at,
    if(l.first_ack_at = toDateTime64(0, 3, 'UTC'), NULL, l.first_ack_at) AS first_ack_at,
    if(l.first_stop_at = toDateTime64(0, 3, 'UTC'), NULL, l.first_stop_at) AS first_stop_at,
    if(l.first_ack_at = toDateTime64(0, 3, 'UTC'), NULL, dateDiff('second', l.first_logged_at, l.first_ack_at)) AS seconds_to_ack,
    if(l.first_stop_at = toDateTime64(0, 3, 'UTC'), NULL, dateDiff('second', l.first_logged_at, l.first_stop_at)) AS seconds_to_stop,
    l.state IN ('open', 'acknowledged')              AS is_active
FROM log AS l
LEFT JOIN events AS e USING (incident_id);
