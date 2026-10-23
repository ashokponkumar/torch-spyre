-- Component state computed from probe samples. The state is a rule over samples and is never
-- written by a person or a model; annotations sit beside it. Constants, all proposals:
--   bucket 5 minutes; a vantage counts as reporting when its newest heartbeat is under 15 minutes old;
--   a component sample is current when it is under 15 minutes old; a window of 3 buckets (15 min) is
--   what "consecutive" means below.
-- Quorum over the vantages that reported a current, known sample (R reporting, F failing):
--   R = 0                       -> unknown
--   R = 1, F = 0                -> operational;  F = 1 -> degraded (one vantage never proves an outage)
--   R = 2, F = 0                -> operational;  F = 1 -> degraded;  F = 2 -> partial_outage, and
--                                  major_outage once both failed in 2 consecutive buckets
--   R >= 3, F = 0               -> operational;  F < R/2 -> degraded;  F >= R/2 and F < R -> partial_outage;
--                                  F = R -> major_outage
-- A 'degraded' sample (slow answer) counts as working for the quorum and lowers operational to degraded.
-- An 'unknown' sample is not a report. Unknown is never shown as operational.

-- Newest sample per component, vantage and bucket, with heartbeat-fresh vantages marked. cur_*
-- aliases: an alias equal to a source column is substituted into the other aggregates (ILLEGAL_AGGREGATION).
CREATE VIEW IF NOT EXISTS v_status_bucket AS
WITH beats AS
(
    SELECT vantage, max(ts) AS last_beat
    FROM status_samples FINAL
    WHERE probe = 'heartbeat'
    GROUP BY vantage
)
SELECT
    s.component AS component,
    s.bucket    AS bucket,
    s.vantage   AS vantage,
    argMax(s.state, s.ts)        AS cur_state,
    argMax(s.latency_ms, s.ts)   AS cur_latency_ms,
    argMax(s.error_class, s.ts)  AS cur_error_class,
    max(s.ts)                    AS cur_ts,
    max(b.last_beat)             AS beat_ts,
    max(b.last_beat) >= now64(3) - INTERVAL 15 MINUTE AS vantage_alive
FROM status_samples AS s FINAL
LEFT JOIN beats AS b ON b.vantage = s.vantage
WHERE s.probe != 'heartbeat' AND s.component != '*'
GROUP BY component, bucket, vantage;

-- The state now: per component, the quorum over each vantage's newest sample of the last 15 minutes.
CREATE VIEW IF NOT EXISTS v_status_current AS
WITH newest AS
(
    SELECT
        component, vantage,
        argMax(cur_state, bucket) AS v_state,
        max(bucket)               AS v_bucket,
        argMax(cur_ts, bucket)    AS v_ts,
        argMax(vantage_alive, bucket) AS v_alive
    FROM v_status_bucket
    WHERE bucket >= toStartOfFiveMinutes(now()) - INTERVAL 15 MINUTE
    GROUP BY component, vantage
),
-- Failing vantages in the two newest buckets, to tell one bad round from a standing failure.
prior AS
(
    SELECT component, bucket, countIf(cur_state = 'failing') AS n_failing, countIf(cur_state IN ('ok', 'degraded', 'failing')) AS n_known
    FROM v_status_bucket
    WHERE bucket >= toStartOfFiveMinutes(now()) - INTERVAL 10 MINUTE
    GROUP BY component, bucket
),
agg AS
(
    SELECT
        component,
        countIf(v_state IN ('ok', 'degraded', 'failing') AND v_alive)              AS reporting,
        countIf(v_state = 'failing' AND v_alive)                                   AS failing,
        countIf(v_state = 'degraded' AND v_alive)                                  AS slow,
        groupArrayIf(vantage, v_state = 'failing' AND v_alive)                     AS failing_from,
        groupArrayIf(vantage, v_state IN ('ok', 'degraded', 'failing') AND v_alive) AS reported_by,
        countIf(NOT v_alive)                                                       AS prober_offline,
        max(v_ts)                                                                  AS last_sample_at
    FROM newest
    GROUP BY component
),
repeat_fail AS
(
    SELECT component, countIf(n_known >= 2 AND n_failing = n_known) AS both_failed_buckets
    FROM prior
    GROUP BY component
)
SELECT
    a.component AS component,
    multiIf(
        a.reporting = 0,                                   'unknown',
        a.failing = 0 AND a.slow = 0,                      'operational',
        a.failing = 0,                                     'degraded',
        a.reporting = 1,                                   'degraded',
        a.reporting = 2 AND a.failing = 1,                 'degraded',
        a.reporting = 2 AND a.failing = 2,                 if(coalesce(r.both_failed_buckets, 0) >= 2, 'major_outage', 'partial_outage'),
        a.failing * 2 < a.reporting,                       'degraded',
        a.failing = a.reporting,                           'major_outage',
                                                           'partial_outage'
    ) AS state,
    a.reporting AS vantages_reporting,
    a.reported_by AS reported_by,
    a.failing_from AS failing_from,
    a.prober_offline AS vantages_offline,
    a.reporting = 0 AS no_data,
    if(a.last_sample_at = toDateTime64(0, 3, 'UTC'), NULL, a.last_sample_at) AS last_sample_at
FROM agg AS a
LEFT JOIN repeat_fail AS r ON r.component = a.component;

-- Per component and day: intervals (one per bucket with a known quorum) that were working, were not,
-- and were unknown, plus the share working. A bucket with no sample from any live vantage is unknown
-- and is left out of the share; the share is NULL when no bucket in the day was known. A degraded
-- bucket counts as working here and is also counted on its own.
CREATE VIEW IF NOT EXISTS v_status_daily AS
WITH per_bucket AS
(
    SELECT
        component, bucket,
        countIf(cur_state IN ('ok', 'degraded', 'failing')) AS n_known,
        countIf(cur_state = 'failing')                      AS n_fail,
        countIf(cur_state = 'degraded')                     AS n_slow
    FROM v_status_bucket
    GROUP BY component, bucket
)
SELECT
    component,
    toDate(bucket)                                          AS day,
    countIf(n_known > 0)                                    AS known_buckets,
    countIf(n_known = 0)                                    AS unknown_buckets,
    countIf(n_known > 0 AND n_fail * 2 >= n_known AND n_fail > 0 AND n_known >= 2) AS down_buckets,
    countIf(n_known = 1 AND n_fail = 1)                     AS single_vantage_fail_buckets,
    countIf(n_known > 0 AND n_fail = 0 AND n_slow > 0)      AS degraded_buckets,
    known_buckets - down_buckets                            AS working_buckets,
    if(known_buckets = 0, NULL, round(working_buckets / known_buckets, 4)) AS uptime_share
FROM per_bucket
GROUP BY component, day;
