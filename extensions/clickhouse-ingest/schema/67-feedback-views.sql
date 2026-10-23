-- Scores built from feedback_signals. Votes are signals, not labels: a score ranks what to review and
-- can demote a signature, and a share of down-votes is not a precision. Constants, all proposals:
--   half-life 30 days (a signal of age t counts 0.5^(t/30d));
--   one actor's current signal of a kind on a target counts once, and an actor's weight is capped at 3;
--   up/affected add, down subtract; miss and chip carry no sign of their own and are counted apart.

-- The current signal per (target, actor, signal): an actor's repeat is a change of mind, so only
-- the newest counts, and the history stays in the table.
CREATE VIEW IF NOT EXISTS v_feedback_current AS
SELECT
    target_type, target_id, actor, signal,
    argMax(source, created_at)            AS cur_source,
    argMax(detail, created_at)            AS cur_detail,
    argMax(weight_hint, created_at)       AS cur_weight_hint,
    argMax(signature_id, created_at)      AS cur_signature_id,
    argMax(signature_version, created_at) AS cur_signature_version,
    max(created_at)                       AS cur_at
FROM feedback_signals
GROUP BY target_type, target_id, actor, signal;

CREATE VIEW IF NOT EXISTS v_feedback_scores AS
SELECT
    target_type, target_id,
    anyLast(cur_signature_id)                                         AS signature_id,
    uniqExact(actor)                                                  AS distinct_voters,
    uniqExactIf(actor, signal IN ('up', 'affected'))                  AS ups,
    uniqExactIf(actor, signal IN ('down', 'not_affected'))            AS downs,
    uniqExactIf(actor, signal = 'miss')                               AS misses,
    round(sumIf(
        least(cur_weight_hint, 3) * exp2(-dateDiff('second', cur_at, now64(3)) / (30 * 86400.)),
        signal IN ('up', 'affected')), 4)                             AS decayed_up,
    round(sumIf(
        least(cur_weight_hint, 3) * exp2(-dateDiff('second', cur_at, now64(3)) / (30 * 86400.)),
        signal IN ('down', 'not_affected')), 4)                       AS decayed_down,
    round(decayed_up - decayed_down, 4)                               AS score,
    max(cur_at)                                                       AS last_signal_at
FROM v_feedback_current
GROUP BY target_type, target_id;

-- Quality of each signature version from the human verdicts on its matches (target_type
-- signature_match). Wilson 95% lower bound of the share of right matches: precision_lower is what to
-- trust, precision_raw is shown beside it, and a signature with no human verdict is `unproven`
-- however many matches it has. A match is labelled once per actor; two actors who disagree both count.
CREATE VIEW IF NOT EXISTS v_signature_quality AS
SELECT
    cur_signature_id        AS signature_id,
    cur_signature_version   AS signature_version,
    uniqExactIf(target_id, signal IN ('up', 'down'))                  AS labelled_matches,
    uniqExactIf(target_id, signal = 'up')                             AS tp_matches,
    uniqExactIf(target_id, signal = 'down')                           AS fp_matches,
    uniqExactIf(actor, signal IN ('up', 'down'))                      AS reviewers,
    countIf(cur_source = 'explicit' AND signal IN ('up', 'down'))     AS human_verdicts,
    if(labelled_matches = 0, NULL, round(tp_matches / labelled_matches, 4)) AS precision_raw,
    if(labelled_matches = 0, NULL, round(
        (tp_matches / labelled_matches + 1.96 * 1.96 / (2 * labelled_matches)
         - 1.96 * sqrt((tp_matches / labelled_matches * (1 - tp_matches / labelled_matches)
                        + 1.96 * 1.96 / (4 * labelled_matches)) / labelled_matches))
        / (1 + 1.96 * 1.96 / labelled_matches), 4))                   AS precision_lower,
    multiIf(human_verdicts = 0, 'unproven', labelled_matches < 20, 'few_labels', 'measured') AS evidence,
    countIf(signal = 'miss')                                          AS misses_reported,
    max(cur_at)                                                       AS last_label_at
FROM v_feedback_current
WHERE target_type = 'signature_match' AND cur_signature_id != ''
GROUP BY signature_id, signature_version;
