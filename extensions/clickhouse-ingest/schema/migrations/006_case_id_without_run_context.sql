-- Re-key test_case_id to the identity-tags-only recipe (identity.RUN_CONTEXT_TAG_NAMESPACES are
-- no longer hashed), so one test has one id across arches and tiers. Every hash input is stored,
-- so history is re-keyed, not abandoned: the old id's run-context tags move onto its run rows.
--
-- The uuid5 below is CaseId.derive in SQL; it reproduced all 50,275 stored ids on prod when
-- given the full tag set. Keep the namespace list equal to RUN_CONTEXT_TAG_NAMESPACES.

CREATE TABLE IF NOT EXISTS case_id_rekey
(
    old_id    UUID,
    new_id    UUID,
    ts        DateTime,
    component LowCardinality(String),
    classname String,
    name      String,
    id_tags   Array(LowCardinality(String)),
    run_tags  Array(LowCardinality(String))
)
ENGINE = MergeTree ORDER BY old_id;

INSERT INTO case_id_rekey
WITH ['platform', 'testtype', 'nightly', 'weekly', 'refcoverage'] AS ctx
SELECT
    test_case_id,
    toUUID(lower(concat(
        substring(h, 1, 8), '-', substring(h, 9, 4), '-5', substring(h, 14, 3), '-',
        substring(hex(bitOr(bitAnd(reinterpretAsUInt8(unhex(concat('0', substring(h, 17, 1)))), 3), 8)), 2, 1),
        substring(h, 18, 3), '-', substring(h, 21, 12)))),
    ts, component, classname, name,
    arrayFilter(t -> NOT has(ctx, splitByString('__', lowerUTF8(trimBoth(t)))[1]), tags),
    arrayFilter(t -> has(ctx, splitByString('__', lowerUTF8(trimBoth(t)))[1]), tags)
FROM
(
    SELECT *,
        hex(substring(SHA1(concat(
            unhex('cb0af9bf28585eab9211f51190531bf3'),
            lowerUTF8(trimBoth(component)), '|',
            lowerUTF8(trimBoth(classname)), '|',
            lowerUTF8(trimBoth(name)), '|',
            arrayStringConcat(arraySort(arrayDistinct(arrayFilter(
                t -> t != '' AND NOT has(ctx, splitByString('__', t)[1]),
                arrayMap(t -> lowerUTF8(trimBoth(t)), tags)))), ','))), 1, 16)) AS h
    FROM test_cases
);

-- One identity row per new id, unless a run-context-free case already holds it.
INSERT INTO test_cases (ts, test_case_id, component, classname, name, tags)
SELECT min(ts), new_id, any(component), argMin(classname, ts), argMin(name, ts), argMin(id_tags, ts)
FROM case_id_rekey
WHERE new_id != old_id AND new_id NOT IN (SELECT test_case_id FROM test_cases)
GROUP BY new_id;

-- audit_uuid/audit_timestamp carried over: a moved row is the same observation.
INSERT INTO test_case_runs
    (ts, run_id, test_case_id, component, status, duration_s, fail_message, props, tags,
     audit_uuid, audit_timestamp)
SELECT r.ts, r.run_id, k.new_id, r.component, r.status, r.duration_s, r.fail_message, r.props,
       k.run_tags, r.audit_uuid, r.audit_timestamp
FROM test_case_runs AS r
INNER JOIN case_id_rekey AS k ON k.old_id = r.test_case_id
WHERE k.new_id != k.old_id;

DELETE FROM test_case_runs
WHERE test_case_id IN (SELECT old_id FROM case_id_rekey WHERE new_id != old_id);

DELETE FROM test_cases
WHERE test_case_id IN (SELECT old_id FROM case_id_rekey WHERE new_id != old_id);

-- The re-insert above fired run_case_counters_mv a second time for every moved row; the
-- counters are per run, not per case, so a full recount is exact.
TRUNCATE TABLE run_case_counters;

INSERT INTO run_case_counters
SELECT
    run_id,
    component,
    count(),
    countIf(status = 'passed'),
    countIf(status = 'failed'),
    countIf(status = 'error'),
    countIf(status = 'skipped'),
    countIf(status = 'xfail'),
    countIf(status = 'xpass')
FROM test_case_runs
GROUP BY run_id, component;

DROP TABLE case_id_rekey;
