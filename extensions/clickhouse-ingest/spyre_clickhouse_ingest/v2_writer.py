# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The schema-v2 write path: dedup guard plus the test_cases/test_case_runs insert.

Rows are built as dicts and ordered by the schema model, so a field cannot be assigned to the
wrong column and the column order lives in exactly one place.
"""

import sys

from . import schema
from .identity import v2_benchmark_id, v2_tags_for_case, v2_test_case_id


def v2_already_ingested(
    client, db: str, run_id: str, component: str, source_file: str = ""
) -> bool:
    """Has THIS source file's rows for this run already landed?

    test_case_runs is a plain MergeTree with no dedup key, so a double ingest of one leg
    DOUBLES its counts -- and v2 dropped the stored counters precisely because they are
    derived from these rows. This check is what keeps that correct.

    Scoped by source file, not just run_id: a sharded run is MANY xml files under ONE
    run_id (the pipeline passes --xml-dir with every shard in a single invocation), so a
    run-level check lets the first shard block all the others. Measured on a real
    Spyre-Next run: 9 of 10 cases silently dropped across 7 shards.

    `props['source_file']` carries the discriminator. props is a Map outside every key, so
    recording it costs no sort-order change.
    """
    table = schema.TEST_CASE_RUNS.qualified(db)
    if source_file:
        rows = client.query(
            f"SELECT count() FROM {table} "
            "WHERE component = {component:String} AND run_id = {run_id:UUID} "
            "AND props['source_file'] = {sf:String}",
            parameters={"component": component, "run_id": run_id, "sf": source_file},
        ).result_rows
    else:
        # No discriminator given: fall back to the run-level check rather than skip
        # dedup entirely, so a caller that cannot name the file is still protected.
        rows = client.query(
            f"SELECT count() FROM {table} "
            "WHERE component = {component:String} AND run_id = {run_id:UUID}",
            parameters={"component": component, "run_id": run_id},
        ).result_rows
    return bool(rows and rows[0][0] > 0)


def insert_v2(
    client, db: str, component: str, run_id: str, cases: list, source_file: str = ""
) -> int:
    """Write test_cases (identity) + test_case_runs (outcome) for one leg.

    Rows are built as dicts and ordered by v2_schema, so a field cannot be assigned to the
    wrong column and the column order lives in exactly one place.

    Dropped from v2 deliberately: filename, suite_name, runner_run_id, and every stored
    counter -- all derivable, and a stored counter invites drift.
    """
    if not cases:
        return 0
    ident_rows, run_rows = {}, []

    skipped_unidentifiable = 0
    for c in cases:
        tags = v2_tags_for_case(c)
        classname, name = c.get("classname", ""), c.get("name", "")
        tcid = v2_test_case_id(component, classname, name, tags)
        if not tcid:
            # Refused identity: writing the row anyway would collide it with every
            # other unidentifiable case rather than merely orphaning it.
            skipped_unidentifiable += 1
            continue
        # Keyed by id: identical identity rows within a leg are one fact.
        ident_rows[tcid] = {
            "test_case_id": tcid,
            "component": component,
            "classname": classname,
            "name": name,
            "tags": tags,
        }
        run_rows.append(
            {
                "run_id": run_id,
                "test_case_id": tcid,
                "component": component,
                "status": c.get("status", ""),
                "duration_s": float(c.get("duration_s", 0) or 0),
                "fail_message": (c.get("fail_message") or "")[:8192],
                # source_file names the xml this row came from, so a sharded run dedups
                # per file instead of the first shard blocking the rest.
                # ran_in names the run that ACTUALLY EXECUTED this case. For a case this
                # run ran it is this run_id; a reuse copy carries the original executor's
                # (see copy_reused_cases). Every "how much did we execute" query must
                # filter props['ran_in'] = run_id -- without it, reuse copies inflate the
                # count. Queries asking "what does this run report for the tier" want the
                # unfiltered total, which is the point of writing the copies at all.
                "props": (
                    {
                        "ran_in": run_id,
                        **({"source_file": source_file} if source_file else {}),
                    }
                ),
            }
        )
    # Cross-run dedup, not just in-leg: test_cases is a plain MergeTree, so re-inserting a
    # known identity appends a duplicate instead of collapsing it.
    schema.insert_identities(client, schema.TEST_CASES, ident_rows, db=db)
    schema.insert(client, schema.TEST_CASE_RUNS, run_rows, db=db)
    if skipped_unidentifiable:
        print(
            f"  [warn] v2: {skipped_unidentifiable} case(s) skipped -- identity not derivable",
            file=sys.stderr,
        )
    return len(run_rows)


def v2_benchmarks_already_ingested(
    client, db: str, run_id: str, component: str
) -> bool:
    """Have this run's benchmark rows already landed?

    benchmark_runs is a plain MergeTree with no dedup key, so a double ingest of one leg
    doubles every number BEHIND A MEAN -- which is worse than a doubled count, because a
    mean of duplicated samples looks plausible. Scoped to (component, run_id) and not to a
    source file: unlike a sharded JUnit run, one perf leg reports its benchmarks in a single
    ingest, so the run IS the unit.
    """
    rows = client.query(
        f"SELECT count() FROM {schema.BENCHMARK_RUNS.qualified(db)} "
        "WHERE component = {component:String} AND run_id = {run_id:UUID}",
        parameters={"component": component, "run_id": run_id},
    ).result_rows
    return bool(rows and rows[0][0] > 0)


def insert_benchmarks_v2(
    client, db: str, component: str, run_id: str, benchmarks: list
) -> int:
    """Write benchmarks (identity) + benchmark_runs (measurements) for one leg.

    Each entry is a dict: name, tags, props, backend, measurements, iterations, and `disc`
    plus `disc_keys` for the identity discriminators (see v2_benchmark_id). One row per
    (benchmark, backend) -- every metric of one benchmark belongs in the measurements Map of a
    single row, since one row per metric multiplies every trend point by the metric count.
    """
    if not benchmarks:
        return 0
    ident_rows, facts = {}, {}
    skipped_unidentifiable = 0
    for b in benchmarks:
        name, tags = b.get("name", ""), b.get("tags") or []
        disc = b.get("disc") or {}
        bid = v2_benchmark_id(component, name, tags, disc, b.get("disc_keys") or ())
        if not bid:
            # Refused identity: writing it anyway would collide this benchmark with every
            # other unidentifiable one rather than merely orphaning it.
            skipped_unidentifiable += 1
            continue
        backend = b.get("backend", "")
        # Keyed by id: the same benchmark reported by two files is one identity, and the
        # richer props win so a merge cannot drop a field the other side set.
        prev = ident_rows.get(bid)
        props = {k: str(v) for k, v in (b.get("props") or {}).items() if v != ""}
        if prev:
            merged = dict(prev["props"])
            merged.update(props)
            props = merged
        ident_rows[bid] = {
            "benchmark_id": bid,
            "component": component,
            "name": name,
            "tags": sorted({t for t in tags if t}),
            "props": props,
        }
        fact = facts.setdefault(
            (bid, backend),
            {
                "run_id": run_id,
                "benchmark_id": bid,
                "component": component,
                "backend": backend,
                "measurements": {},
                "iterations": int(b.get("iterations") or 0),
                "props": {k: str(v) for k, v in (b.get("run_props") or {}).items()},
            },
        )
        fact["measurements"].update(b.get("measurements") or {})
        fact["iterations"] = max(fact["iterations"], int(b.get("iterations") or 0))
    # The DDL's CHECK refuses an empty map, so an unmeasured benchmark would fail the whole
    # insert -- and a 720-minute perf leg must not die on a parse gap. Dropped with a warning.
    run_rows = [f for f in facts.values() if f["measurements"]]
    dropped = len(facts) - len(run_rows)
    kept = {f["benchmark_id"] for f in run_rows}
    schema.insert_identities(
        client,
        schema.BENCHMARKS,
        {k: v for k, v in ident_rows.items() if k in kept},
        db=db,
    )
    schema.insert(client, schema.BENCHMARK_RUNS, run_rows, db=db)
    if skipped_unidentifiable:
        print(
            f"  [warn] v2: {skipped_unidentifiable} benchmark(s) skipped -- identity not "
            "derivable",
            file=sys.stderr,
        )
    if dropped:
        print(
            f"  [warn] v2: {dropped} benchmark(s) skipped -- no measurements parsed",
            file=sys.stderr,
        )
    return len(run_rows)
