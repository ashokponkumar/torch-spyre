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

"""optimize_hypotheses: the newest version per hypothesis is its state, and bad rows fail the insert."""

import pytest
from spyre_clickhouse_ingest.apply_schema import SCHEMA_DIR, SchemaApplier


@pytest.fixture
def chdb_session():
    session = pytest.importorskip("chdb.session")
    s = session.Session()
    s.query("CREATE DATABASE IF NOT EXISTS t ENGINE = Atomic")
    s.query("USE t")
    for stmt in SchemaApplier.statements(
        (SCHEMA_DIR / "77-optimize-hypotheses.sql").read_text()
    ):
        s.query(stmt)
    yield s
    s.cleanup()


def _rows(s, sql):
    out = s.query(sql, "TabSeparated").data()
    out = out.decode() if isinstance(out, bytes) else out
    return [line.split("\t") for line in out.splitlines() if line]


def _insert(s, values):
    s.query(
        f"INSERT INTO optimize_hypotheses (hypothesis_id, version, state, origin, statement, lever) VALUES {values}"
    )


def test_view_reports_the_newest_version(chdb_session):
    _insert(
        chdb_session,
        "('H-0a0a0a', 1, 'draft', 'link', 'idea', ''), ('H-0a0a0a', 3, 'refuted', 'link', 'idea', 'faster'), "
        "('H-0a0a0a', 2, 'survived', 'link', 'idea', 'faster'), ('H-0b0b0b', 1, 'draft', 'chat', 'other', '')",
    )
    rows = _rows(
        chdb_session,
        "SELECT hypothesis_id, latest_version, current_state, lever_kind FROM v_optimize_hypotheses ORDER BY hypothesis_id",
    )
    assert rows == [
        ["H-0a0a0a", "3", "refuted", "faster"],
        ["H-0b0b0b", "1", "draft", ""],
    ]


@pytest.mark.parametrize(
    "values",
    [
        "('bad', 1, 'draft', 'chat', 'x', '')",
        "('H-0c0c0c', 1, 'magic', 'chat', 'x', '')",
        "('H-0c0c0c', 1, 'draft', 'email', 'x', '')",
        "('H-0c0c0c', 1, 'draft', 'chat', 'x', 'more-cards')",
        "('H-0c0c0c', 1, 'draft', 'chat', '', '')",
    ],
)
def test_constraints_reject_bad_rows(chdb_session, values):
    with pytest.raises(Exception, match="VIOLATED_CONSTRAINT|violated"):
        _insert(chdb_session, values)
