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

"""Apply schema/*.sql to a ClickHouse database, in filename order.

The DDL files are the ONLY definition of these tables. The alternative this replaces -- an
ALTER ... ADD COLUMN IF NOT EXISTS pass run from the ingest on every insert -- is how
hw_failure_diagnostics reached 45 columns with an empty sorting key and no checked-in shape:
each writer added what it needed, nothing declared the whole, and the accumulated result was
not reproducible from anything in the repo.

Every statement is CREATE ... IF NOT EXISTS, so applying is idempotent and safe to re-run. It
does NOT alter an existing table: a table whose live shape has diverged is reported, not
silently patched, because reconciling it needs a decision (an ORDER BY cannot be changed in
place, so it means DROP+CREATE or a rewrite through a temp table).
"""

import argparse
import re
import sys
from pathlib import Path

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schema"

# A trailing ';' is optional in the files, and two of them carry ';' inside comment prose, so
# statements are split after comments are stripped rather than on every ';' in the text.
_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def sql_files(schema_dir: Path = SCHEMA_DIR) -> list:
    """The DDL files in apply order, which is filename order (the numeric prefix encodes the
    dependency: a view cannot precede the table it selects from)."""
    return sorted(schema_dir.glob("*.sql"))


def statements(text: str) -> list:
    """The executable statements in one file, comments removed.

    clickhouse_connect sends one statement per command(), so a multi-statement file must be
    split; splitting the RAW text breaks on a ';' that appears inside a comment.
    """
    stripped = _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub("", text))
    return [s.strip() for s in stripped.split(";") if s.strip()]


def apply_file(client, path: Path, dry_run: bool = False) -> int:
    """Execute one DDL file. Returns the number of statements applied."""
    stmts = statements(path.read_text())
    for stmt in stmts:
        if dry_run:
            print(f"    {stmt.splitlines()[0][:100]}")
            continue
        client.command(stmt)
    return len(stmts)


def apply_all(client, schema_dir: Path = SCHEMA_DIR, dry_run: bool = False) -> int:
    """Apply every DDL file in order. Returns the total statements applied.

    No try/except: a DDL failure means the database does not have the shape the writers assume,
    and continuing past it would produce exactly the partially-migrated state this module
    exists to prevent.
    """
    total = 0
    for path in sql_files(schema_dir):
        stmts = apply_file(client, path, dry_run=dry_run)
        total += stmts
        print(f"  {path.name:34} {stmts} statement(s)")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the v2 schema DDL to a ClickHouse database"
    )
    parser.add_argument(
        "--schema-dir",
        type=Path,
        default=SCHEMA_DIR,
        help=f"Directory of *.sql files (default: {SCHEMA_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the statements that would run, without connecting",
    )
    args = parser.parse_args()

    files = sql_files(args.schema_dir)
    if not files:
        print(f"[error] No .sql files in {args.schema_dir}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print(f"[dry-run] Would apply {len(files)} file(s) from {args.schema_dir}:")
        apply_all(None, args.schema_dir, dry_run=True)
        return

    from .client import client_summary, get_client

    print(f"[info] Applying {len(files)} file(s) to {client_summary()} ...")
    client = get_client()
    total = apply_all(client, args.schema_dir)
    print(f"[info] Applied {total} statement(s) from {len(files)} file(s).")


if __name__ == "__main__":
    main()
