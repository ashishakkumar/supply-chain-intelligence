"""Apply the SQL files in db/migrations.

The API process creates any missing tables. It does not load or empty them.
applied_changes records each filename so a file runs once. There is no Alembic.
"""

import os
from pathlib import Path

from sqlalchemy import text

from app.db import engine_for

ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS = ROOT / "db" / "migrations"

_CREATE_APPLIED = """
CREATE TABLE IF NOT EXISTS applied_changes (
    filename text PRIMARY KEY,
    applied_at timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""


def _statements(sql: str) -> list[str]:
    """Split one migration file into statements on ';'.

    Each file is one table: ports, shipments, port_events, quarantine_shipments,
    quarantine_events. A semicolon inside a string literal would split wrong.
    """
    # ponytail: split on ';'. Breaks if a string literal contains one. Upgrade path: a SQL parser.
    return [part.strip() for part in sql.split(";") if part.strip()]


def apply(database_url: str, migrations_dir: Path) -> list[str]:
    """Run pending *.sql files in filename order. Return the names that ran.

    Creates applied_changes if needed, skips files already recorded, and inserts
    the filename in the same transaction as its statements. Called by db-migrate
    and by the seed command before it looks at the CSVs.
    """
    engine = engine_for(database_url)
    ran: list[str] = []
    try:
        with engine.begin() as conn:
            conn.execute(text(_CREATE_APPLIED))
            applied = set(conn.execute(text("SELECT filename FROM applied_changes")).scalars())
            if not migrations_dir.is_dir():
                return ran
            for path in sorted(migrations_dir.glob("*.sql")):
                if path.name in applied:
                    continue
                for statement in _statements(path.read_text()):
                    conn.execute(text(statement))
                conn.execute(
                    text("INSERT INTO applied_changes (filename) VALUES (:filename)"),
                    {"filename": path.name},
                )
                ran.append(path.name)
    finally:
        engine.dispose()
    return ran


def main() -> int:
    """Entry point for `uv run db-migrate`. Prints each applied file, or nothing pending."""
    url = os.environ["DATABASE_URL"]
    ran = apply(url, MIGRATIONS)
    if not ran:
        print("no pending changes")
        return 0
    for name in ran:
        print(f"applied {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
