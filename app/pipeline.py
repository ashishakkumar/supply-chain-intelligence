"""Raw CSV to curated Postgres.

Layers: raw files on disk, curated rows from app.dq, then the serving tables.
The seed is idempotent. A second run sees ports already filled and changes nothing.
A failure rolls the insert back, so a crashed first seed leaves the tables empty.
Raw files are never modified. Startup of the API does not call this module.
"""

import csv
import os
from collections.abc import Callable
from pathlib import Path

from sqlalchemy import insert, text
from sqlalchemy.orm import Session, sessionmaker

from app.db import (
    Event,
    Port,
    QuarantineEvent,
    QuarantineShipment,
    Shipment,
    engine_for,
)
from app.dq import curate_events, curate_ports, curate_shipments
from app.migrate import MIGRATIONS, apply


def load(
    database_url: str,
    input_dir: Path,
    before_commit: Callable[[Session], None] | None = None,
) -> bool:
    """Load cleaned CSVs into Postgres once. Return True only when this call inserted rows.

    Layered pipeline, derived fields, idempotent.
    Steps:
    1. Apply any SQL files that have not run yet (creates tables, does not load them).
    2. If ports already has a row, stop. A second seed does not duplicate or rewrite.
    3. Curate ports, then shipments, then events. Critical DQ rows go to quarantine.
    4. Insert curated and quarantine rows in one transaction and commit.
    5. On any error, roll back and re-raise. before_commit exists so a test can
       force that failure and prove the rollback.
    """
    apply(database_url, MIGRATIONS)
    engine = engine_for(database_url)
    factory = sessionmaker(bind=engine)
    session = factory()
    try:
        if session.scalar(text("SELECT 1 FROM ports LIMIT 1")) is not None:
            return False
        ports = curate_ports(_read_csv(input_dir / "ports.csv"))
        port_codes = {str(port["port_code"]) for port in ports}
        shipments, quarantined_shipments = curate_shipments(
            _read_csv(input_dir / "shipments.csv"),
            port_codes,
        )
        events, quarantined_events = curate_events(
            _read_csv(input_dir / "port_events.csv"),
            port_codes,
        )
        _insert(session, Port, ports)
        _insert(session, Shipment, shipments)
        _insert(session, Event, events)
        _insert(
            session,
            QuarantineShipment,
            [
                {
                    "shipment_id": str(item["raw"]["shipment_id"]),
                    "reason": item["reason"],
                    "raw": item["raw"],
                }
                for item in quarantined_shipments
            ],
        )
        _insert(
            session,
            QuarantineEvent,
            [
                {
                    "event_id": str(item["raw"]["event_id"]),
                    "reason": item["reason"],
                    "raw": item["raw"],
                }
                for item in quarantined_events
            ],
        )
        if before_commit is not None:
            before_commit(session)
        session.commit()
        return True
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _insert(session: Session, model: type, rows: list[dict[str, object]]) -> None:
    """Insert one batch into a curated or quarantine table. An empty batch is a no-op."""
    if not rows:
        return
    session.execute(insert(model), rows)


def main() -> int:
    """Entry point for `uv run db-seed`.

    Reads DATABASE_URL and DATA_DIR (default data/raw). Prints "seeded" or
    "already seeded". Does not start the API.
    """
    seeded = load(os.environ["DATABASE_URL"], Path(os.environ.get("DATA_DIR", "data/raw")))
    print("seeded" if seeded else "already seeded")
    return 0


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read one raw CSV as a list of string dicts. The file is not rewritten.

    Shared with the DQ report endpoint so both see the same raw rows.
    Quality checks run against the CSVs, not against a store.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    raise SystemExit(main())
