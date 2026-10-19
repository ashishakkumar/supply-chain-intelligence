import csv
import os
from collections.abc import Callable
from pathlib import Path

from sqlalchemy import insert, text
from sqlalchemy.orm import Session, sessionmaker

from app.db import (
    Base,
    Event,
    Port,
    QuarantineEvent,
    QuarantineShipment,
    Shipment,
    engine_for,
)
from app.dq import curate_events, curate_ports, curate_shipments

WIPE = """
TRUNCATE TABLE
    port_events,
    shipments,
    quarantine_events,
    quarantine_shipments,
    ports
RESTART IDENTITY CASCADE
"""


def load(
    database_url: str,
    input_dir: Path,
    before_commit: Callable[[Session], None] | None = None,
) -> None:
    engine = engine_for(database_url)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    session = factory()
    try:
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
        session.execute(text(WIPE))
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
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _insert(session: Session, model: type, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    session.execute(insert(model), rows)


def main() -> int:
    load(os.environ["DATABASE_URL"], Path(os.environ.get("DATA_DIR", "data/raw")))
    return 0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    raise SystemExit(main())
