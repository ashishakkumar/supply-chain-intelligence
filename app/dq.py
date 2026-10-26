"""Standalone data-quality checks and the curated-row builders.

Runnable without Postgres: python -m app.dq --input data/raw
writes data/quality/report.json. GET /data-quality/report calls build_report
on the same CSVs. The checks do not insert anything.

curate_ports, curate_shipments, and curate_events apply the decisions in that
report when the seed runs: quarantine the critical rows, fix the warning rows
we chose to fix, and flag-and-keep the rest by storing the cleaned value.
"""

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from app.domain import delay_hours, on_time_flag, route_key, transit_days

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
CANONICAL_STATUS = {
    "delivered": "DELIVERED",
    "delayed": "DELAYED",
    "cancelled": "CANCELLED",
    "completed": "COMPLETED",
    "complete": "COMPLETED",
    "": "UNKNOWN",
    "n/a": "UNKNOWN",
}
CANONICAL_CARGO = {
    "furniture": "Furniture",
    "automotive parts": "Automotive Parts",
    "machinery": "Machinery",
    "pharmaceuticals": "Pharmaceuticals",
    "electronics": "Electronics",
    "chemicals": "Chemicals",
    "consumer goods": "Consumer Goods",
    "apparel": "Apparel",
    "food & beverage": "Food & Beverage",
    "raw materials": "Raw Materials",
}
SENTINEL_VESSELS = {"VSL-0000", "VSL-9999"}
RAW_CANONICAL_STATUS = {"DELIVERED", "DELAYED", "CANCELLED"}


def build_report(
    shipments: list[dict[str, str]],
    ports: list[dict[str, str]],
    events: list[dict[str, str]],
) -> list[dict[str, object]]:
    """One record per DQ issue: name, rows, percentage, severity, action.

    Percentages use the file the check belongs to (shipments, ports, or events),
    not a mix of the three. Severity is critical, warning, or info.
    Critical means quarantine. Warning means fix or flag-and-keep. Info means
    leave it (early arrival is a real early ship; blank notes are allowed).

    The twelve-plus issues the files contain are all listed here. This function
    only counts. It does not drop, fix, or write a row.
    """
    port_codes = {port["port_code"] for port in ports}
    shipment_total = len(shipments)
    port_total = len(ports)
    event_total = len(events)
    identical, conflicting = _duplicate_rows(shipments)
    return [
        _check(
            "identical_duplicate_shipment_id",
            identical,
            shipment_total,
            "warning",
            "Keep one copy of each identical shipment id.",
        ),
        _check(
            "conflicting_duplicate_shipment_id",
            conflicting,
            shipment_total,
            "critical",
            "Quarantine every row when the same shipment id has different values.",
        ),
        _check(
            "unknown_port",
            sum(1 for row in shipments if _unknown_port(row, port_codes)),
            shipment_total,
            "critical",
            "Quarantine the shipment. The port code is not in the port file.",
        ),
        _check(
            "noncanonical_status",
            sum(1 for row in shipments if row["status"] not in RAW_CANONICAL_STATUS),
            shipment_total,
            "warning",
            "Map delivered to DELIVERED, Complete and COMPLETED to COMPLETED, blank and N/A to UNKNOWN.",
        ),
        _check(
            "noncanonical_cargo",
            sum(1 for row in shipments if _cargo_dirty(row["cargo_type"])),
            shipment_total,
            "warning",
            "Strip and map cargo to one canonical label. Leave a blank cargo null.",
        ),
        _check(
            "zero_container_count",
            sum(1 for row in shipments if _container(row) == 0),
            shipment_total,
            "warning",
            "Flag and keep the raw value.",
        ),
        _check(
            "extreme_container_count",
            sum(1 for row in shipments if _container(row) > 40),
            shipment_total,
            "warning",
            "Flag and keep the raw value. The model clips this feature at 40.",
        ),
        _check(
            "missing_weight",
            sum(1 for row in shipments if not row["weight_tons"].strip()),
            shipment_total,
            "warning",
            "Leave the weight null.",
        ),
        _check(
            "negative_weight",
            sum(1 for row in shipments if _negative_weight(row["weight_tons"])),
            shipment_total,
            "warning",
            "Set the weight to null.",
        ),
        _check(
            "missing_actual_timestamps",
            sum(1 for row in shipments if _missing_actuals(row)),
            shipment_total,
            "warning",
            "Keep both timestamps null. Do not impute them.",
        ),
        _check(
            "actual_departure_before_booking",
            sum(1 for row in shipments if _departed_before_booking(row)),
            shipment_total,
            "warning",
            "Flag and keep the timestamps.",
        ),
        _check(
            "early_arrival",
            sum(1 for row in shipments if _early(row)),
            shipment_total,
            "info",
            "none",
        ),
        _check(
            "missing_port_country",
            sum(1 for port in ports if not port["country"].strip()),
            port_total,
            "warning",
            "Flag the blank country. Do not fill a value that is not in the file.",
        ),
        _check(
            "conflicting_event_id",
            _conflicting_event_rows(events),
            event_total,
            "critical",
            "Quarantine every row with that event id. Do not pick a winner.",
        ),
        _check(
            "blank_event_type",
            sum(1 for event in events if not event["event_type"].strip()),
            event_total,
            "warning",
            "Keep the row and store a null event type.",
        ),
        _check(
            "blank_event_notes",
            sum(1 for event in events if not event["notes"].strip()),
            event_total,
            "info",
            "Allow a null note.",
        ),
        _check(
            "sentinel_vessel",
            sum(1 for event in events if event["vessel_id"] in SENTINEL_VESSELS),
            event_total,
            "warning",
            "Flag and keep the event. The vessel is not on a shipment.",
        ),
    ]


def curate_ports(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Clean ports for the curated ports table.

    A blank country stays null. BEANR is the one row. Do not fill a country
    that is not in the file. Congestion is stored as a float for the check
    constraint and for the model features.
    """
    return [
        {
            "port_code": row["port_code"],
            "port_name": row["port_name"],
            "country": row["country"].strip() or None,
            "region": row["region"],
            "timezone": row["timezone"],
            "avg_congestion_score": float(row["avg_congestion_score"]),
        }
        for row in rows
    ]


def curate_shipments(
    rows: list[dict[str, str]],
    port_codes: set[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Split raw shipments into curated rows and quarantine rows.

    Steps, in file order so a rerun keeps the same row:
    1. Group by shipment_id.
    2. Conflicting duplicates: quarantine every row. Do not pick a winner.
    3. Identical duplicates: keep one copy. The extra copies are not quarantined.
    4. Unknown origin or destination (XXTST, ZZZZZ, UNKNW, or any code missing
       from ports): quarantine the shipment.
    5. Otherwise clean the row: canonical status and cargo, null bad weights,
       and the derived delay, on-time, route, and transit fields.

    Returns (kept, quarantined).
    """
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    order: list[str] = []
    for row in rows:
        if row["shipment_id"] not in groups:
            order.append(row["shipment_id"])
        groups[row["shipment_id"]].append(row)

    kept: list[dict[str, object]] = []
    quarantined: list[dict[str, object]] = []
    for shipment_id in order:
        group = groups[shipment_id]
        if len(group) > 1 and not all(item == group[0] for item in group):
            quarantined.extend(
                _quarantine(item, "conflicting_shipment_id") for item in group
            )
            continue
        candidate = group[0]
        if _unknown_port(candidate, port_codes):
            quarantined.append(_quarantine(candidate, "unknown_port"))
            continue
        kept.append(_clean_shipment(candidate))
    return kept, quarantined


def curate_events(
    rows: list[dict[str, str]],
    port_codes: set[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Split raw port events into curated rows and quarantine rows.

    Conflicting duplicate event ids: quarantine every row with that id.
    A port code that is not in the port file: quarantine that event.
    Blank event type and blank notes become null and the row is kept.
    Sentinel vessels VSL-0000 and VSL-9999 are kept; the check only counts them.
    """
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    order: list[str] = []
    for row in rows:
        if row["event_id"] not in groups:
            order.append(row["event_id"])
        groups[row["event_id"]].append(row)

    kept: list[dict[str, object]] = []
    quarantined: list[dict[str, object]] = []
    for event_id in order:
        group = groups[event_id]
        if len(group) > 1 and not all(item == group[0] for item in group):
            quarantined.extend(
                _quarantine(item, "conflicting_event_id") for item in group
            )
            continue
        candidate = group[0]
        if candidate["port_code"] not in port_codes:
            quarantined.append(_quarantine(candidate, "unknown_port"))
            continue
        kept.append(_clean_event(candidate))
    return kept, quarantined


def main(argv: list[str] | None = None) -> int:
    """CLI for the standalone DQ step.

    python -m app.dq --input data/raw
    Optional --output, default data/quality/report.json.
    Does not connect to Postgres and does not depend on the seed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/quality/report.json")
    args = parser.parse_args(argv)
    directory = Path(args.input)
    report = {
        "checks": build_report(
            _read_csv(directory / "shipments.csv"),
            _read_csv(directory / "ports.csv"),
            _read_csv(directory / "port_events.csv"),
        )
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read one raw CSV. Used only by the standalone command, not by the API."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _check(
    name: str,
    rows_affected: int,
    total: int,
    severity: str,
    action: str,
) -> dict[str, object]:
    """One check object for the JSON report.

    percentage is rows_affected / total * 100, rounded to 2 decimals.
    total is the row count of the file the check belongs to. An empty file
    reports 0.0 so the command still runs.
    """
    percentage = 0.0 if total == 0 else round(rows_affected / total * 100, 2)
    return {
        "name": name,
        "rows_affected": rows_affected,
        "percentage": percentage,
        "severity": severity,
        "action": action,
    }


def _duplicate_rows(rows: list[dict[str, str]]) -> tuple[int, int]:
    """Count identical vs conflicting duplicate shipment ids.

    Identical: same id, same values. Warning, keep one copy.
    Conflicting: same id, different values. Critical, quarantine both rows.
    The counts are rows, not distinct ids.
    """
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["shipment_id"]].append(row)
    identical = 0
    conflicting = 0
    for group in groups.values():
        if len(group) < 2:
            continue
        if all(item == group[0] for item in group):
            identical += len(group)
        else:
            conflicting += len(group)
    return identical, conflicting


def _conflicting_event_rows(rows: list[dict[str, str]]) -> int:
    """Rows whose event_id is shared with a different payload.

    Critical. Quarantine every row with that id. event_id cannot be the
    idempotency key for a later streaming load, because these payloads differ.
    """
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["event_id"]].append(row)
    affected = 0
    for group in groups.values():
        if len(group) > 1 and not all(item == group[0] for item in group):
            affected += len(group)
    return affected


def _unknown_port(row: dict[str, str], port_codes: set[str]) -> bool:
    """True when origin or destination is not a port_code in ports.csv.

    The planted codes are XXTST, ZZZZZ, and UNKNW. Critical: quarantine.
    """
    return row["origin_port"] not in port_codes or row["destination_port"] not in port_codes


def _cargo_dirty(value: str) -> bool:
    """True when cargo is blank, or its case or whitespace is not the canonical label.

    Warning. The curated value is the canonical label. A blank stays null.
    """
    canonical = _cargo(value)
    if canonical is None:
        return True
    return value != canonical


def _container(row: dict[str, str]) -> int:
    """Container count as an int. Zero and values above 40 are warnings, not drops.

    The model clips the feature at 40. The curated column keeps the raw count.
    """
    return int(float(row["container_count"]))


def _negative_weight(value: str) -> bool:
    """True when weight_tons is present and below zero. Warning: store null."""
    text = value.strip()
    return bool(text) and float(text) < 0


def _missing_actuals(row: dict[str, str]) -> bool:
    """True when both actual departure and actual arrival are blank.

    Warning. Leave both null. Do not impute. Delay stays null, so the row
    is not a training label.
    """
    return not row["actual_departure"].strip() and not row["actual_arrival"].strip()


def _departed_before_booking(row: dict[str, str]) -> bool:
    """True when actual departure is earlier than booking_date.

    Warning. Flag and keep the timestamps. Booking date is still the feature.
    """
    departure = _timestamp(row["actual_departure"])
    booking = _timestamp(row["booking_date"])
    return departure is not None and booking is not None and departure < booking


def _early(row: dict[str, str]) -> bool:
    """True when actual arrival is before planned arrival. Info: an early ship is on time."""
    actual = _timestamp(row["actual_arrival"])
    planned = _timestamp(row["planned_arrival"])
    return actual is not None and planned is not None and actual < planned


def _timestamp(value: str) -> datetime | None:
    """Parse 'YYYY-MM-DD HH:MM:SS' from a CSV cell. Blank is null. No timezone."""
    text = (value or "").strip()
    if not text:
        return None
    return datetime.strptime(text, TIMESTAMP_FORMAT)


def _status(value: str) -> str:
    """Map a raw status onto the canonical label. Warning.

    delivered -> DELIVERED. Complete and COMPLETED -> COMPLETED.
    Blank and N/A -> UNKNOWN. Anything else -> UNKNOWN.
    DELIVERED, DELAYED, and CANCELLED are already canonical.
    """
    return CANONICAL_STATUS.get(value.strip().casefold(), "UNKNOWN")


def _cargo(value: str) -> str | None:
    """Canonical cargo label, or null when the cell is blank. Warning."""
    key = value.strip().casefold()
    if not key:
        return None
    return CANONICAL_CARGO.get(key, value.strip())


def _weight(value: str) -> float | None:
    """Weight in tons. Blank and negative both become null. A positive value is kept."""
    text = value.strip()
    if not text:
        return None
    number = float(text)
    if number < 0:
        return None
    return number


def _clean_shipment(row: dict[str, str]) -> dict[str, object]:
    """One curated shipment, including the five derived fields.

    actual_delay_hours, on_time_flag, route_key, transit_days_planned,
    transit_days_actual. Status, cargo, and weight are the cleaned values.
    Container count is the raw integer, including 0 and values above 40.
    """
    actual_arrival = _timestamp(row["actual_arrival"])
    planned_arrival = _timestamp(row["planned_arrival"])
    delay = delay_hours(actual_arrival, planned_arrival)
    return {
        "shipment_id": row["shipment_id"],
        "customer_id": row["customer_id"],
        "origin_port": row["origin_port"],
        "destination_port": row["destination_port"],
        "vessel_id": row["vessel_id"],
        "booking_date": _timestamp(row["booking_date"]),
        "planned_departure": _timestamp(row["planned_departure"]),
        "actual_departure": _timestamp(row["actual_departure"]),
        "planned_arrival": planned_arrival,
        "actual_arrival": actual_arrival,
        "container_count": _container(row),
        "cargo_type": _cargo(row["cargo_type"]),
        "weight_tons": _weight(row["weight_tons"]),
        "status": _status(row["status"]),
        "actual_delay_hours": delay,
        "on_time_flag": on_time_flag(delay),
        "route_key": route_key(row["origin_port"], row["destination_port"]),
        "transit_days_planned": transit_days(
            planned_arrival,
            _timestamp(row["planned_departure"]),
        ),
        "transit_days_actual": transit_days(
            actual_arrival,
            _timestamp(row["actual_departure"]),
        ),
    }


def _clean_event(row: dict[str, str]) -> dict[str, object]:
    """One curated port event. Blank type and blank notes are null. The vessel is kept."""
    return {
        "event_id": row["event_id"],
        "event_timestamp": _timestamp(row["event_timestamp"]),
        "port_code": row["port_code"],
        "vessel_id": row["vessel_id"],
        "event_type": row["event_type"].strip() or None,
        "delay_minutes": int(float(row["delay_minutes"])),
        "notes": row["notes"].strip() or None,
    }


def _quarantine(row: dict[str, str], reason: str) -> dict[str, object]:
    """Wrap a raw CSV row with the reason it was kept out of the curated table.

    reason is conflicting_shipment_id, unknown_port, or conflicting_event_id.
    The raw dict is stored as JSON so the original values are still inspectable.
    """
    return {"reason": reason, "raw": row}


if __name__ == "__main__":
    raise SystemExit(main())
