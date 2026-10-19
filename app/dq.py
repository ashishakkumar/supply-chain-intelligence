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
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _check(
    name: str,
    rows_affected: int,
    total: int,
    severity: str,
    action: str,
) -> dict[str, object]:
    percentage = 0.0 if total == 0 else round(rows_affected / total * 100, 2)
    return {
        "name": name,
        "rows_affected": rows_affected,
        "percentage": percentage,
        "severity": severity,
        "action": action,
    }


def _duplicate_rows(rows: list[dict[str, str]]) -> tuple[int, int]:
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
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["event_id"]].append(row)
    affected = 0
    for group in groups.values():
        if len(group) > 1 and not all(item == group[0] for item in group):
            affected += len(group)
    return affected


def _unknown_port(row: dict[str, str], port_codes: set[str]) -> bool:
    return row["origin_port"] not in port_codes or row["destination_port"] not in port_codes


def _cargo_dirty(value: str) -> bool:
    canonical = _cargo(value)
    if canonical is None:
        return True
    return value != canonical


def _container(row: dict[str, str]) -> int:
    return int(float(row["container_count"]))


def _negative_weight(value: str) -> bool:
    text = value.strip()
    return bool(text) and float(text) < 0


def _missing_actuals(row: dict[str, str]) -> bool:
    return not row["actual_departure"].strip() and not row["actual_arrival"].strip()


def _departed_before_booking(row: dict[str, str]) -> bool:
    departure = _timestamp(row["actual_departure"])
    booking = _timestamp(row["booking_date"])
    return departure is not None and booking is not None and departure < booking


def _early(row: dict[str, str]) -> bool:
    actual = _timestamp(row["actual_arrival"])
    planned = _timestamp(row["planned_arrival"])
    return actual is not None and planned is not None and actual < planned


def _timestamp(value: str) -> datetime | None:
    text = (value or "").strip()
    if not text:
        return None
    return datetime.strptime(text, TIMESTAMP_FORMAT)


def _status(value: str) -> str:
    return CANONICAL_STATUS.get(value.strip().casefold(), "UNKNOWN")


def _cargo(value: str) -> str | None:
    key = value.strip().casefold()
    if not key:
        return None
    return CANONICAL_CARGO.get(key, value.strip())


def _weight(value: str) -> float | None:
    text = value.strip()
    if not text:
        return None
    number = float(text)
    if number < 0:
        return None
    return number


def _clean_shipment(row: dict[str, str]) -> dict[str, object]:
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
    return {"reason": reason, "raw": row}


if __name__ == "__main__":
    raise SystemExit(main())
