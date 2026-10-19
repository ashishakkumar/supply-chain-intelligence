from collections.abc import Mapping
from datetime import datetime

QUARTER_END_MONTH = {1: 4, 2: 7, 3: 10}


def delay_hours(actual: datetime | None, planned: datetime | None) -> float | None:
    if actual is None or planned is None:
        return None
    return (actual - planned).total_seconds() / 3600


def on_time_flag(delay: float | None) -> bool | None:
    if delay is None:
        return None
    return delay <= 24


def transit_days(end: datetime | None, start: datetime | None) -> float | None:
    if end is None or start is None:
        return None
    return (end - start).total_seconds() / 86400


def route_key(origin: str, destination: str) -> str:
    return f"{origin} → {destination}"


def route_stats(delays: list[float | None]) -> dict[str, float | int | None]:
    known = [delay for delay in delays if delay is not None]
    if not known:
        average = None
        rate = None
    else:
        average = sum(known) / len(known)
        rate = sum(1 for delay in known if on_time_flag(delay)) / len(known)
    return {
        "shipment_count": len(delays),
        "average_delay_hours": average,
        "on_time_rate": rate,
    }


def resolve_port(token: str, ports: list[Mapping[str, str]]) -> str | None:
    stripped = token.strip()
    if any(port["port_code"] == stripped for port in ports):
        return stripped
    matches = [
        port["port_code"]
        for port in ports
        if port["port_name"].casefold() == stripped.casefold()
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def rank_routes(
    rows: list[dict[str, object]],
    start: datetime,
    end: datetime,
    limit: int = 20,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[float | None]] = {}
    for row in rows:
        booking = row["booking_date"]
        if not isinstance(booking, datetime) or booking < start or booking >= end:
            continue
        origin = str(row["origin"])
        destination = str(row["destination"])
        delay = row["delay"]
        grouped.setdefault((origin, destination), []).append(
            None if delay is None else float(delay)
        )
    ranked: list[dict[str, object]] = []
    for (origin, destination), delays in grouped.items():
        stats = route_stats(delays)
        average = stats["average_delay_hours"]
        if average is None:
            continue
        ranked.append(
            {
                "origin": origin,
                "destination": destination,
                "route_key": route_key(origin, destination),
                "average_delay_hours": average,
                "on_time_rate": stats["on_time_rate"],
                "shipment_count": stats["shipment_count"],
            }
        )
    ranked.sort(key=lambda item: float(item["average_delay_hours"]), reverse=True)
    return ranked[:limit]


def latest_quarter(
    bookings: list[datetime],
    now: datetime | None = None,
) -> dict[str, datetime | int]:
    """Return the quarter of the latest booking. The clock is ignored on purpose."""
    del now
    latest = max(bookings)
    quarter = (latest.month - 1) // 3 + 1
    start_month = (quarter - 1) * 3 + 1
    start = datetime(latest.year, start_month, 1)
    if quarter == 4:
        end = datetime(latest.year + 1, 1, 1)
    else:
        end = datetime(latest.year, QUARTER_END_MONTH[quarter], 1)
    return {"year": latest.year, "quarter": quarter, "start": start, "end": end}
