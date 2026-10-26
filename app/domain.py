"""
This module contains the domain logic for the application.
It is used to define the domain entities and the logic for the application.
What is domain logic?
Domain logic is the business logic for the application.
It is used to define the business rules for the application.
why is it important?
It is important because it helps to keep the business logic separate from the technical logic.
It helps to keep the code organized and easy to understand.
It helps to avoid code duplication.
It helps to make the code more maintainable.
It helps to make the code more testable.
It helps to make the code more reusable.
It helps to make the code more scalable.

Derived fields live here, with no database and no HTTP:
actual_delay_hours, on_time_flag, route_key, transit_days_planned, transit_days_actual.
Route stats and the assistant's "highest delay this quarter" ranking use the
same functions, so the API and the tools cannot disagree on the definition of on time.
"""

from collections.abc import Mapping
from datetime import datetime

QUARTER_END_MONTH = {1: 4, 2: 7, 3: 10}


def delay_hours(actual: datetime | None, planned: datetime | None) -> float | None:
    """actual_delay_hours: actual arrival minus planned arrival, in hours.

    Null-safe. If either timestamp is missing the delay is null. Do not impute it.
    The ML label is this value above 24 hours. The hours themselves are not a
    model feature, because actual arrival is not known at booking time.
    """
    if actual is None or planned is None:
        return None
    return (actual - planned).total_seconds() / 3600 ## why this exists ? because we want to get the delay in hours , the csv
    # is in seconds, so we need to convert it to hours


def on_time_flag(delay: float | None) -> bool | None:
    """on_time_flag: true when delay is at most 24 hours.

    Null delay stays null. An early ship has a negative delay, so it is on time.
    The model predicts the opposite case: arrival more than 24 hours late.
    """
    if delay is None:
        return None
    return delay <= 24


def transit_days(end: datetime | None, start: datetime | None) -> float | None:
    """Days between two timestamps, or null if either side is missing.

    Used twice on a curated shipment:
    transit_days_planned is planned arrival minus planned departure.
    transit_days_actual is actual arrival minus actual departure.
    Only the planned value is a booking-time model feature.
    """
    if end is None or start is None:
        return None
    return (end - start).total_seconds() / 86400


def route_key(origin: str, destination: str) -> str:
    """route_key: origin port, an arrow, destination port.

    A display label, not a join key. The real keys are the two port codes.
    """
    return f"{origin} → {destination}"


def route_stats(delays: list[float | None]) -> dict[str, float | int | None]:
    """Aggregated route metrics for GET /routes/.../stats and the assistant.

    shipment_count includes rows whose delay is null.
    average_delay_hours and on_time_rate use only the known delays.
    Both are null when every delay is null, so a route of missing arrivals
    does not look like a zero-delay route.
    """
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
    """Turn a port code or a port name into one port code.

    The sample question "Shanghai to Rotterdam" does not use codes.
    An exact code wins. A name matches case-insensitively, and only when one
    port has that name. Anything else is unknown, and the caller returns 404.
    """
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
    """Rank routes by average delay for "highest average delay this quarter".

    Groups rows inside [start, end) by origin and destination, drops routes
    with no known delay, and returns the highest averages first, capped by limit.
    The quarter window is chosen by latest_quarter, not by this function.
    """
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
    """Quarter of the latest booking_date in the data, not the wall clock.

    "This quarter" in the sample question is the latest quarter present in the
    file (2025 Q4, bookings through 14 Oct 2025). now is accepted and ignored
    so a caller cannot accidentally rank against today's calendar.
    The window is [start, end): 1 Jan, 1 Apr, 1 Jul, or 1 Oct, through the
    first day of the next quarter.
    """
    del now
    latest = max(bookings)
    quarter = (latest.month - 1) // 3 + 1 # why not just divide by 3 +1 ? e.g. 
    start_month = (quarter - 1) * 3 + 1
    start = datetime(latest.year, start_month, 1)
    if quarter == 4:
        end = datetime(latest.year + 1, 1, 1)
    else:
        end = datetime(latest.year, QUARTER_END_MONTH[quarter], 1)
    return {"year": latest.year, "quarter": quarter, "start": start, "end": end}
