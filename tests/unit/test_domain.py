from datetime import datetime

from app.domain import (
    delay_hours,
    latest_quarter,
    on_time_flag,
    resolve_port,
    route_key,
    route_stats,
    transit_days,
)


def test_delay_hours_keeps_partial_hours() -> None:
    planned = datetime(2024, 1, 1, 0, 0)
    actual = datetime(2024, 1, 1, 1, 30)
    assert delay_hours(actual, planned) == 1.5


def test_delay_hours_is_null_when_either_side_is_missing() -> None:
    moment = datetime(2024, 1, 1, 0, 0)
    assert delay_hours(None, moment) is None
    assert delay_hours(moment, None) is None
    assert delay_hours(None, None) is None


def test_on_time_boundaries() -> None:
    assert on_time_flag(24) is True
    assert on_time_flag(25) is False
    assert on_time_flag(-3) is True
    assert on_time_flag(None) is None


def test_missing_arrival_has_null_transit_days() -> None:
    departure = datetime(2024, 1, 1, 0, 0)
    assert transit_days(None, departure) is None


def test_route_key_uses_the_arrow_label() -> None:
    assert route_key("CNSHA", "NLRTM") == "CNSHA → NLRTM"


def test_route_stats_excludes_unknown_arrivals_from_the_rate() -> None:
    stats = route_stats([-2.0, 24.0, 25.0, None])
    assert stats["shipment_count"] == 4
    assert stats["on_time_rate"] == 2 / 3
    assert stats["average_delay_hours"] == (-2.0 + 24.0 + 25.0) / 3


def test_route_stats_does_not_divide_by_zero_when_arrival_is_unknown() -> None:
    stats = route_stats([None])
    assert stats["shipment_count"] == 1
    assert stats["average_delay_hours"] is None
    assert stats["on_time_rate"] is None


def test_route_stats_for_a_route_with_no_shipments() -> None:
    stats = route_stats([])
    assert stats["shipment_count"] == 0
    assert stats["average_delay_hours"] is None
    assert stats["on_time_rate"] is None


def test_resolve_port_accepts_a_code_or_a_unique_name() -> None:
    ports = [{"port_code": "CNSHA", "port_name": "Shanghai"}]
    assert resolve_port("shanghai", ports) == "CNSHA"
    assert resolve_port("CNSHA", ports) == "CNSHA"
    assert resolve_port("Atlantis", ports) is None


def test_latest_quarter_uses_the_data_not_the_clock() -> None:
    window = latest_quarter(
        [datetime(2025, 10, 14, 19, 0)],
        now=datetime(2026, 10, 18),
    )
    assert window == {
        "year": 2025,
        "quarter": 4,
        "start": datetime(2025, 10, 1),
        "end": datetime(2026, 1, 1),
    }
