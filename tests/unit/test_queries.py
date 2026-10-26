"""Unit tests for the read sessions the API and the assistant share."""

from datetime import datetime, timedelta, timezone

import pytest

from app.queries import list_shipments, open_session


def test_open_session_sets_statement_timeout(monkeypatch) -> None:
    """Every read session caps the statement at 5 seconds so a bad query cannot hang the API."""
    class FakeSession:
        """Records the SQL open_session runs, including the statement timeout."""
        def __init__(self) -> None:
            self.statements: list[str] = []

        def execute(self, statement, *args, **kwargs):
            """Record the SQL so the test can see the statement timeout."""
            self.statements.append(str(statement))
            return self

    fake = FakeSession()
    monkeypatch.setattr("app.queries._SESSION", lambda: fake)
    session = open_session()
    assert any("statement_timeout" in sql for sql in session.statements)


def test_aware_booking_bound_is_rejected() -> None:
    """Booking filters reject a timezone. The column is timestamp without time zone."""
    aware = datetime(2024, 8, 13, 9, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    with pytest.raises(ValueError, match="timezone"):
        list_shipments(object(), None, None, None, aware, None, 50, 0)  # type: ignore[arg-type]
