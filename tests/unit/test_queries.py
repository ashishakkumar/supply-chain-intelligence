from datetime import datetime, timedelta, timezone

import pytest

from app.queries import list_shipments, open_session


def test_open_session_sets_statement_timeout(monkeypatch) -> None:
    class FakeSession:
        def __init__(self) -> None:
            self.statements: list[str] = []

        def execute(self, statement, *args, **kwargs):
            self.statements.append(str(statement))
            return self

    fake = FakeSession()
    monkeypatch.setattr("app.queries._SESSION", lambda: fake)
    session = open_session()
    assert any("statement_timeout" in sql for sql in session.statements)


def test_aware_booking_bound_is_rejected() -> None:
    aware = datetime(2024, 8, 13, 9, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    with pytest.raises(ValueError, match="timezone"):
        list_shipments(object(), None, None, None, aware, None, 50, 0)  # type: ignore[arg-type]
