"""Unit tests for SQL migrations. Tables come from the SQL files, not from the ORM classes."""

from pathlib import Path

from sqlalchemy import create_engine, text

from app.migrate import apply, main


def test_sql_file_runs_once(tmp_path: Path) -> None:
    """A migration file is recorded in applied_changes and skipped the second time."""
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "001_create_demo.sql").write_text("CREATE TABLE demo (id INTEGER);\n")
    url = f"sqlite:///{tmp_path / 'app.db'}"

    assert apply(url, migrations) == ["001_create_demo.sql"]
    assert apply(url, migrations) == []

    with create_engine(url).connect() as conn:
        conn.execute(text("SELECT id FROM demo"))
        recorded = conn.execute(text("SELECT filename FROM applied_changes")).scalars().all()
    assert recorded == ["001_create_demo.sql"]


def test_sql_file_runs_every_statement(tmp_path: Path) -> None:
    """Every statement in the file runs, not only the first one before a semicolon."""
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "001_demo.sql").write_text(
        "CREATE TABLE demo (id INTEGER);\nINSERT INTO demo (id) VALUES (1);\n"
    )
    url = f"sqlite:///{tmp_path / 'app.db'}"

    assert apply(url, migrations) == ["001_demo.sql"]

    with create_engine(url).connect() as conn:
        assert conn.execute(text("SELECT id FROM demo")).scalar() == 1


def test_main_does_not_create_tables_from_models(tmp_path: Path, monkeypatch) -> None:
    """db-migrate applies the SQL directory. It does not create shipments or ports from the Python classes."""
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "001_create_demo.sql").write_text("CREATE TABLE demo (id INTEGER);\n")
    url = f"sqlite:///{tmp_path / 'app.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setattr("app.migrate.MIGRATIONS", migrations)

    assert main() == 0

    with create_engine(url).connect() as conn:
        conn.execute(text("SELECT id FROM demo"))
        names = set(
            conn.execute(
                text("SELECT name FROM sqlite_master WHERE type = 'table'")
            ).scalars()
        )
    assert "shipments" not in names
    assert "ports" not in names
