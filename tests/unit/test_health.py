import logging

from fastapi.testclient import TestClient

from app.main import create_app, database_check
from app.ml.predict import reset_artifact


def test_database_check_reports_down_without_failing_the_process() -> None:
    def fail() -> None:
        raise RuntimeError("database down")

    assert database_check(fail) == "down"
    assert database_check(lambda: None) == "ok"


def test_unhandled_error_is_json_and_logged(monkeypatch, capsys) -> None:
    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("app.api.list_shipments", explode)
    logging.getLogger("app.access").handlers.clear()
    with TestClient(create_app(), raise_server_exceptions=False) as client:
        response = client.get("/shipments")
    assert response.status_code == 500
    assert response.json()["error"] == {
        "code": "internal_error",
        "message": "Internal server error",
    }
    assert "boom" not in response.text
    assert '"status_code": 500' in capsys.readouterr().out


def test_corrupt_model_file_is_unavailable(monkeypatch, tmp_path) -> None:
    import joblib

    path = tmp_path / "bad.joblib"
    joblib.dump({"pipeline": object(), "sklearn_version": "0.0.0"}, path)
    monkeypatch.setenv("MODEL_PATH", str(path))
    reset_artifact()
    from app.ml.predict import get_artifact

    assert get_artifact() is None
    with TestClient(create_app()) as client:
        response = client.post("/predict-delay", json={"shipment_id": "SHP-1"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_unavailable"


def test_lifespan_loads_a_missing_model_without_crashing(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    reset_artifact()
    from app.ml import predict

    with TestClient(create_app()):
        assert predict._LOADED is True
        assert predict._ARTIFACT is None
