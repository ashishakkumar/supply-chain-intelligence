"""Unit tests for health, structured errors, and a model that fails closed."""

import logging

from fastapi.testclient import TestClient

from app.main import create_app, database_check
from app.ml.predict import reset_artifact


def test_database_check_reports_down_without_failing_the_process() -> None:
    """A database failure is the word "down", not an exception that kills the process."""

    def fail() -> None:
        """Stand-in for a database connection that cannot be opened."""
        raise RuntimeError("database down")

    assert database_check(fail) == "down"
    assert database_check(lambda: None) == "ok"


def test_unhandled_error_is_json_and_logged(monkeypatch, capsys) -> None:
    """An unexpected error is 500 JSON, the traceback stays out of the body, and the access log still records 500."""

    def explode(*args, **kwargs):
        """Stand-in for a route that raises before it can build a response."""
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
    """A joblib file saved with another sklearn version is not a model. Predict returns 503."""
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


def test_docs_show_an_example_for_every_input() -> None:
    """Swagger can try a call without inventing an id, a port, or a body.

    The sample values are rows that exist after a normal seed, so Try it out
    on /docs returns 200.
    """
    paths = create_app().openapi()["paths"]

    shipment = _param(paths["/shipments/{shipment_id}"]["get"], "shipment_id")
    assert _example(shipment) == "SHP-00623"

    listing = paths["/shipments"]["get"]
    assert _example(_param(listing, "origin")) == "NLRTM"
    assert _example(_param(listing, "destination")) == "PKKAR"
    assert _example(_param(listing, "status")) == "DELIVERED"
    assert _example(_param(listing, "from")) == "2024-08-13 09:00:00"
    assert _example(_param(listing, "to")) == "2024-08-13 09:00:00"

    route = paths["/routes/{origin}/{destination}/stats"]["get"]
    assert _example(_param(route, "origin")) == "Shanghai"
    assert _example(_param(route, "destination")) == "Rotterdam"

    body = paths["/predict-delay"]["post"]["requestBody"]["content"]["application/json"]
    assert _example(body) == {"shipment_id": "SHP-00623"}


def _param(operation: dict, name: str) -> dict:
    """The OpenAPI parameter object with this name."""
    return next(item for item in operation["parameters"] if item["name"] == name)


def _example(node: dict) -> object:
    """The first declared example value on a parameter or a request body."""
    return next(iter(node["examples"].values()))["value"]


def test_lifespan_loads_a_missing_model_without_crashing(monkeypatch, tmp_path) -> None:
    """Startup loads the artefact once. A missing file does not stop the process."""
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    reset_artifact()
    from app.ml import predict

    with TestClient(create_app()):
        assert predict._LOADED is True
        assert predict._ARTIFACT is None
