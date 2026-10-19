import os
from pathlib import Path

import joblib
from sklearn import __version__ as sklearn_version

_ARTIFACT: dict[str, object] | None = None
_LOADED = False


def predicted_late(score: float) -> bool:
    return score >= 0.5


def _read_artifact() -> dict[str, object] | None:
    path = Path(os.environ.get("MODEL_PATH", "models/delay_model.joblib"))
    if not path.exists():
        return None
    try:
        loaded = joblib.load(path)
    except Exception:  # noqa: BLE001 — a bad file is an unavailable model, not a crash
        return None
    if not isinstance(loaded, dict) or "pipeline" not in loaded:
        return None
    if loaded.get("sklearn_version") != sklearn_version:
        return None
    return loaded


def get_artifact() -> dict[str, object] | None:
    global _ARTIFACT, _LOADED
    if _LOADED:
        return _ARTIFACT
    _ARTIFACT = _read_artifact()
    _LOADED = True
    return _ARTIFACT


def reset_artifact() -> None:
    global _ARTIFACT, _LOADED
    _ARTIFACT = None
    _LOADED = False
