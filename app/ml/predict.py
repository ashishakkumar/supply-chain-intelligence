"""Load the saved delay model and apply the 0.5 decision threshold.

The file is read once per process. A missing file, a bad file, or a file saved
with a different scikit-learn than the pinned 1.9.1 is treated as no model.
POST /predict-delay then returns 503. The process still starts.
"""

import os
from pathlib import Path

import joblib
from sklearn import __version__ as sklearn_version

_ARTIFACT: dict[str, object] | None = None
_LOADED = False


def predicted_late(score: float) -> bool:
    """True when late_score is at least 0.5.

    The score is a ranking score, not a calibrated probability. class_weight
    balanced changes the loss, not this threshold. In production, watch the
    share of scores above 0.5 against the training base rate.
    """
    return score >= 0.5


def _read_artifact() -> dict[str, object] | None:
    """Read MODEL_PATH (default models/delay_model.joblib), or return None.

    None means the endpoint must not invent a score. The dict holds the sklearn
    pipeline, model_version, and the sklearn version it was saved with.
    """
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
    """Return the model loaded at startup. Later calls reuse that result.

    Called from the FastAPI lifespan and again on each predict request. The
    second call does not touch the disk.
    """
    global _ARTIFACT, _LOADED
    if _LOADED:
        return _ARTIFACT
    _ARTIFACT = _read_artifact()
    _LOADED = True
    return _ARTIFACT


def reset_artifact() -> None:
    """Forget the cached artefact so the next get_artifact reads the file again.

    Tests use this. The running API does not.
    """
    global _ARTIFACT, _LOADED
    _ARTIFACT = None
    _LOADED = False
