# src/predict_router.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import joblib
import tensorflow as tf          # tylko do wczytania nn.h5
from typing import Dict, Tuple

# ───── pliki i katalogi ───────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]    # …/venv/src -> back to venv
MODELS_DIR  = ROOT / "models"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

MODEL_PATHS: Dict[str, Path] = {
    "logistic": MODELS_DIR / "logistic.pkl",
    "tree"    : MODELS_DIR / "tree.pkl",
    "forest"  : MODELS_DIR / "forest.pkl",
    "svm"     : MODELS_DIR / "svm.pkl",
    "knn"     : MODELS_DIR / "knn.pkl",
    "nn"      : MODELS_DIR / "nn.h5",
}

# ───── ŁADOWANIE W PAMIĘĆ (singleton) ─────────────────────
def _load_scaler():
    return joblib.load(SCALER_PATH)

def _load_models():
    models = {}
    for name, path in MODEL_PATHS.items():
        if name == "nn":
            models[name] = tf.keras.models.load_model(path)
        else:
            models[name] = joblib.load(path)
    return models

SCALER = _load_scaler()
MODELS = _load_models()
FEATURE_DIM = SCALER.mean_.shape[0]        # 30

# ───── PREPROCESS → PREDICT → POSTPROCESS ─────────────────
def _preprocess(features: list[float]) -> np.ndarray:
    """Zamienia listę 30 liczb na zeskalowany wektor 2-D (1,30)."""
    arr = np.asarray(features, dtype=float).reshape(1, -1)
    return SCALER.transform(arr)

def _postprocess(pred: np.ndarray | int,
                 proba: float | None) -> dict:
    return {
        "label": int(pred),
        "probability": None if proba is None else round(float(proba), 4)
    }

# ───── GŁÓWNA FUNKCJA ROUTERA ─────────────────────────────
def predict(model_name: str,
            features: list[float]) -> dict:
    """
    Parameters
    ----------
    model_name : str
        Jedna z nazw kluczy w MODEL_PATHS.
    features   : list[float]
        Dokładnie 30 wartości cech (w kolejności takiej jak w treningu).

    Returns
    -------
    dict : {"label": 0/1, "probability": float|None, "model": str}
    """
    if model_name not in MODELS:
        raise ValueError(f"Model '{model_name}' not found.")
    if len(features) != FEATURE_DIM:
        raise ValueError(f"Expected {FEATURE_DIM} features, got {len(features)}.")

    X = _preprocess(features)
    model = MODELS[model_name]

    if model_name == "nn":
        proba = float(model.predict(X)[0, 0])
        label = int(proba >= 0.5)
    else:
        proba = float(model.predict_proba(X)[0, 1]) \
                if hasattr(model, "predict_proba") else None
        label = int(model.predict(X)[0])

    return {
        "model": model_name,
        **_postprocess(label, proba)
    }

# ───── PROSTA DEMONSTRACJA CLI ────────────────────────────
if __name__ == "__main__":
    # losowy (nieskalowany) wektor tylko do szybkiego testu
    sample = [12.5]*30
    for name in MODEL_PATHS:
        print(name, predict(name, sample))
