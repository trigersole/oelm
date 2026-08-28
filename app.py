"""FastAPI inference service for the two-level OELM XGBoost models.

The four affective targets share one binary scale:

* 0 = Low  (DAiSEE levels Very Low + Low)
* 1 = High (DAiSEE levels High + Very High)
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Literal

import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


LABEL_COLS = ["Boredom", "Engagement", "Confusion", "Frustration"]
LEVEL_NAMES = ["Low", "High"]
MODEL_VERSION = "xgb-binary-v1"

ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "binary_models"
FEATURE_ORDER_PATH = MODEL_DIR / "feature_order.json"
MODEL_CONFIG_PATH = MODEL_DIR / "model_config.json"


def load_json(path: Path) -> dict | list:
    if not path.exists():
        raise FileNotFoundError(f"Required model resource not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


FEATURE_ORDER = list(load_json(FEATURE_ORDER_PATH))
MODEL_CONFIG = dict(load_json(MODEL_CONFIG_PATH))
DECISION_THRESHOLDS = {
    label: float(MODEL_CONFIG["decision_thresholds"][label]) for label in LABEL_COLS
}
BEST_ITERATIONS = {
    label: int(MODEL_CONFIG["best_iterations"][label]) for label in LABEL_COLS
}

MODEL_FILES = {
    label: MODEL_DIR / f"model_binary_{label}.ubj" for label in LABEL_COLS
}
models: dict[str, xgb.Booster] = {}


def load_models() -> None:
    """Load each binary XGBoost head exactly once."""
    if models:
        return
    for label, path in MODEL_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Binary model file not found: {path}")
        booster = xgb.Booster()
        booster.load_model(path)
        models[label] = booster
    print(f"[OELM] Loaded binary models: {list(models)}")


app = FastAPI(title="OELM Binary XGBoost API", version="2.0.0")

allowed_origins = [
    origin.strip()
    for origin in os.getenv("OELM_ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    load_models()


class PredictRequest(BaseModel):
    # The browser sends seven aggregate statistics for each of 52 blendshapes.
    agg_features: dict[str, float]


class LabelPrediction(BaseModel):
    label: Literal[0, 1]
    level: Literal["Low", "High"]
    probabilities: dict[int, float]
    threshold: float


class PredictResponse(BaseModel):
    Boredom: LabelPrediction
    Engagement: LabelPrediction
    Confusion: LabelPrediction
    Frustration: LabelPrediction


@app.get("/")
def root() -> dict:
    return {
        "service": "OELM Binary XGBoost API",
        "model_version": MODEL_VERSION,
        "levels": LEVEL_NAMES,
        "health": "/health",
        "model_info": "/model-info",
        "api_docs": "/docs",
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "levels": LEVEL_NAMES,
        "feature_count": len(FEATURE_ORDER),
        "models_loaded": list(models.keys()),
    }


@app.get("/model-info")
def model_info() -> dict:
    return {
        "model_version": MODEL_VERSION,
        "model_type": "direct binary XGBoost",
        "labels": LABEL_COLS,
        "levels": {0: "Low", 1: "High"},
        "source_level_mapping": {
            "Low": ["Very Low", "Low"],
            "High": ["High", "Very High"],
        },
        "feature_count": len(FEATURE_ORDER),
        "decision_thresholds": DECISION_THRESHOLDS,
        "best_iterations": BEST_ITERATIONS,
        "training_seed": MODEL_CONFIG["training_seed"],
    }


def ordered_feature_row(agg_features: dict[str, float]) -> np.ndarray:
    missing = [key for key in FEATURE_ORDER if key not in agg_features]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing {len(missing)} features. First 5: {missing[:5]}",
        )

    non_finite = [
        key for key in FEATURE_ORDER if not math.isfinite(float(agg_features[key]))
    ]
    if non_finite:
        raise HTTPException(
            status_code=400,
            detail=f"Non-finite feature values. First 5: {non_finite[:5]}",
        )

    return np.asarray(
        [[agg_features[key] for key in FEATURE_ORDER]], dtype=np.float32
    )


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest) -> dict:
    if not models:
        load_models()

    data_matrix = xgb.DMatrix(
        ordered_feature_row(body.agg_features), feature_names=FEATURE_ORDER
    )
    result: dict[str, dict] = {}

    for label in LABEL_COLS:
        output = np.asarray(
            models[label].predict(
                data_matrix,
                iteration_range=(0, BEST_ITERATIONS[label] + 1),
            ),
            dtype=np.float64,
        ).reshape(-1)
        if output.size != 1:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected prediction shape for {label}: {output.shape}",
            )

        high_probability = float(np.clip(output[0], 0.0, 1.0))
        low_probability = 1.0 - high_probability
        threshold = DECISION_THRESHOLDS[label]
        predicted_label = int(high_probability >= threshold)

        result[label] = {
            "label": predicted_label,
            "level": LEVEL_NAMES[predicted_label],
            "probabilities": {
                0: round(low_probability, 5),
                1: round(high_probability, 5),
            },
            "threshold": round(threshold, 5),
        }

    return result
