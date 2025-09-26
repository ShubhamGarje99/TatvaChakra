"""FastAPI application exposing TatvaChakra's AI helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
from fastapi import FastAPI, HTTPException

from .model_store import ModelStore
from .models import (
    ActionSuggestion,
    FeatureVector,
    ImputeRequest,
    ImputeResponse,
    LciaGoalRequest,
    LciaGoalResponse,
    LciaMethodRecommendation,
    ValidationMessage,
    ValidationResponse,
    WhatIfRequest,
    WhatIfResponse,
    clamp_feature_bounds,
    default_feature_vector,
)

app = FastAPI(title="TatvaChakra AI Service", version="0.1.0")

_model_store: ModelStore | None = None


@app.on_event("startup")
def load_models() -> None:
    model_dir = Path("models")
    if not model_dir.exists():
        raise RuntimeError(
            "Model artifacts were not found. Run `python -m ml.train_models` first."
        )
    global _model_store
    _model_store = ModelStore(model_dir)
    _model_store.load()


@app.post("/api/impute", response_model=ImputeResponse)
def impute_energy(request: ImputeRequest) -> ImputeResponse:
    if _model_store is None:
        raise HTTPException(status_code=500, detail="Model store not initialised")
    defaults = default_feature_vector()
    feature_vector = request.features.to_dense(defaults)
    # When the user provides additional hints, integrate them.
    if request.transport_tkm_hint is not None:
        feature_vector["transport_tkm_per_t"] = float(request.transport_tkm_hint)

    feature_order = _model_store.metadata.get("feature_columns", list(defaults.keys()))
    features = np.array([[feature_vector[col] for col in feature_order]])
    prediction = float(_model_store.imputer.predict(features)[0])
    mae = float(_model_store.metadata.get("imputer_mae", 0.0))
    interval = 1.5 * mae
    lower = max(0.0, prediction - interval)
    upper = prediction + interval
    importances = _model_store.metadata.get("imputer_importances", {})

    return ImputeResponse(
        predicted_energy_kwh_per_t=prediction,
        lower_bound=lower,
        upper_bound=upper,
        mae=mae,
        feature_importances=importances,
    )


def _apply_action(base_config: Dict[str, float], delta: Dict[str, float]) -> Dict[str, float]:
    updated = base_config.copy()
    for key, value in delta.items():
        updated[key] = updated.get(key, 0.0) + value
    return clamp_feature_bounds(updated)


@app.post("/api/whatif", response_model=WhatIfResponse)
def run_what_if(request: WhatIfRequest) -> WhatIfResponse:
    if _model_store is None:
        raise HTTPException(status_code=500, detail="Model store not initialised")

    defaults = default_feature_vector()
    base_config = clamp_feature_bounds(request.current_config.to_dense(defaults))
    feature_order = _model_store.metadata.get("feature_columns", list(defaults.keys()))
    base_vector = np.array([[base_config[key] for key in feature_order]])
    base_gwp = float(_model_store.surrogate_gwp.predict(base_vector)[0])
    base_circ = float(_model_store.surrogate_circularity.predict(base_vector)[0])

    candidate_actions = [
        ActionSuggestion(
            name="Increase recycled content",
            description="Boost recycled content share by 10 percentage points",
            delta_config={"recycled_content_pct": 10.0},
            projected_delta_gwp=0.0,
            projected_delta_circularity=0.0,
        ),
        ActionSuggestion(
            name="Raise renewable electricity share",
            description="Source more renewable power (+20pp)",
            delta_config={"renewable_share_pct": 20.0},
            projected_delta_gwp=0.0,
            projected_delta_circularity=0.0,
        ),
        ActionSuggestion(
            name="Optimise transport logistics",
            description="Reduce transport intensity by 20%",
            delta_config={"transport_tkm_per_t": -0.2 * base_config["transport_tkm_per_t"]},
            projected_delta_gwp=0.0,
            projected_delta_circularity=0.0,
        ),
        ActionSuggestion(
            name="Improve energy efficiency",
            description="Cut electricity intensity by 15%",
            delta_config={"electricity_kwh_per_t": -0.15 * base_config["electricity_kwh_per_t"]},
            projected_delta_gwp=0.0,
            projected_delta_circularity=0.0,
        ),
    ]

    evaluated_actions: List[ActionSuggestion] = []
    best_config = base_config
    best_gwp = base_gwp
    best_circ = base_circ

    for action in candidate_actions:
        updated = _apply_action(base_config, action.delta_config)
        vector = np.array([[updated[key] for key in feature_order]])
        gwp = float(_model_store.surrogate_gwp.predict(vector)[0])
        circ = float(_model_store.surrogate_circularity.predict(vector)[0])
        delta = {key: updated[key] - base_config.get(key, 0.0) for key in feature_order if abs(updated[key] - base_config.get(key, 0.0)) > 1e-6}
        evaluated_actions.append(
            ActionSuggestion(
                name=action.name,
                description=action.description,
                delta_config=delta,
                projected_delta_gwp=gwp - base_gwp,
                projected_delta_circularity=circ - base_circ,
            )
        )
        if gwp < best_gwp:
            best_config = updated
            best_gwp = gwp
            best_circ = circ

    evaluated_actions.sort(key=lambda a: a.projected_delta_gwp)
    delta_gwp = best_gwp - base_gwp
    delta_circ = best_circ - base_circ

    return WhatIfResponse(
        base_gwp=base_gwp,
        base_circularity=base_circ,
        updated_config=best_config,
        delta_gwp=delta_gwp,
        delta_circularity=delta_circ,
        ranked_actions=evaluated_actions[:3],
    )


@app.post("/api/recommend_lcia", response_model=LciaGoalResponse)
def recommend_lcia_methods(request: LciaGoalRequest) -> LciaGoalResponse:
    keywords = (request.goal_description or "").lower()
    focus = [item.lower() for item in request.focus_areas]

    recommendations: List[Dict[str, str]] = []

    def add(name: str, rationale: str) -> None:
        recommendations.append({"name": name, "rationale": rationale})

    if any(word in keywords for word in ["climate", "carbon", "gwp"]):
        add("IPCC GWP 100a", "Goal references climate impact; prioritise CO2e")
    if any(word in keywords for word in ["tox", "human health"]):
        add("USEtox", "Toxicity focus detected in goal statement")
    if any(word in keywords for word in ["water", "scarcity"]):
        add("AWARE", "Water availability metrics requested")
    if any(word in keywords for word in ["circular", "recycl"]):
        add("Material Circularity Indicators", "Circularity focus; emphasise recovery flows")

    if not recommendations:
        add("IPCC GWP 100a", "Default baseline for carbon accounting")

    return LciaGoalResponse(
        recommendations=[
            LciaMethodRecommendation(**item) for item in recommendations
        ]
    )


@app.post("/api/validate_inputs", response_model=ValidationResponse)
def validate_inputs(request: FeatureVector) -> ValidationResponse:
    if _model_store is None:
        raise HTTPException(status_code=500, detail="Model store not initialised")

    defaults = default_feature_vector()
    features = request.to_dense(defaults)
    bounds = clamp_feature_bounds(features)

    messages: List[ValidationMessage] = []
    for key, value in features.items():
        if bounds[key] != value:
            messages.append(
                ValidationMessage(
                    field=key,
                    message=f"Value {value:.2f} adjusted to feasible range {bounds[key]:.2f}",
                    severity="warning",
                )
            )

    # Simple z-score anomaly detection using training metadata if available.
    stats = _model_store.metadata.get("feature_stats", {})
    for key, value in features.items():
        stat = stats.get(key)
        if not stat:
            continue
        mean = stat.get("mean")
        std = stat.get("std") or 1.0
        if std <= 0:
            continue
        z = abs(value - mean) / std
        if z > 3:
            messages.append(
                ValidationMessage(
                    field=key,
                    message=f"Value {value:.2f} is {z:.1f}σ away from the training mean",
                    severity="warning",
                )
            )

    return ValidationResponse(messages=messages)

