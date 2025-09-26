"""Pydantic schemas for the TatvaChakra FastAPI service."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class FeatureVector(BaseModel):
    """Represents the controllable features for ML predictions."""

    electricity_kwh_per_t: Optional[float] = Field(None, ge=0)
    fuel_l_per_t: Optional[float] = Field(None, ge=0)
    transport_tkm_per_t: Optional[float] = Field(None, ge=0)
    renewable_share_pct: Optional[float] = Field(None, ge=0, le=100)
    recycled_content_pct: Optional[float] = Field(None, ge=0, le=100)
    recovery_rate_pct: Optional[float] = Field(None, ge=0, le=100)
    upstream_input_count: Optional[float] = Field(None, ge=0)

    def to_dense(self, defaults: Dict[str, float]) -> Dict[str, float]:
        data = defaults.copy()
        for key, value in self.dict(exclude_none=True).items():
            data[key] = float(value)
        return data


class ImputeRequest(BaseModel):
    process_name: Optional[str] = None
    category: Optional[str] = None
    location: Optional[str] = None
    scrap_ratio_pct: Optional[float] = Field(None, ge=0, le=100)
    throughput_t: Optional[float] = Field(None, ge=0)
    transport_tkm_hint: Optional[float] = Field(None, ge=0)
    features: FeatureVector = Field(default_factory=FeatureVector)


class ImputeResponse(BaseModel):
    predicted_energy_kwh_per_t: float
    lower_bound: float
    upper_bound: float
    mae: float
    feature_importances: Dict[str, float]


class WhatIfRequest(BaseModel):
    current_config: FeatureVector


class ActionSuggestion(BaseModel):
    name: str
    description: str
    delta_config: Dict[str, float]
    projected_delta_gwp: float
    projected_delta_circularity: float


class WhatIfResponse(BaseModel):
    base_gwp: float
    base_circularity: float
    updated_config: Dict[str, float]
    delta_gwp: float
    delta_circularity: float
    ranked_actions: List[ActionSuggestion]


class LciaGoalRequest(BaseModel):
    goal_description: str
    sector: Optional[str] = None
    focus_areas: List[str] = Field(default_factory=list)


class LciaMethodRecommendation(BaseModel):
    name: str
    rationale: str


class LciaGoalResponse(BaseModel):
    recommendations: List[LciaMethodRecommendation]


class ValidationMessage(BaseModel):
    field: str
    message: str
    severity: str = Field("warning", regex="^(info|warning|error)$")


class ValidationResponse(BaseModel):
    messages: List[ValidationMessage]


def clamp_feature_bounds(features: Dict[str, float]) -> Dict[str, float]:
    """Apply feasibility limits to controllable features."""

    bounds = {
        "recycled_content_pct": (0.0, 95.0),
        "renewable_share_pct": (0.0, 90.0),
        "electricity_kwh_per_t": (0.0, None),
        "transport_tkm_per_t": (0.0, None),
        "fuel_l_per_t": (0.0, None),
        "recovery_rate_pct": (0.0, 100.0),
    }
    clamped = features.copy()
    for key, (lower, upper) in bounds.items():
        if key not in clamped:
            continue
        value = clamped[key]
        if lower is not None:
            value = max(lower, value)
        if upper is not None:
            value = min(upper, value)
        clamped[key] = value
    return clamped


def default_feature_vector() -> Dict[str, float]:
    return {
        "electricity_kwh_per_t": 0.0,
        "fuel_l_per_t": 0.0,
        "transport_tkm_per_t": 0.0,
        "renewable_share_pct": 0.0,
        "recycled_content_pct": 0.0,
        "recovery_rate_pct": 0.0,
        "upstream_input_count": 0.0,
    }

