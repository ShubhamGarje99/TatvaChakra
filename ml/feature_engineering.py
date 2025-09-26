"""Utilities for transforming OpenLCA exports into ML-ready tables.

This module implements data extraction helpers that parse process JSON
files from an OpenLCA 2.x export. The code is deliberately defensive so it
can tolerate partially-missing metadata during a hackathon workflow.

The exported structure we expect is the default "directory" export where
processes are stored under ``processes/*.json``. Each process JSON object
contains metadata (name, category, location) and a list of exchanges. For
training we need a handful of engineered features per process, including
energy/carrier intensities and circularity-relevant ratios.

Two public entry-points are provided:

``build_imputer_dataset``
    Returns a :class:`pandas.DataFrame` with engineered features and the
    regression target required for the missing-value imputer. The target is
    ``energy_kwh_per_t`` by default but can be configured.

``build_surrogate_dataset``
    Returns a feature matrix and a dictionary of target arrays to train the
    surrogate models used by the what-if recommender. The function can blend
    real process data with synthetic sweeps over controllable parameters so
    the surrogate learns useful gradients even with a small dataset.

The functions intentionally avoid persisting anything to disk; the training
script can decide how/where to store the resulting matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import json
import math

import numpy as np
import pandas as pd


@dataclass
class ProcessRecord:
    """Convenience container for the engineered features of one process."""

    process_id: str
    name: str
    category: str
    location: str
    reference_amount_t: float
    features: Dict[str, float]


ENERGY_KEYWORDS = {
    "electricity",
    "power",
    "energy",
}

TRANSPORT_KEYWORDS = {
    "transport",
    "freight",
    "tkm",
}

FUEL_KEYWORDS = {
    "diesel",
    "gasoline",
    "fuel",
    "lpg",
    "natural gas",
}


UNIT_TO_TON = {
    "kg": 1 / 1000.0,
    "kilogram": 1 / 1000.0,
    "ton": 1.0,
    "t": 1.0,
    "lb": 0.000453592,
}

UNIT_TO_KWH = {
    "kwh": 1.0,
    "kilowatt hour": 1.0,
    "mwh": 1000.0,
    "gwh": 1_000_000.0,
    "mj": 1 / 3.6,
    "gj": 1_000 / 3.6,
}

UNIT_TO_LITRE = {
    "l": 1.0,
    "liter": 1.0,
    "litre": 1.0,
    "ml": 0.001,
    "m3": 1_000.0,
}

UNIT_TO_TKM = {
    "tkm": 1.0,
    "t*km": 1.0,
    "ton kilometer": 1.0,
    "tonne kilometre": 1.0,
}


def _iter_process_json(dataset_dir: Path) -> Iterable[Tuple[Path, Dict]]:
    """Yield process JSON payloads from an OpenLCA export directory."""

    for path in sorted(dataset_dir.rglob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError:
            continue
        if payload.get("@type", "").lower() != "process":
            continue
        yield path, payload


def _normalise_unit(value: float, unit: Optional[Dict], mapping: Dict[str, float]) -> Optional[float]:
    """Convert ``value`` to the canonical unit using ``mapping``.

    ``unit`` is the OpenLCA unit reference dictionary which may contain a
    ``name`` field. If the unit is unknown we simply return ``None`` so the
    caller can ignore the exchange for that aggregate.
    """

    if unit is None:
        return None
    unit_name = unit.get("name", "").lower()
    if unit_name in mapping:
        return value * mapping[unit_name]
    return None


def _categorise_exchange(exchange: Dict) -> Tuple[str, Optional[float]]:
    """Classify an exchange as electricity, fuel, transport or other.

    Returns a tuple ``(category, amount)`` where ``category`` is one of
    ``{"electricity", "fuel", "transport", "other"}`` and ``amount`` is
    the amount converted to the category's canonical unit (kWh, L, tkm). If
    the exchange cannot be converted ``amount`` is ``None``.
    """

    flow = exchange.get("flow", {})
    name = flow.get("name", "").lower()
    amount = exchange.get("amount", 0.0)
    unit = exchange.get("unit")

    # Attempt to detect by unit first (more reliable than keywords).
    kwh = _normalise_unit(amount, unit, UNIT_TO_KWH)
    if kwh is not None:
        return "electricity", kwh

    litre = _normalise_unit(amount, unit, UNIT_TO_LITRE)
    if litre is not None:
        return "fuel", litre

    tkm = _normalise_unit(amount, unit, UNIT_TO_TKM)
    if tkm is not None:
        return "transport", tkm

    # Fall back to keyword heuristics.
    if any(keyword in name for keyword in ENERGY_KEYWORDS):
        return "electricity", amount
    if any(keyword in name for keyword in TRANSPORT_KEYWORDS):
        return "transport", amount
    if any(keyword in name for keyword in FUEL_KEYWORDS):
        return "fuel", amount

    return "other", None


def _reference_amount_t(process: Dict) -> Optional[float]:
    """Return the reference product amount converted to tonnes."""

    for exchange in process.get("exchanges", []):
        if not exchange.get("isInput", False) and exchange.get("isQuantitativeReference"):
            amount = exchange.get("amount", 0.0)
            converted = _normalise_unit(amount, exchange.get("unit"), UNIT_TO_TON)
            if converted is not None and converted > 0:
                return converted
    return None


def extract_process_features(process: Dict) -> Optional[ProcessRecord]:
    """Transform a single process JSON payload into engineered features."""

    ref_amount = _reference_amount_t(process)
    if not ref_amount:
        return None

    location = (process.get("location") or {}).get("code") or "GLO"
    category = (process.get("category") or {}).get("name") or "unspecified"

    electricity_kwh = 0.0
    fuel_l = 0.0
    transport_tkm = 0.0
    upstream_inputs = 0
    renewable_share_pct = 0.0
    recycled_content_pct = 0.0
    recovery_rate_pct = 0.0

    for exchange in process.get("exchanges", []):
        if exchange.get("isInput"):
            upstream_inputs += 1
        category_tag, converted = _categorise_exchange(exchange)
        if converted is None:
            continue
        if category_tag == "electricity":
            electricity_kwh += converted
            # Attempt to infer renewable share from flow name keywords.
            flow_name = exchange.get("flow", {}).get("name", "").lower()
            if "renewable" in flow_name or "hydro" in flow_name or "solar" in flow_name:
                renewable_share_pct += converted
        elif category_tag == "fuel":
            fuel_l += converted
        elif category_tag == "transport":
            transport_tkm += converted

    # Normalise by reference throughput (per tonne values).
    electricity_kwh_per_t = electricity_kwh / ref_amount
    fuel_l_per_t = fuel_l / ref_amount
    transport_tkm_per_t = transport_tkm / ref_amount

    # Estimate renewable share percentage relative to electricity use if possible.
    if electricity_kwh > 0:
        renewable_share_pct = min(100.0, max(0.0, 100.0 * renewable_share_pct / electricity_kwh))
    else:
        renewable_share_pct = 0.0

    # EoL exchanges for circularity proxies.
    for exchange in process.get("exchanges", []):
        if not exchange.get("isInput"):
            flow_name = exchange.get("flow", {}).get("name", "").lower()
            amount = exchange.get("amount", 0.0)
            if "recycled" in flow_name:
                recycled_content_pct = max(recycled_content_pct, 100.0 * amount / max(ref_amount, 1e-9))
            if "recovery" in flow_name or "reused" in flow_name:
                recovery_rate_pct = max(recovery_rate_pct, 100.0 * amount / max(ref_amount, 1e-9))

    features = {
        "electricity_kwh_per_t": electricity_kwh_per_t,
        "fuel_l_per_t": fuel_l_per_t,
        "transport_tkm_per_t": transport_tkm_per_t,
        "upstream_input_count": float(upstream_inputs),
        "renewable_share_pct": renewable_share_pct,
        "recycled_content_pct": recycled_content_pct,
        "recovery_rate_pct": recovery_rate_pct,
    }

    process_id = process.get("@id") or process.get("refId") or process.get("name", "unknown")
    name = process.get("name", process_id)

    return ProcessRecord(
        process_id=process_id,
        name=name,
        category=category,
        location=location,
        reference_amount_t=ref_amount,
        features=features,
    )


def build_imputer_dataset(dataset_dir: Path, target: str = "electricity_kwh_per_t") -> pd.DataFrame:
    """Build a DataFrame containing features + target for the imputer model."""

    rows: List[Dict[str, float]] = []
    for _, process in _iter_process_json(Path(dataset_dir)):
        record = extract_process_features(process)
        if record is None:
            continue
        features = record.features.copy()
        if not math.isfinite(features.get(target, float("nan"))):
            continue
        rows.append(
            {
                "process_id": record.process_id,
                "name": record.name,
                "category": record.category,
                "location": record.location,
                **features,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No processes with the requested target were found in the dataset")
    return df


def _synthetic_sweeps(base_row: pd.Series, sweep_size: int = 20) -> pd.DataFrame:
    """Generate synthetic perturbations around a process feature vector."""

    rng = np.random.default_rng(42)
    sweeps = []
    for _ in range(sweep_size):
        row = base_row.copy()
        row["recycled_content_pct"] = float(np.clip(row["recycled_content_pct"] + rng.normal(5, 10), 0, 95))
        row["renewable_share_pct"] = float(np.clip(row["renewable_share_pct"] + rng.normal(10, 15), 0, 100))
        row["electricity_kwh_per_t"] = float(np.clip(row["electricity_kwh_per_t"] * rng.uniform(0.7, 1.3), 0, None))
        row["transport_tkm_per_t"] = float(np.clip(row["transport_tkm_per_t"] * rng.uniform(0.5, 1.5), 0, None))
        row["fuel_l_per_t"] = float(np.clip(row["fuel_l_per_t"] * rng.uniform(0.5, 1.4), 0, None))
        sweeps.append(row)
    return pd.DataFrame(sweeps)


def build_surrogate_dataset(
    dataset_dir: Path,
    gwp_column: str = "gwp_kg_co2e_per_t",
    circ_column: str = "circularity_score",
    synthetic_multiplier: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Return feature matrix X and targets dict for surrogate training.

    The OpenLCA export typically stores LCIA results separately. For a hackathon
    workflow we expect a pre-computed CSV named ``lcia_results.csv`` in the
    dataset directory with the columns ``process_id``, ``gwp_kg_co2e_per_t`` and
    ``circularity_score``. The helper merges engineered features with this
    table. When the table is missing the function synthesises placeholder
    targets from the available features so the rest of the pipeline can be
    demoed without full LCIA integration.
    """

    feature_df = build_imputer_dataset(dataset_dir)
    results_path = Path(dataset_dir) / "lcia_results.csv"
    targets: Dict[str, np.ndarray]

    if results_path.exists():
        lcia_df = pd.read_csv(results_path)
        merged = feature_df.merge(lcia_df, on="process_id", how="left")
        if merged[gwp_column].isna().any():
            merged[gwp_column] = merged[gwp_column].fillna(merged[gwp_column].median())
        if merged[circ_column].isna().any():
            merged[circ_column] = merged[circ_column].fillna(merged[circ_column].median())
    else:
        merged = feature_df.copy()
        # Lightweight synthetic targets as fallbacks.
        merged[gwp_column] = (
            0.8 * merged["electricity_kwh_per_t"]
            + 1.2 * merged["fuel_l_per_t"]
            + 0.1 * merged["transport_tkm_per_t"]
            - 0.3 * merged["recycled_content_pct"]
            - 0.2 * merged["renewable_share_pct"]
        )
        merged[circ_column] = (
            0.4 * merged["recycled_content_pct"]
            + 0.6 * merged["recovery_rate_pct"]
            - 0.05 * merged["transport_tkm_per_t"]
        )

    sweep_frames = [merged]
    if synthetic_multiplier > 0:
        for _, row in merged.iterrows():
            sweep_frames.append(_synthetic_sweeps(row, sweep_size=synthetic_multiplier))
    augmented = pd.concat(sweep_frames, ignore_index=True)

    feature_cols = list_feature_columns()

    targets = {
        "gwp": augmented[gwp_column].to_numpy(dtype=float),
        "circularity": augmented[circ_column].to_numpy(dtype=float),
    }

    X = augmented[feature_cols].to_numpy(dtype=float)
    return augmented[feature_cols], targets


def list_feature_columns() -> List[str]:
    """Return the canonical ordering of feature columns used by the models."""

    return [
        "electricity_kwh_per_t",
        "fuel_l_per_t",
        "transport_tkm_per_t",
        "renewable_share_pct",
        "recycled_content_pct",
        "recovery_rate_pct",
        "upstream_input_count",
    ]

