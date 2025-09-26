"""Offline training pipeline for the TatvaChakra AI helpers.

The script orchestrates feature extraction, model training and artifact
persistence for both the missing-value imputer and the surrogate recommender
models. It is intentionally lightweight so it can run on a laptop without a
GPU during a hackathon. The resulting ``.joblib`` artifacts are loaded by the
FastAPI service at runtime.

Usage
-----

.. code-block:: bash

    python -m ml.train_models --dataset ./data/openlca_export \
        --output-dir ./models

The command expects an OpenLCA export directory (see
:mod:`ml.feature_engineering` for the exact assumptions). The output directory
will contain three files:

``imputer.joblib``
    Gradient Boosting model predicting ``electricity_kwh_per_t``.
``surrogate_gwp.joblib``
    Random Forest surrogate for total GWP.
``surrogate_circularity.joblib``
    Random Forest surrogate for the circularity score.

A ``metadata.json`` file is also stored to capture metrics such as MAE and R²
so the FastAPI layer can expose confidence bands.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from .feature_engineering import (
    build_imputer_dataset,
    build_surrogate_dataset,
    list_feature_columns,
)


def train_imputer(df: pd.DataFrame) -> Tuple[GradientBoostingRegressor, float, Dict[str, float]]:
    """Train the gradient boosting imputer and return model + MAE."""

    feature_cols = list_feature_columns()
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["electricity_kwh_per_t"].to_numpy(dtype=float)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)
    mae = float(mean_absolute_error(y_valid, y_pred))

    # Feature importances for transparency.
    importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
    return model, mae, importance


def train_surrogates(
    feature_df: pd.DataFrame, targets: Dict[str, np.ndarray]
) -> Tuple[RandomForestRegressor, RandomForestRegressor, Dict[str, float]]:
    """Train the surrogate regressors for GWP and circularity."""

    feature_cols = feature_df.columns.tolist()
    X = feature_df.to_numpy(dtype=float)
    indices = np.arange(len(X))
    train_idx, valid_idx = train_test_split(indices, test_size=0.2, random_state=42)

    rf_gwp = RandomForestRegressor(n_estimators=300, random_state=42)
    rf_circ = RandomForestRegressor(n_estimators=300, random_state=1337)

    X_train = X[train_idx]
    X_valid = X[valid_idx]
    gwp_train = targets["gwp"][train_idx]
    gwp_valid = targets["gwp"][valid_idx]
    circ_train = targets["circularity"][train_idx]
    circ_valid = targets["circularity"][valid_idx]

    rf_gwp.fit(X_train, gwp_train)
    rf_circ.fit(X_train, circ_train)

    gwp_pred = rf_gwp.predict(X_valid)
    circ_pred = rf_circ.predict(X_valid)

    metrics = {
        "gwp_r2": float(r2_score(gwp_valid, gwp_pred)),
        "circ_r2": float(r2_score(circ_valid, circ_pred)),
        "gwp_mae": float(mean_absolute_error(gwp_valid, gwp_pred)),
        "circ_mae": float(mean_absolute_error(circ_valid, circ_pred)),
        "feature_columns": feature_cols,
    }
    return rf_gwp, rf_circ, metrics


def save_artifacts(
    output_dir: Path,
    imputer: GradientBoostingRegressor,
    surrogate_gwp: RandomForestRegressor,
    surrogate_circ: RandomForestRegressor,
    metadata: Dict,
) -> None:
    """Persist the trained models and metadata to ``output_dir``."""

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(imputer, output_dir / "imputer.joblib")
    joblib.dump(surrogate_gwp, output_dir / "surrogate_gwp.joblib")
    joblib.dump(surrogate_circ, output_dir / "surrogate_circularity.joblib")
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TatvaChakra ML helpers")
    parser.add_argument("--dataset", type=Path, required=True, help="OpenLCA export directory")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("models"), help="Where to store trained models"
    )
    parser.add_argument(
        "--synthetic-multiplier",
        type=int,
        default=10,
        help="Synthetic sweep count per process for surrogate augmentation",
    )
    args = parser.parse_args()

    print("[tatvachakra] Building imputer dataset...")
    imputer_df = build_imputer_dataset(args.dataset)
    imputer_model, imputer_mae, importance = train_imputer(imputer_df)

    print("[tatvachakra] Building surrogate dataset...")
    surrogate_features, targets = build_surrogate_dataset(
        args.dataset, synthetic_multiplier=args.synthetic_multiplier
    )
    surrogate_gwp, surrogate_circ, surrogate_metrics = train_surrogates(
        surrogate_features, targets
    )

    feature_stats = {
        col: {
            "mean": float(imputer_df[col].mean()),
            "std": float(imputer_df[col].std(ddof=0) or 0.0),
        }
        for col in list_feature_columns()
    }

    metadata = {
        "imputer_mae": imputer_mae,
        "imputer_importances": importance,
        "feature_stats": feature_stats,
        **surrogate_metrics,
    }

    print("[tatvachakra] Saving artifacts to", args.output_dir)
    save_artifacts(args.output_dir, imputer_model, surrogate_gwp, surrogate_circ, metadata)
    print("[tatvachakra] Done!")


if __name__ == "__main__":
    main()

