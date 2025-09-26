# TatvaChakra AI Toolkit

This repository contains a hackathon-ready machine learning toolkit that powers
AI-assisted sustainability workflows for TatvaChakra. It includes:

- Feature engineering utilities that transform OpenLCA exports into tabular
  datasets ready for modelling.
- Training scripts for the missing-value imputer and the what-if surrogate
  models.
- A FastAPI backend that exposes the imputer, what-if optimisation,
  LCIA method recommender and input validation endpoints.

## Project structure

```
├── backend/
│   └── app/
│       ├── main.py               # FastAPI application
│       ├── model_store.py        # Model loading helper
│       └── models.py             # Pydantic schemas & helpers
├── ml/
│   ├── feature_engineering.py    # OpenLCA feature extraction utilities
│   └── train_models.py           # Offline training pipeline
├── models/                       # Directory for trained `.joblib` artifacts
├── requirements.txt              # Python dependencies
└── README.md
```

## Getting started

1. **Install dependencies** (Python 3.10+ recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Prepare training data** using an OpenLCA directory export. Place the export
   (e.g. `data/openlca_export/`) on disk. Optionally include a
   `lcia_results.csv` with columns `process_id`, `gwp_kg_co2e_per_t` and
   `circularity_score` to use real LCIA results. Without this file synthetic
   targets are generated automatically for demo purposes.

3. **Train the models**:

   ```bash
   python -m ml.train_models --dataset ./data/openlca_export --output-dir ./models
   ```

   The command saves the imputer and surrogate models to `./models` along with a
   `metadata.json` file containing evaluation metrics and feature statistics.

4. **Run the FastAPI service**:

   ```bash
   uvicorn backend.app.main:app --reload
   ```

   The service exposes the following endpoints:

   - `POST /api/impute` – Predicts electricity intensity (kWh/t) with
     confidence bounds and feature importances.
   - `POST /api/whatif` – Evaluates greedy optimisation moves using the
     surrogate model and returns the top actions.
   - `POST /api/recommend_lcia` – Rule-based LCIA method recommendations based
     on the user's goal description.
   - `POST /api/validate_inputs` – Flags out-of-bounds inputs using feasibility
     clamps and z-score anomaly detection.

## Development notes

- The feature engineering utilities are intentionally defensive: they tolerate
  missing metadata, infer units, and generate synthetic targets when LCIA
  results are unavailable.
- Model metadata contains `feature_columns`, mean absolute errors and feature
  statistics so the API can expose uncertainty ranges and perform validation.
- For richer explanations you can integrate SHAP or other XAI tooling by
  extending `backend/app/main.py`.

