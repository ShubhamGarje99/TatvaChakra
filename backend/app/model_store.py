"""Runtime model loading utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import joblib
from sklearn.base import RegressorMixin


class ModelStore:
    """Lightweight loader that keeps ML artifacts in memory."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory {model_dir} does not exist")
        self._imputer: Optional[RegressorMixin] = None
        self._surrogate_gwp: Optional[RegressorMixin] = None
        self._surrogate_circularity: Optional[RegressorMixin] = None
        self.metadata: Dict = {}

    def load(self) -> None:
        self._imputer = joblib.load(self.model_dir / "imputer.joblib")
        self._surrogate_gwp = joblib.load(self.model_dir / "surrogate_gwp.joblib")
        self._surrogate_circularity = joblib.load(
            self.model_dir / "surrogate_circularity.joblib"
        )
        metadata_path = self.model_dir / "metadata.json"
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as handle:
                self.metadata = json.load(handle)

    @property
    def imputer(self) -> RegressorMixin:
        if self._imputer is None:
            raise RuntimeError("ModelStore.load() must be called before accessing models")
        return self._imputer

    @property
    def surrogate_gwp(self) -> RegressorMixin:
        if self._surrogate_gwp is None:
            raise RuntimeError("ModelStore.load() must be called before accessing models")
        return self._surrogate_gwp

    @property
    def surrogate_circularity(self) -> RegressorMixin:
        if self._surrogate_circularity is None:
            raise RuntimeError("ModelStore.load() must be called before accessing models")
        return self._surrogate_circularity

