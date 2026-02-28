from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _to_1d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 2 and a.shape[1] == 1:
        return a[:, 0]
    return a.reshape(-1)


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    y_true = _to_1d(y_true)
    y_pred = _to_1d(y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    y_true = _to_1d(y_true)
    y_pred = _to_1d(y_pred)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100.0)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = _to_1d(y_true)
    y_pred = _to_1d(y_pred)
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "MSE": mse,
        "MAPE": mape(y_true, y_pred),
        "SMAPE": smape(y_true, y_pred),
    }


@dataclass(frozen=True)
class SplitMetrics:
    train: Dict[str, float]
    val: Dict[str, float]
    test: Dict[str, float]

