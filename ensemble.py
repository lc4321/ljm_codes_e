from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from metrics import regression_metrics


@dataclass
class EnsembleResult:
    name: str
    metrics_by_split: Dict[str, Dict[str, float]]
    predictions_by_split: Dict[str, pd.DataFrame]  # columns: [id,true,pred]
    extra: Dict[str, object]


def _stack_preds(predictions_by_model: Dict[str, pd.DataFrame], id_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
   
    model_names = list(predictions_by_model.keys())
    base = predictions_by_model[model_names[0]].copy()
    base = base[[id_col, "y_true", "y_pred"]].rename(columns={"y_pred": model_names[0]})
    for name in model_names[1:]:
        df = predictions_by_model[name][[id_col, "y_pred"]].rename(columns={"y_pred": name})
        base = base.merge(df, on=id_col, how="inner")
    ids = base[id_col].to_numpy()
    y_true = base["y_true"].to_numpy()
    X_meta = base[model_names].to_numpy()
    return ids, y_true, X_meta


def weighted_average_ensemble(
    *,
    id_col: str,
    base_predictions: Dict[str, Dict[str, pd.DataFrame]],
    base_metrics: Dict[str, Dict[str, Dict[str, float]]],
    weight_power: float = 2.0,
    name: str = "Optimized_Ensemble_Weighted",
) -> EnsembleResult:
    # weights from val R2 (non-negative)
    r2s = {m: max(0.0, float(base_metrics[m]["val"]["R2"])) for m in base_metrics.keys()}
    raw_w = {m: (r2s[m] ** weight_power) for m in r2s.keys()}
    s = sum(raw_w.values())
    if s <= 0:
        weights = {m: 1.0 / len(raw_w) for m in raw_w.keys()}
    else:
        weights = {m: raw_w[m] / s for m in raw_w.keys()}

    metrics_by_split: Dict[str, Dict[str, float]] = {}
    predictions_by_split: Dict[str, pd.DataFrame] = {}

    for split in ["train", "val", "test"]:
      
        preds_for_split = {m: base_predictions[m][split] for m in weights.keys()}
        ids, y_true, X_meta = _stack_preds(preds_for_split, id_col=id_col)
        
        model_names = list(weights.keys())
        w = np.array([weights[m] for m in model_names], dtype=np.float64).reshape(1, -1)
        y_pred = (X_meta * w).sum(axis=1)
        metrics_by_split[split] = regression_metrics(y_true, y_pred)
        df = pd.DataFrame({id_col: ids, "y_true": y_true, "y_pred": y_pred})
        predictions_by_split[split] = df

    return EnsembleResult(
        name=name,
        metrics_by_split=metrics_by_split,
        predictions_by_split=predictions_by_split,
        extra={"weights": weights, "weight_power": weight_power, "base_r2_val": r2s},
    )


def ridge_stacking_ensemble(
    *,
    id_col: str,
    base_predictions: Dict[str, Dict[str, pd.DataFrame]],
    alpha: float = 1.0,
    name: str = "Optimized_Ensemble_Ridge",
) -> EnsembleResult:
    # Build training meta set from train predictions (no test leakage)
    train_preds = {m: base_predictions[m]["train"] for m in base_predictions.keys()}
    val_preds = {m: base_predictions[m]["val"] for m in base_predictions.keys()}
    test_preds = {m: base_predictions[m]["test"] for m in base_predictions.keys()}

    ids_tr, y_tr, X_tr = _stack_preds(train_preds, id_col=id_col)
    _, y_val, X_val = _stack_preds(val_preds, id_col=id_col)
    ids_te, y_te, X_te = _stack_preds(test_preds, id_col=id_col)

    model_names = list(base_predictions.keys())
    ridge = Ridge(alpha=float(alpha), fit_intercept=True, random_state=42)
    ridge.fit(X_tr, y_tr)

    def predict_df(ids: np.ndarray, y_true: np.ndarray, X: np.ndarray) -> pd.DataFrame:
        y_pred = ridge.predict(X)
        return pd.DataFrame({id_col: ids, "y_true": y_true, "y_pred": y_pred})

    metrics_by_split: Dict[str, Dict[str, float]] = {}
    predictions_by_split: Dict[str, pd.DataFrame] = {}

    # train
    y_pred_tr = ridge.predict(X_tr)
    metrics_by_split["train"] = regression_metrics(y_tr, y_pred_tr)
    predictions_by_split["train"] = pd.DataFrame({id_col: ids_tr, "y_true": y_tr, "y_pred": y_pred_tr})

    # val (ids from first val df)
    ids_val = val_preds[model_names[0]][id_col].to_numpy()
    y_pred_val = ridge.predict(X_val)
    metrics_by_split["val"] = regression_metrics(y_val, y_pred_val)
    predictions_by_split["val"] = pd.DataFrame({id_col: ids_val, "y_true": y_val, "y_pred": y_pred_val})

    # test
    y_pred_te = ridge.predict(X_te)
    metrics_by_split["test"] = regression_metrics(y_te, y_pred_te)
    predictions_by_split["test"] = pd.DataFrame({id_col: ids_te, "y_true": y_te, "y_pred": y_pred_te})

    coef = ridge.coef_.reshape(-1).tolist()
    intercept = float(ridge.intercept_)
    coef_map = {m: float(c) for m, c in zip(model_names, coef)}

    return EnsembleResult(
        name=name,
        metrics_by_split=metrics_by_split,
        predictions_by_split=predictions_by_split,
        extra={"alpha": float(alpha), "intercept": intercept, "coefficients": coef_map},
    )

