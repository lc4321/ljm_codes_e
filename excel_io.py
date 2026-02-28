from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import pandas as pd


def write_excel(path: str | Path, sheets: Mapping[str, pd.DataFrame]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            safe_name = str(name)[:31]  # Excel sheet name limit
            df.to_excel(writer, sheet_name=safe_name, index=False)
    return str(path)


def build_metrics_long(metrics_by_model: Dict[str, Dict[str, Dict[str, float]]]) -> pd.DataFrame:
    rows = []
    for model_name, by_split in metrics_by_model.items():
        for split, metrics in by_split.items():
            row = {"model": model_name, "split": split}
            row.update(metrics)
            rows.append(row)
    return pd.DataFrame(rows)


def build_predictions_wide(
    predictions_by_model: Dict[str, Dict[str, pd.DataFrame]],
    id_col: str,
    split: str,
) -> pd.DataFrame:
    model_names = list(predictions_by_model.keys())
    base = predictions_by_model[model_names[0]][split][[id_col, "y_true", "y_pred"]].rename(
        columns={"y_pred": f"y_pred_{model_names[0]}"}
    )
    for name in model_names[1:]:
        df = predictions_by_model[name][split][[id_col, "y_pred"]].rename(columns={"y_pred": f"y_pred_{name}"})
        base = base.merge(df, on=id_col, how="inner")
    return base


def build_training_history_sheets(histories_by_model: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    sheets: Dict[str, pd.DataFrame] = {}
    for model_name, df in histories_by_model.items():
        sheets[model_name] = df.copy()
    return sheets

