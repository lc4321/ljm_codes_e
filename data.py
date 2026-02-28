from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
import torch
from torch.utils.data import DataLoader, Dataset


class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def create_scaler(name: str, random_state: int) -> object:
    name = name.lower()
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    if name == "robust":
        return RobustScaler()
    if name == "maxabs":
        return MaxAbsScaler()
    if name == "quantile":
        return QuantileTransformer(
            n_quantiles=100,
            output_distribution="uniform",
            random_state=random_state,
        )
    if name == "power":
        return PowerTransformer(method="yeo-johnson", standardize=True)
    raise ValueError(f"Unknown scaler: {name}")


@dataclass(frozen=True)
class DataBundle:
    df_raw: pd.DataFrame
    id_name: str
    target_name: str
    feature_names: list[str]

    split_assignments: pd.DataFrame  # columns: [id, split]

    x_scaler: object
    y_scaler: object

    X_train_scaled: np.ndarray
    X_val_scaled: np.ndarray
    X_test_scaled: np.ndarray

    y_train_scaled: np.ndarray
    y_val_scaled: np.ndarray
    y_test_scaled: np.ndarray

    X_train_raw: np.ndarray
    X_val_raw: np.ndarray
    X_test_raw: np.ndarray

    y_train_raw: np.ndarray
    y_val_raw: np.ndarray
    y_test_raw: np.ndarray

    id_train: np.ndarray
    id_val: np.ndarray
    id_test: np.ndarray

    def make_loaders(
        self, batch_size: int, num_workers: int = 0
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_loader = DataLoader(
            TabularDataset(self.X_train_scaled, self.y_train_scaled),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            TabularDataset(self.X_val_scaled, self.y_val_scaled),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        test_loader = DataLoader(
            TabularDataset(self.X_test_scaled, self.y_test_scaled),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        return train_loader, val_loader, test_loader

    def inverse_transform_y(self, y_scaled: np.ndarray) -> np.ndarray:
        y_scaled = np.asarray(y_scaled).reshape(-1, 1)
        y = self.y_scaler.inverse_transform(y_scaled)
        return y.reshape(-1)


def _choose_id_and_target_columns(
    df: pd.DataFrame, id_column: Optional[str], target_column: Optional[str]
) -> Tuple[str, str]:
    if df.shape[1] < 3:
        raise ValueError(f"Excel must contain at least 3 columns (id, features, target). Got {df.shape[1]}.")

    id_name = id_column or str(df.columns[0])
    target_name = target_column or str(df.columns[-1])
    if id_name not in df.columns:
        raise ValueError(f"id_column {id_name!r} not found in columns: {df.columns.tolist()}")
    if target_name not in df.columns:
        raise ValueError(f"target_column {target_name!r} not found in columns: {df.columns.tolist()}")
    if id_name == target_name:
        raise ValueError("id_column and target_column cannot be the same.")
    return id_name, target_name


def load_excel_as_tabular(
    path: str,
    id_column: Optional[str],
    target_column: Optional[str],
) -> Tuple[pd.DataFrame, str, str, list[str]]:
    df = pd.read_excel(path)
    if df.empty:
        raise ValueError("Excel is empty.")

    df = df.dropna(axis=1, how="all")
    df = df.reset_index(drop=True)

    id_name, target_name = _choose_id_and_target_columns(df, id_column, target_column)

    # Keep id, drop non-numeric feature columns (except id and target)
    cols = df.columns.tolist()
    keep_cols = [id_name, target_name]
    feature_cols = [c for c in cols if c not in keep_cols]

    non_numeric = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        df = df.drop(columns=non_numeric)
        feature_cols = [c for c in feature_cols if c not in non_numeric]

    if df[feature_cols + [target_name]].isnull().any().any():
        missing = df[feature_cols + [target_name]].isnull().sum()
        missing = missing[missing > 0]
        raise ValueError(f"Dataset contains missing values:\n{missing}")

    if len(feature_cols) < 1:
        raise ValueError("No numeric feature columns found.")

    return df, id_name, target_name, feature_cols


def build_data_bundle(
    excel_path: str,
    id_column: Optional[str],
    target_column: Optional[str],
    test_size: float,
    val_size: float,
    random_state: int,
    scaler_x: str,
    scaler_y: str,
) -> DataBundle:
    df, id_name, target_name, feature_names = load_excel_as_tabular(
        excel_path, id_column=id_column, target_column=target_column
    )

    ids = df[id_name].to_numpy()
    X = df[feature_names].to_numpy(dtype=np.float64)
    y = df[target_name].to_numpy(dtype=np.float64).reshape(-1, 1)

    # Split train/val/test
    X_temp, X_test, y_temp, y_test, id_temp, id_test = train_test_split(
        X,
        y,
        ids,
        test_size=test_size,
        random_state=random_state,
    )
    val_size_adjusted = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(
        X_temp,
        y_temp,
        id_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
    )

    # Fit scalers on train only (avoid leakage)
    x_scaler_obj = create_scaler(scaler_x, random_state)
    y_scaler_obj = create_scaler(scaler_y, random_state)

    X_train_scaled = x_scaler_obj.fit_transform(X_train)
    X_val_scaled = x_scaler_obj.transform(X_val)
    X_test_scaled = x_scaler_obj.transform(X_test)

    y_train_scaled = y_scaler_obj.fit_transform(y_train)
    y_val_scaled = y_scaler_obj.transform(y_val)
    y_test_scaled = y_scaler_obj.transform(y_test)

    split_assignments = pd.DataFrame(
        {id_name: np.concatenate([id_train, id_val, id_test]),
         "split": (["train"] * len(id_train)) + (["val"] * len(id_val)) + (["test"] * len(id_test))}
    )

    return DataBundle(
        df_raw=df,
        id_name=id_name,
        target_name=target_name,
        feature_names=list(feature_names),
        split_assignments=split_assignments,
        x_scaler=x_scaler_obj,
        y_scaler=y_scaler_obj,
        X_train_scaled=X_train_scaled.astype(np.float32),
        X_val_scaled=X_val_scaled.astype(np.float32),
        X_test_scaled=X_test_scaled.astype(np.float32),
        y_train_scaled=y_train_scaled.astype(np.float32),
        y_val_scaled=y_val_scaled.astype(np.float32),
        y_test_scaled=y_test_scaled.astype(np.float32),
        X_train_raw=X_train.astype(np.float32),
        X_val_raw=X_val.astype(np.float32),
        X_test_raw=X_test.astype(np.float32),
        y_train_raw=y_train.reshape(-1).astype(np.float32),
        y_val_raw=y_val.reshape(-1).astype(np.float32),
        y_test_raw=y_test.reshape(-1).astype(np.float32),
        id_train=id_train,
        id_val=id_val,
        id_test=id_test,
    )

