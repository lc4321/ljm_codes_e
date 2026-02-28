from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

from data import DataBundle
from metrics import regression_metrics


def get_device(device: str) -> torch.device:
    device = (device or "auto").lower()
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainResult:
    model_name: str
    checkpoint_path: str
    history: pd.DataFrame
    metrics_by_split: Dict[str, Dict[str, float]]
    predictions_by_split: Dict[str, pd.DataFrame]  # columns: [id,true,pred]


def _predict_scaled(model: nn.Module, X_scaled: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(X_scaled, dtype=torch.float32, device=device)
        y_hat = model(x).detach().cpu().numpy().reshape(-1)
    return y_hat


def _eval_split(
    model: nn.Module,
    X_scaled: np.ndarray,
    y_raw: np.ndarray,
    ids: np.ndarray,
    data: DataBundle,
    device: torch.device,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    pred_scaled = _predict_scaled(model, X_scaled, device)
    pred_raw = data.inverse_transform_y(pred_scaled)
    metrics = regression_metrics(y_raw, pred_raw)
    df = pd.DataFrame({data.id_name: ids, "y_true": y_raw, "y_pred": pred_raw})
    return metrics, df


def train_one_model(
    model: nn.Module,
    model_name: str,
    data: DataBundle,
    out_dir: str | Path,
    *,
    device: torch.device,
    epochs: int,
    patience: int,
    batch_size: int,
    num_workers: int,
    learning_rate: float,
    weight_decay: float,
    noise_std: float,
    grad_clip: float,
) -> TrainResult:
    out_dir = Path(out_dir)
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = model_dir / f"{model_name}_best.pt"

    train_loader, val_loader, _ = data.make_loaders(batch_size=batch_size, num_workers=num_workers)

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(5, patience // 6), verbose=False
    )

    best_val_r2 = -np.inf
    best_epoch = -1
    bad_epochs = 0
    best_state = None

    history_rows = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).view(-1)
            if noise_std and noise_std > 0:
                xb = xb + torch.randn_like(xb) * float(noise_std)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb).view(-1)
            loss = criterion(pred, yb)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()
            train_losses.append(loss.detach().cpu().item())

        # Evaluate on full train/val (original scale R2)
        train_pred_scaled = _predict_scaled(model, data.X_train_scaled, device)
        val_pred_scaled = _predict_scaled(model, data.X_val_scaled, device)
        train_pred_raw = data.inverse_transform_y(train_pred_scaled)
        val_pred_raw = data.inverse_transform_y(val_pred_scaled)

        train_loss_epoch = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss_epoch = float(
            criterion(
                torch.as_tensor(val_pred_scaled, dtype=torch.float32),
                torch.as_tensor(data.y_val_scaled.reshape(-1), dtype=torch.float32),
            ).item()
        )
        train_r2 = float(r2_score(data.y_train_raw, train_pred_raw))
        val_r2 = float(r2_score(data.y_val_raw, val_pred_raw))
        lr = float(optimizer.param_groups[0]["lr"])

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_epoch,
                "val_loss": val_loss_epoch,
                "train_r2": train_r2,
                "val_r2": val_r2,
                "lr": lr,
            }
        )

        scheduler.step(val_loss_epoch)

        improved = val_r2 > best_val_r2 + 1e-8
        if improved:
            best_val_r2 = val_r2
            best_epoch = epoch
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            break

    history = pd.DataFrame(history_rows)

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(
            {
                "model_name": model_name,
                "best_epoch": best_epoch,
                "best_val_r2": float(best_val_r2),
                "state_dict": best_state,
            },
            ckpt_path,
        )

    metrics_by_split: Dict[str, Dict[str, float]] = {}
    predictions_by_split: Dict[str, pd.DataFrame] = {}
    for split, Xs, yr, ids in [
        ("train", data.X_train_scaled, data.y_train_raw, data.id_train),
        ("val", data.X_val_scaled, data.y_val_raw, data.id_val),
        ("test", data.X_test_scaled, data.y_test_raw, data.id_test),
    ]:
        m, df = _eval_split(model, Xs, yr, ids, data, device)
        metrics_by_split[split] = m
        predictions_by_split[split] = df

    return TrainResult(
        model_name=model_name,
        checkpoint_path=str(ckpt_path),
        history=history,
        metrics_by_split=metrics_by_split,
        predictions_by_split=predictions_by_split,
    )

