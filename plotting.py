from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


def setup_matplotlib(chinese: bool = True) -> None:
    try:
        from matplotlib import font_manager as fm

        candidates = [
            "Noto Sans CJK SC",
            "Noto Sans CJK JP",
            "Noto Sans CJK TC",
            "WenQuanYi Micro Hei",
            "WenQuanYi Zen Hei",
            "SimHei",
            "Microsoft YaHei",
            "PingFang SC",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]
        available = {f.name for f in fm.fontManager.ttflist}
        chosen = [name for name in candidates if name in available]
        if not chosen:
            chosen = ["DejaVu Sans"]

        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": chosen,
                "axes.unicode_minus": False,
                "figure.dpi": 100,
                "savefig.dpi": 300,
                "axes.grid": True,
                "grid.alpha": 0.25,
                "grid.linestyle": "--",
            }
        )
        if not chinese:
            plt.rcParams.update({"font.sans-serif": ["DejaVu Sans"]})
    except Exception:
        plt.rcParams.update({"axes.unicode_minus": False})


def _ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def savefig(path: str | Path) -> str:
    p = _ensure_parent(path)
    plt.tight_layout()
    plt.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close()
    return str(p)


def plot_training_curves(
    history: pd.DataFrame, out_path: str | Path, model_name: str
) -> str:
    if history.empty:
        return ""

    epochs = history["epoch"].to_numpy()
    train_loss = history["train_loss"].to_numpy()
    val_loss = history["val_loss"].to_numpy()
    train_r2 = history["train_r2"].to_numpy()
    val_r2 = history["val_r2"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1, ax2 = axes

    ax1.plot(epochs, train_loss, label="train_loss")
    ax1.plot(epochs, val_loss, label="val_loss")
    ax1.set_title(f"{model_name} - Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (MSE on scaled y)")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.legend()

    ax2.plot(epochs, train_r2, label="train_R2")
    ax2.plot(epochs, val_r2, label="val_R2")
    ax2.set_title(f"{model_name} - R² Curve")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("R² (original scale)")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.legend()

    return savefig(out_path)


def plot_loss_curves_comparison(
    histories_by_model: Dict[str, pd.DataFrame],
    out_path: str | Path,
    *,
    title: str = "Loss Curves (Train/Val)",
) -> str:
    """
    Single figure: compare train/val loss curves across multiple models.
    """
    setup_matplotlib(chinese=True)
    items = [(name, df) for name, df in histories_by_model.items() if isinstance(df, pd.DataFrame) and not df.empty]
    if not items:
        return ""

    # Keep stable order for readability
    items = sorted(items, key=lambda x: str(x[0]).lower())
    n = len(items)
    cmap = plt.get_cmap("tab10")
    palette = [cmap(i % cmap.N) for i in range(max(1, n))]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6))
    ax_tr, ax_val = axes

    for (model_name, history), color in zip(items, palette):
        if not {"epoch", "train_loss", "val_loss"}.issubset(set(history.columns)):
            continue
        epochs = history["epoch"].to_numpy()
        ax_tr.plot(epochs, history["train_loss"].to_numpy(), label=str(model_name), color=color, linewidth=2.0)
        ax_val.plot(epochs, history["val_loss"].to_numpy(), label=str(model_name), color=color, linewidth=2.0)

    ax_tr.set_title("Train Loss")
    ax_val.set_title("Val Loss")
    for ax in (ax_tr, ax_val):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE on scaled y)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    handles, labels = ax_tr.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False)
        ax_tr.legend().remove()
        ax_val.legend().remove()

    fig.suptitle(title, y=1.02, fontsize=14, fontweight="bold")
    fig.subplots_adjust(bottom=0.18)
    return savefig(out_path)


def plot_metrics_comparison(
    metrics_df: pd.DataFrame, out_dir: str | Path, split: str
) -> Dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, str] = {}
    df = metrics_df.copy()
    df = df[df["split"] == split].copy()
    if df.empty:
        return paths

    for metric in ["R2", "RMSE", "MAE", "MAPE", "SMAPE"]:
        fig, ax = plt.subplots(figsize=(10, 5))
        order = df.sort_values(metric, ascending=(metric != "R2"))["model"].tolist()
        df_m = df.set_index("model").loc[order]
        values = df_m[metric].to_numpy(dtype=float, copy=False)
        xs = np.arange(len(order), dtype=float)
        colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, num=max(1, len(order))))
        ax.bar(xs, values, color=colors, edgecolor="none")
        ax.set_title(f"{split} - {metric} Model Comparison")
        ax.set_xticks(xs)
        ax.set_xticklabels(order, rotation=30, ha="right")
        ax.set_xlabel("model")
        ax.set_ylabel(metric)
        path = out_dir / f"{split}_{metric}_comparison.png"
        paths[metric] = savefig(path)

    return paths


def plot_true_vs_pred(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str | Path,
    title: str,
    max_points: int = 2000,
) -> str:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    n = len(y_true)
    if n > max_points:
        idx = np.random.RandomState(42).choice(n, size=max_points, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]

    fig = plt.figure(figsize=(6.5, 6.5))
    plt.scatter(y_true, y_pred, s=18, alpha=0.7, edgecolors="none")
    vmin = float(min(y_true.min(), y_pred.min()))
    vmax = float(max(y_true.max(), y_pred.max()))
    plt.plot([vmin, vmax], [vmin, vmax], color="black", linewidth=1.5, linestyle="--", label="y = x")
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.title(title)
    plt.legend(loc="best")
    plt.axis("equal")
    plt.xlim(vmin, vmax)
    plt.ylim(vmin, vmax)
    return savefig(out_path)


def plot_residual_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str | Path,
    title: str,
    max_points: int = 2000,
) -> str:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    resid = y_pred - y_true
    n = len(y_true)
    if n > max_points:
        idx = np.random.RandomState(42).choice(n, size=max_points, replace=False)
        y_true_s = y_true[idx]
        y_pred_s = y_pred[idx]
        resid_s = resid[idx]
    else:
        y_true_s, y_pred_s, resid_s = y_true, y_pred, resid

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax1, ax2, ax3 = axes

    ax1.scatter(y_pred_s, resid_s, s=18, alpha=0.7, edgecolors="none")
    ax1.axhline(0.0, color="black", linewidth=1.0)
    ax1.set_title("Residual vs Pred")
    ax1.set_xlabel("Pred")
    ax1.set_ylabel("Residual (Pred-True)")

    ax2.hist(resid_s, bins=30, color="steelblue", alpha=0.85, edgecolor="white", linewidth=0.5)
    ax2.set_title("Residual Distribution")
    ax2.set_xlabel("Residual")
    ax2.set_ylabel("Count")

    abs_err = np.abs(resid_s)
    ax3.scatter(y_true_s, abs_err, s=18, alpha=0.7, edgecolors="none")
    ax3.set_title("|Error| vs True")
    ax3.set_xlabel("True")
    ax3.set_ylabel("Absolute Error")

    fig.suptitle(title, y=1.02, fontsize=14, fontweight="bold")
    return savefig(out_path)


def plot_error_cdf(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str | Path,
    title: str,
) -> str:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    abs_err = np.abs(y_pred - y_true)
    x = np.sort(abs_err)
    y = np.linspace(0, 1, num=len(x), endpoint=True)
    fig = plt.figure(figsize=(7, 5))
    plt.plot(x, y, linewidth=2.0, color="steelblue")
    plt.title(title)
    plt.xlabel("Absolute Error")
    plt.ylabel("CDF")
    plt.grid(True, alpha=0.3)
    return savefig(out_path)


def plot_residual_qq(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str | Path,
    title: str,
) -> str:
    from scipy import stats

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    resid = y_pred - y_true
    fig = plt.figure(figsize=(6.5, 6.5))
    stats.probplot(resid, dist="norm", plot=plt)
    plt.title(title)
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Ordered Residuals")
    return savefig(out_path)


def plot_feature_importance_bar(
    importance_df: pd.DataFrame,
    out_path: str | Path,
    title: str,
    value_col: str,
    top_k: int = 20,
) -> str:
    df = importance_df.copy()
    df = df.sort_values(value_col, ascending=False).head(top_k)
    df = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(df) + 1)))
    vals = df[value_col].to_numpy(dtype=float, copy=False)
    feats = df["feature"].astype(str).to_list()
    ys = np.arange(len(df), dtype=float)
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, num=max(1, len(df))))
    ax.barh(ys, vals, color=colors, edgecolor="none")
    ax.set_yticks(ys)
    ax.set_yticklabels(feats)
    ax.set_title(title)
    ax.set_xlabel(value_col)
    ax.set_ylabel("feature")
    return savefig(out_path)


def plot_heatmap(
    matrix: pd.DataFrame, out_path: str | Path, title: str, center: float = 0.0
) -> str:
    df = matrix.copy()
    values = df.to_numpy(dtype=float, copy=False)
    fig, ax = plt.subplots(figsize=(12, 10))
    vmax = float(np.nanmax(np.abs(values))) if values.size else 1.0
    vmin = -vmax
    im = ax.imshow(values, cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_xticklabels([str(c) for c in df.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(df.shape[0]))
    ax.set_yticklabels([str(i) for i in df.index], rotation=0)

    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            v = values[i, j]
            if not np.isfinite(v):
                txt = "nan"
            else:
                txt = f"{float(v):.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return savefig(out_path)


def plot_pdp(
    pdp_df: pd.DataFrame,
    out_path: str | Path,
    title: str,
    x_col: str = "x",
    y_col: str = "y_mean",
) -> str:
    x = pd.to_numeric(pdp_df[x_col], errors="coerce").to_numpy()
    y = pd.to_numeric(pdp_df[y_col], errors="coerce").to_numpy()
    fig = plt.figure(figsize=(7.5, 5.0))
    plt.plot(x, y, marker="o", linewidth=2.0)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True, alpha=0.25, linestyle="--")
    return savefig(out_path)
