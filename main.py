from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_tmpdir() -> None:
   
    try:
        import tempfile

        fd, p = tempfile.mkstemp()
        os.close(fd)
        try:
            os.remove(p)
        except Exception:
            pass
        return
    except Exception:
        pass

    fallback = Path(__file__).resolve().parent / ".tmp"
    try:
        fallback.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    os.environ["TMPDIR"] = str(fallback)
    os.environ.setdefault("TMP", str(fallback))
    os.environ.setdefault("TEMP", str(fallback))


_ensure_tmpdir()

import argparse
import warnings
from typing import Dict

import numpy as np
import pandas as pd
import torch

from config import ExperimentConfig
from data import build_data_bundle
from ensemble import ridge_stacking_ensemble, weighted_average_ensemble
from excel_io import build_metrics_long, build_predictions_wide, build_training_history_sheets, write_excel
from models import build_models, count_parameters
from plotting import (
    plot_loss_curves_comparison,
    plot_metrics_comparison,
    plot_residual_diagnostics,
    plot_residual_qq,
    plot_error_cdf,
    plot_training_curves,
    plot_true_vs_pred,
    setup_matplotlib,
)
from trainer import get_device, train_one_model
from utils import chdir_project_root, ensure_dir, now_run_id, save_json, set_global_seed


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="codes_e_new_20251225 - training/evaluation/export")
    parser.add_argument("--data", dest="data_path", default=ExperimentConfig.data_path)
    parser.add_argument("--test-size", type=float, default=ExperimentConfig.test_size)
    parser.add_argument("--val-size", type=float, default=ExperimentConfig.val_size)
    parser.add_argument("--seed", type=int, default=ExperimentConfig.random_state)
    parser.add_argument("--device", type=str, default=ExperimentConfig.device)
    parser.add_argument("--epochs", type=int, default=ExperimentConfig.epochs)
    parser.add_argument("--patience", type=int, default=ExperimentConfig.patience)
    parser.add_argument("--batch-size", type=int, default=ExperimentConfig.batch_size)
    parser.add_argument("--lr", type=float, default=ExperimentConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=ExperimentConfig.weight_decay)
    parser.add_argument("--noise-std", type=float, default=ExperimentConfig.noise_std)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--quick", action="store_true", help="fewer epochs for pipeline sanity check")
    args = parser.parse_args()

    cfg = ExperimentConfig(
        data_path=args.data_path,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed,
        device=args.device,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        noise_std=args.noise_std,
        run_name=args.run_name,
        quick=bool(args.quick),
    )

    if cfg.quick:
        cfg = ExperimentConfig(
            **{
                **cfg.__dict__,
                "epochs": min(120, cfg.epochs),
                "patience": min(20, cfg.patience),
            }
        )

    return cfg


def main() -> int:
    chdir_project_root()
    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))

    warnings.filterwarnings("ignore", message="CUDA initialization:")
    warnings.filterwarnings("ignore", message="The NumPy global RNG was seeded")

    cfg = parse_args()
    setup_matplotlib(chinese=True)
    set_global_seed(cfg.random_state)

    run_id = cfg.run_name or now_run_id()
    run_dir = ensure_dir(Path(cfg.run_root) / run_id)
    models_dir = ensure_dir(run_dir / "models")
    excel_dir = ensure_dir(run_dir / "excel")
    figures_dir = ensure_dir(run_dir / "figures")

    save_json(run_dir / "config.json", cfg.__dict__)

    # 1) Data
    data = build_data_bundle(
        excel_path=cfg.data_path,
        id_column=cfg.id_column,
        target_column=cfg.target_column,
        test_size=cfg.test_size,
        val_size=cfg.val_size,
        random_state=cfg.random_state,
        scaler_x=cfg.scaler_x,
        scaler_y=cfg.scaler_y,
    )

    data_summary = {
        "data_path": cfg.data_path,
        "n_samples": int(data.df_raw.shape[0]),
        "n_features": int(len(data.feature_names)),
        "target": data.target_name,
        "id": data.id_name,
        "train_samples": int(len(data.id_train)),
        "val_samples": int(len(data.id_val)),
        "test_samples": int(len(data.id_test)),
    }

    # Save split assignments
    split_df = data.split_assignments.copy()
    split_excel = write_excel(excel_dir / "data_splits.xlsx", {"splits": split_df})

    # Save run info to Excel as well (方便在其他电脑复现/复绘)
    cfg_df = pd.DataFrame([{"key": k, "value": v} for k, v in cfg.__dict__.items()])
    data_df = pd.DataFrame([{"key": k, "value": v} for k, v in data_summary.items()])
    write_excel(excel_dir / "run_info.xlsx", {"config": cfg_df, "data_summary": data_df})

    # 2) Train base models
    device = get_device(cfg.device)
    input_dim = data.X_train_scaled.shape[1]

    all_models = build_models(input_dim=input_dim)
    models_selected = {k: v for k, v in all_models.items() if k in set(cfg.model_names)}

    base_results: Dict[str, object] = {}
    base_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    base_predictions: Dict[str, Dict[str, pd.DataFrame]] = {}
    histories: Dict[str, pd.DataFrame] = {}

    for model_name, model in models_selected.items():
        result = train_one_model(
            model=model,
            model_name=model_name,
            data=data,
            out_dir=run_dir,
            device=device,
            epochs=int(cfg.epochs),
            patience=int(cfg.patience),
            batch_size=int(cfg.batch_size),
            num_workers=int(cfg.num_workers),
            learning_rate=float(cfg.learning_rate),
            weight_decay=float(cfg.weight_decay),
            noise_std=float(cfg.noise_std),
            grad_clip=float(cfg.grad_clip),
        )
        base_results[model_name] = result
        base_metrics[model_name] = result.metrics_by_split
        base_predictions[model_name] = result.predictions_by_split
        histories[model_name] = result.history

        # per-model training curve
        plot_training_curves(
            result.history,
            figures_dir / f"{model_name}_training_curves.png",
            model_name=model_name,
        )

    # Single loss curve comparison across all base models
    plot_loss_curves_comparison(
        histories,
        figures_dir / "loss_curves_comparison.png",
        title="Loss Curves (Train/Val, Base Models)",
    )

    # 3) Ensembles (OE)
    ensemble_results = {}
    if cfg.enable_weighted_ensemble:
        ensemble_results["Optimized_Ensemble_Weighted"] = weighted_average_ensemble(
            id_col=data.id_name,
            base_predictions=base_predictions,
            base_metrics=base_metrics,
            name="Optimized_Ensemble_Weighted",
        )
    if cfg.enable_ridge_ensemble:
        ensemble_results["Optimized_Ensemble_Ridge"] = ridge_stacking_ensemble(
            id_col=data.id_name,
            base_predictions=base_predictions,
            alpha=1.0,
            name="Optimized_Ensemble_Ridge",
        )

    
    import pickle

    for ens_name, ens in ensemble_results.items():
        base_model_names = list(models_selected.keys())
        payload: Dict[str, object] = {
            "schema_version": 1,
            "name": ens_name,
            "base_models": base_model_names,
        }
        if "weights" in ens.extra:
            def _finite_or_none(v: object) -> float | None:
                try:
                    vv = float(v)  # type: ignore[arg-type]
                    return vv if np.isfinite(vv) else None
                except Exception:
                    return None

            payload.update(
                {
                    "type": "weighted_average",
                    "weight_source_split": "val",
                    "weight_source_metric": "R2",
                    "weight_power": float(ens.extra.get("weight_power", 2.0)),
                    "base_r2_val": {k: _finite_or_none(ens.extra.get("base_r2_val", {}).get(k)) for k in base_model_names},
                    "weights": {k: float(ens.extra["weights"].get(k, 0.0)) for k in base_model_names},
                    "formula": "y_hat = sum(w_i * y_hat_i)",
                }
            )
        elif "coefficients" in ens.extra:
            payload.update(
                {
                    "type": "ridge_stacking",
                    "fit_meta_split": "train",
                    "alpha": float(ens.extra.get("alpha", 1.0)),
                    "intercept": float(ens.extra.get("intercept", 0.0)),
                    "coefficients": {k: float(ens.extra["coefficients"].get(k, 0.0)) for k in base_model_names},
                    "formula": "y_hat = intercept + sum(coef_i * y_hat_i)",
                }
            )

        save_json(models_dir / f"{ens_name}.json", payload)
        with (models_dir / f"{ens_name}.pkl").open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Merge metrics/preds
    all_metrics: Dict[str, Dict[str, Dict[str, float]]] = {**base_metrics}
    all_preds: Dict[str, Dict[str, pd.DataFrame]] = {**base_predictions}
    for ens_name, ens in ensemble_results.items():
        all_metrics[ens_name] = ens.metrics_by_split
        all_preds[ens_name] = ens.predictions_by_split

    metrics_df = build_metrics_long(all_metrics)
    metrics_excel = write_excel(excel_dir / "metrics.xlsx", {"metrics": metrics_df})

    # Predictions wide
    pred_sheets = {
        "train": build_predictions_wide(all_preds, id_col=data.id_name, split="train"),
        "val": build_predictions_wide(all_preds, id_col=data.id_name, split="val"),
        "test": build_predictions_wide(all_preds, id_col=data.id_name, split="test"),
    }
    preds_excel = write_excel(excel_dir / "predictions.xlsx", pred_sheets)

    # Training history
    history_sheets = build_training_history_sheets(histories)
    history_excel = write_excel(excel_dir / "training_history.xlsx", history_sheets)

    # 4) Evaluation plots
    for split in ["train", "val", "test"]:
        plot_metrics_comparison(metrics_df, figures_dir, split=split)

    # Per-model scatter + residual diagnostics (all models including ensembles)
    for model_name in all_preds.keys():
        for split in ["train", "val", "test"]:
            dfp = all_preds[model_name][split]
            y_true = dfp["y_true"].to_numpy()
            y_pred = dfp["y_pred"].to_numpy()
            plot_true_vs_pred(
                y_true,
                y_pred,
                figures_dir / f"{model_name}_{split}_true_vs_pred.png",
                title=f"{model_name} ({split}) - True vs Pred (y=x)",
                max_points=cfg.max_scatter_points,
            )
            plot_residual_diagnostics(
                y_true,
                y_pred,
                figures_dir / f"{model_name}_{split}_residuals.png",
                title=f"{model_name} ({split}) - Residual diagnostics",
                max_points=cfg.max_scatter_points,
            )
            plot_error_cdf(
                y_true,
                y_pred,
                figures_dir / f"{model_name}_{split}_abs_error_cdf.png",
                title=f"{model_name} ({split}) - Absolute Error CDF",
            )
            plot_residual_qq(
                y_true,
                y_pred,
                figures_dir / f"{model_name}_{split}_residual_qq.png",
                title=f"{model_name} ({split}) - Residual Q-Q Plot",
            )

    # 5) OE mechanism export (Excel only)
    for ens_name, ens in ensemble_results.items():
        if "weights" in ens.extra:
            weights = {k: float(ens.extra["weights"][k]) for k in models_selected.keys()}
            weight_power = float(ens.extra.get("weight_power", 2.0))
            base_r2_val = {k: float(ens.extra.get("base_r2_val", {}).get(k, np.nan)) for k in models_selected.keys()}

            mech_rows = []
            for k in models_selected.keys():
                r2 = base_r2_val.get(k, np.nan)
                raw = (max(0.0, float(r2)) ** weight_power) if np.isfinite(r2) else np.nan
                mech_rows.append(
                    {
                        "model": k,
                        "val_R2": float(r2) if np.isfinite(r2) else np.nan,
                        "weight_power": float(weight_power),
                        "raw_weight": float(raw) if np.isfinite(raw) else np.nan,
                        "weight": float(weights.get(k, 0.0)),
                    }
                )
            mech_df = pd.DataFrame(mech_rows).sort_values("weight", ascending=False).reset_index(drop=True)
            meta_df = pd.DataFrame(
                [
                    {"key": "ensemble", "value": ens_name},
                    {"key": "method", "value": "weighted_average"},
                    {"key": "weight_source_split", "value": "val"},
                    {"key": "weight_source_metric", "value": "R2"},
                    {"key": "weight_power", "value": float(weight_power)},
                    {"key": "sum_weights", "value": float(sum(weights.values()))},
                ]
            )
        elif "coefficients" in ens.extra:
            coef = {k: float(ens.extra["coefficients"][k]) for k in models_selected.keys()}
            intercept = float(ens.extra.get("intercept", 0.0))
            alpha = float(ens.extra.get("alpha", 1.0))
            mech_df = (
                pd.DataFrame([{"model": k, "coef": float(v), "abs_coef": abs(float(v))} for k, v in coef.items()])
                .sort_values("abs_coef", ascending=False)
                .reset_index(drop=True)
            )
            meta_df = pd.DataFrame(
                [
                    {"key": "ensemble", "value": ens_name},
                    {"key": "method", "value": "ridge_stacking"},
                    {"key": "fit_meta_split", "value": "train"},
                    {"key": "alpha", "value": float(alpha)},
                    {"key": "fit_intercept", "value": True},
                    {"key": "intercept", "value": float(intercept)},
                    {"key": "formula", "value": "y_hat = intercept + Σ(coef_i * y_hat_i)"},
                ]
            )
        else:
            continue

        write_excel(excel_dir / f"oe_mechanism_{ens_name}.xlsx", {"OE": mech_df, "meta": meta_df})

    print("\n✅ 运行完成")
    print(f"- run_dir: {run_dir}")
    print(f"- metrics: {metrics_excel}")
    print(f"- predictions: {preds_excel}")
    print(f"- history: {history_excel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
