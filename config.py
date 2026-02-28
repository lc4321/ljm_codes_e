from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass(frozen=True)
class ExperimentConfig:
    # Data
    data_path: str = "data/factor1016.xlsx"
    id_column: Optional[str] = None  # None => use first column
    target_column: Optional[str] = None  # None => use last column
    test_size: float = 0.12
    val_size: float = 0.18
    random_state: int = 42

    # Scaling
    scaler_x: str = "standard"  # standard|minmax|robust|maxabs|quantile|power
    scaler_y: str = "standard"

    # Training
    batch_size: int = 32
    epochs: int = 800
    patience: int = 60
    learning_rate: float = 0.008
    weight_decay: float = 1e-4
    noise_std: float = 0.003
    grad_clip: float = 1.0

    # Runtime
    device: str = "auto"  # auto|cpu|cuda
    num_workers: int = 0

    # Project I/O
    run_root: str = "runs"
    run_name: Optional[str] = None

    # Models (4+ base models)
    model_names: Sequence[str] = field(
        default_factory=lambda: (
            "MLP",
            "ResidualMLP",
            "FeatureGateNet",
            "WideDeepNet",
        )
    )

    # Ensembles
    enable_weighted_ensemble: bool = True
    enable_ridge_ensemble: bool = True

    # Plots
    max_scatter_points: int = 2000

    # Execution mode
    quick: bool = False
