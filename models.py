from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: Tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.10,
    ):
        super().__init__()
        layers = []
        dim = input_dim
        for h in hidden:
            layers.extend(
                [
                    nn.Linear(dim, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                ]
            )
            dim = h
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.10):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=dropout)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.act(out)
        return out


class ResidualMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        width: int = 128,
        n_blocks: int = 4,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(width, dropout=dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.Linear(width, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1),
        )
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x).squeeze(-1)


class FeatureGateNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        gate_hidden: int = 64,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, input_dim),
            nn.Sigmoid(),
        )
        self.predictor = MLP(input_dim=input_dim, hidden=(128, 64, 32), dropout=dropout)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gate(x)
        x_gated = x * g
        return self.predictor(x_gated)


class WideDeepNet(nn.Module):
    """
      y = Linear(x) + MLP(x)
    """

    def __init__(self, input_dim: int, dropout: float = 0.10):
        super().__init__()
        self.wide = nn.Linear(input_dim, 1)
        self.deep = MLP(input_dim=input_dim, hidden=(128, 64, 32), dropout=dropout)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wide_out = self.wide(x).squeeze(-1)
        deep_out = self.deep(x)
        return wide_out + deep_out


def build_models(input_dim: int) -> Dict[str, nn.Module]:
    return {
        "MLP": MLP(input_dim=input_dim),
        "ResidualMLP": ResidualMLP(input_dim=input_dim),
        "FeatureGateNet": FeatureGateNet(input_dim=input_dim),
        "WideDeepNet": WideDeepNet(input_dim=input_dim),
    }

