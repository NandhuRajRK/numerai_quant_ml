"""MLX-based neural tabular models for Apple Silicon experiments."""

from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def _mlx_modules() -> tuple[Any, Any, Any]:
    """Import MLX modules lazily so the repo works without the extra."""
    mx = importlib.import_module("mlx.core")
    nn = importlib.import_module("mlx.nn")
    optim = importlib.import_module("mlx.optimizers")
    return mx, nn, optim


@dataclass
class MLXTabularMLPConfig:
    """Serializable training configuration for the MLX MLP wrapper."""

    hidden_dims: list[int]
    dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    patience: int
    validation_fraction: float
    random_seed: int


class _TabularMLPNet:
    """Factory wrapper around an MLX nn.Module for tabular regression."""

    def __init__(self, input_dim: int, config: MLXTabularMLPConfig) -> None:
        mx, nn, _ = _mlx_modules()
        self._mx = mx
        self._nn = nn
        self._config = config
        dims = [input_dim, *config.hidden_dims, 1]
        layers: list[Any] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1], strict=False):
            layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ]
            )
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.model = nn.Sequential(*layers)

    def __call__(self, inputs: Any, training: bool = False) -> Any:
        return self.model(inputs)

    def parameters(self) -> Any:
        return self.model.parameters()

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def save_weights(self, path: str) -> None:
        self.model.save_weights(path)

    def load_weights(self, path: str) -> None:
        self.model.load_weights(path)


class MLXTabularMLPRegressor:
    """A small MLX MLP regressor with standardization and early stopping."""

    def __init__(
        self,
        *,
        hidden_dims: tuple[int, ...] = (512, 256, 64),
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 4096,
        max_epochs: int = 30,
        patience: int = 5,
        validation_fraction: float = 0.15,
        random_seed: int = 42,
    ) -> None:
        self.config = MLXTabularMLPConfig(
            hidden_dims=list(hidden_dims),
            dropout=float(dropout),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            batch_size=int(batch_size),
            max_epochs=int(max_epochs),
            patience=int(patience),
            validation_fraction=float(validation_fraction),
            random_seed=int(random_seed),
        )
        self.feature_mean_: np.ndarray | None = None
        self.feature_std_: np.ndarray | None = None
        self.target_mean_: float = 0.0
        self.target_std_: float = 1.0
        self.input_dim_: int | None = None
        self._net: _TabularMLPNet | None = None

    @staticmethod
    def _standardize_features(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        safe_std = np.where(std == 0.0, 1.0, std)
        return ((X - mean) / safe_std).astype(np.float32)

    def _build_net(self, input_dim: int) -> _TabularMLPNet:
        return _TabularMLPNet(input_dim, self.config)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MLXTabularMLPRegressor:
        mx, nn, optim = _mlx_modules()
        rng = np.random.default_rng(self.config.random_seed)

        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32).reshape(-1)
        self.feature_mean_ = X_np.mean(axis=0)
        self.feature_std_ = X_np.std(axis=0)
        self.target_mean_ = float(y_np.mean())
        self.target_std_ = float(y_np.std()) or 1.0
        X_scaled = self._standardize_features(X_np, self.feature_mean_, self.feature_std_)
        y_scaled = ((y_np - self.target_mean_) / self.target_std_).astype(np.float32)

        num_rows = len(X_scaled)
        indices = np.arange(num_rows)
        rng.shuffle(indices)
        valid_size = max(1, int(num_rows * self.config.validation_fraction))
        valid_idx = indices[:valid_size]
        train_idx = indices[valid_size:]
        if len(train_idx) == 0:
            train_idx = valid_idx
            valid_idx = indices[:1]

        X_train = X_scaled[train_idx]
        y_train = y_scaled[train_idx]
        X_valid = X_scaled[valid_idx]
        y_valid = y_scaled[valid_idx]

        self.input_dim_ = X_scaled.shape[1]
        self._net = self._build_net(self.input_dim_)
        optimizer = optim.AdamW(
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        def loss_fn(model: Any, batch_x: Any, batch_y: Any) -> Any:
            predictions = model(batch_x).squeeze(-1)
            return mx.mean((predictions - batch_y) ** 2)

        loss_and_grad_fn = nn.value_and_grad(self._net.model, loss_fn)
        best_loss = float("inf")
        best_weights_path = Path("/tmp") / f"mlx_mlp_best_{id(self)}.safetensors"
        epochs_without_improvement = 0

        for epoch in range(self.config.max_epochs):
            self._net.train()
            shuffled = rng.permutation(len(X_train))
            for start in range(0, len(X_train), self.config.batch_size):
                batch_idx = shuffled[start : start + self.config.batch_size]
                batch_x = mx.array(X_train[batch_idx])
                batch_y = mx.array(y_train[batch_idx])
                loss, grads = loss_and_grad_fn(self._net.model, batch_x, batch_y)
                optimizer.update(self._net.model, grads)
                mx.eval(loss, self._net.model.parameters(), optimizer.state)

            valid_loss = self._evaluate_loss(X_valid, y_valid)
            LOGGER.info(
                "MLX MLP epoch %s/%s | valid_mse=%.6f",
                epoch + 1,
                self.config.max_epochs,
                valid_loss,
            )
            if valid_loss + 1e-6 < best_loss:
                best_loss = valid_loss
                epochs_without_improvement = 0
                self._net.save_weights(str(best_weights_path))
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.config.patience:
                    break

        if best_weights_path.exists():
            self._net.load_weights(str(best_weights_path))
            best_weights_path.unlink(missing_ok=True)

        self._net.eval()
        return self

    def _evaluate_loss(self, X_valid: np.ndarray, y_valid: np.ndarray) -> float:
        mx, _, _ = _mlx_modules()
        if self._net is None:
            raise RuntimeError("Model has not been fit.")
        self._net.eval()
        predictions = self._net(mx.array(X_valid)).squeeze(-1)
        loss = mx.mean((predictions - mx.array(y_valid)) ** 2)
        return float(loss.item())

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        mx, _, _ = _mlx_modules()
        if self._net is None or self.feature_mean_ is None or self.feature_std_ is None:
            raise RuntimeError("Model has not been fit.")
        X_np = np.asarray(X, dtype=np.float32)
        X_scaled = self._standardize_features(X_np, self.feature_mean_, self.feature_std_)
        outputs = self._net(mx.array(X_scaled)).squeeze(-1)
        predictions = np.asarray(outputs, dtype=np.float32) * self.target_std_ + self.target_mean_
        return predictions.astype(float)

    def save_model(self, path: str) -> None:
        if self._net is None or self.feature_mean_ is None or self.feature_std_ is None:
            raise RuntimeError("Model has not been fit.")
        model_dir = Path(path)
        model_dir.mkdir(parents=True, exist_ok=True)
        self._net.save_weights(str(model_dir / "weights.safetensors"))
        np.savez(
            model_dir / "scaler.npz",
            feature_mean=self.feature_mean_,
            feature_std=self.feature_std_,
            target_mean=np.array([self.target_mean_], dtype=np.float32),
            target_std=np.array([self.target_std_], dtype=np.float32),
            input_dim=np.array([self.input_dim_], dtype=np.int32),
        )
        with (model_dir / "config.json").open("w", encoding="utf-8") as file:
            json.dump(self.config.__dict__, file, indent=2)

    @classmethod
    def load_model(cls, path: str) -> MLXTabularMLPRegressor:
        model_dir = Path(path)
        with (model_dir / "config.json").open("r", encoding="utf-8") as file:
            raw_config = json.load(file)
        instance = cls(
            hidden_dims=tuple(raw_config["hidden_dims"]),
            dropout=float(raw_config["dropout"]),
            learning_rate=float(raw_config["learning_rate"]),
            weight_decay=float(raw_config["weight_decay"]),
            batch_size=int(raw_config["batch_size"]),
            max_epochs=int(raw_config["max_epochs"]),
            patience=int(raw_config["patience"]),
            validation_fraction=float(raw_config["validation_fraction"]),
            random_seed=int(raw_config["random_seed"]),
        )
        scaler = np.load(model_dir / "scaler.npz")
        instance.feature_mean_ = scaler["feature_mean"]
        instance.feature_std_ = scaler["feature_std"]
        instance.target_mean_ = float(scaler["target_mean"][0])
        instance.target_std_ = float(scaler["target_std"][0])
        instance.input_dim_ = int(scaler["input_dim"][0])
        instance._net = instance._build_net(instance.input_dim_)
        instance._net.load_weights(str(model_dir / "weights.safetensors"))
        instance._net.eval()
        return instance
