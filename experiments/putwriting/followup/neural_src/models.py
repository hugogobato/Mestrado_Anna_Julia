"""Modelos neurais diretos e retomáveis para os 22 horizontes."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from utils import qlike_by_horizon, set_seed


def _build_tsmixer(input_size: int, n_features: int, horizon: int,
                   n_block: int, ff_dim: int, dropout: float):
    import torch
    from torch import nn

    class MixerBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.time_norm = nn.LayerNorm(input_size)
            self.time_mlp = nn.Sequential(
                nn.Linear(input_size, ff_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(ff_dim, input_size),
            )
            self.feature_norm = nn.LayerNorm(n_features)
            self.feature_mlp = nn.Sequential(
                nn.Linear(n_features, ff_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(ff_dim, n_features),
            )

        def forward(self, x):
            z = x.transpose(1, 2)
            z = z + self.time_mlp(self.time_norm(z))
            z = z.transpose(1, 2)
            return z + self.feature_mlp(self.feature_norm(z))

    class DirectTSMixer(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.Sequential(*[MixerBlock() for _ in range(n_block)])
            self.head = nn.Sequential(
                nn.LayerNorm(n_features), nn.Flatten(),
                nn.Linear(input_size * n_features, horizon),
            )

        def forward(self, x):
            return self.head(self.blocks(x))

    return DirectTSMixer()


def _build_lstm(n_features: int, horizon: int, hidden_size: int, dropout: float):
    import torch
    from torch import nn

    class DirectLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden_size, batch_first=True,
                                num_layers=1, dropout=0.0)
            self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Dropout(dropout),
                                      nn.Linear(hidden_size, horizon))

        def forward(self, x):
            z, _ = self.lstm(x)
            return self.head(z[:, -1])

    return DirectLSTM()


class DirectVolatilityRegressor:
    """Treinador PyTorch com log-target, early stopping e histórico."""

    def __init__(self, model_type: str = "tsmixerx", input_size: int = 66,
                 n_block: int = 3, ff_dim: int = 64, hidden_size: int = 64,
                 dropout: float = 0.1, learning_rate: float = 1e-3,
                 batch_size: int = 64, max_steps: int = 1000,
                 loss: str = "huber", val_check_steps: int = 20,
                 early_stop_patience: int = 10, seed: int = C.SEED,
                 device: str | None = None):
        self.model_type = model_type.lower()
        self.input_size = int(input_size)
        self.n_block = int(n_block)
        self.ff_dim = int(ff_dim)
        self.hidden_size = int(hidden_size)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.max_steps = int(max_steps)
        self.loss_name = loss
        self.val_check_steps = int(val_check_steps)
        self.early_stop_patience = int(early_stop_patience)
        self.seed = int(seed)
        self.requested_device = device
        self.model = None
        self.history: list[dict] = []

    def _build(self, n_features: int):
        if self.model_type == "lstm":
            return _build_lstm(n_features, C.HORIZON, self.hidden_size, self.dropout)
        return _build_tsmixer(self.input_size, n_features, C.HORIZON,
                              self.n_block, self.ff_dim, self.dropout)

    def _fit_scalers(self, x: np.ndarray, y: np.ndarray) -> None:
        flat = x.reshape(-1, x.shape[-1]).astype(float)
        self.x_center = np.nanmedian(flat, axis=0)
        self.x_center = np.where(np.isfinite(self.x_center), self.x_center, 0.0)
        q75 = np.nanpercentile(flat, 75, axis=0)
        q25 = np.nanpercentile(flat, 25, axis=0)
        self.x_scale = q75 - q25
        self.x_scale = np.where(np.isfinite(self.x_scale) & (self.x_scale > 1e-8), self.x_scale, 1.0)

        log_y = np.log(np.maximum(y.astype(float), C.EPS))
        self.y_center = np.mean(log_y, axis=0)
        self.y_scale = np.std(log_y, axis=0)
        self.y_scale = np.where(np.isfinite(self.y_scale) & (self.y_scale > 1e-8), self.y_scale, 1.0)

    def _scale_x(self, x: np.ndarray) -> np.ndarray:
        return np.nan_to_num((x - self.x_center) / self.x_scale,
                             nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _scale_y(self, y: np.ndarray) -> np.ndarray:
        return ((np.log(np.maximum(y, C.EPS)) - self.y_center) / self.y_scale).astype(np.float32)

    def _inverse_y(self, y_scaled: np.ndarray) -> np.ndarray:
        return np.exp(np.clip(y_scaled * self.y_scale + self.y_center, -30, 10))

    def fit(self, train_x: np.ndarray, train_y: np.ndarray,
            val_x: np.ndarray, val_y: np.ndarray, trial=None) -> "DirectVolatilityRegressor":
        import torch
        from torch import nn

        set_seed(self.seed)
        if self.requested_device:
            self.device = torch.device(self.requested_device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fit_scalers(train_x, train_y)
        tx = torch.tensor(self._scale_x(train_x), device=self.device)
        ty = torch.tensor(self._scale_y(train_y), device=self.device)
        vx = torch.tensor(self._scale_x(val_x), device=self.device)
        self.model = self._build(train_x.shape[-1]).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        if self.loss_name == "mse":
            loss_fn = nn.MSELoss()
        elif self.loss_name == "mae":
            loss_fn = nn.L1Loss()
        else:
            loss_fn = nn.SmoothL1Loss()

        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed)
        best_state, best_score, bad_checks = None, np.inf, 0
        self.history = []
        batch_size = max(1, min(self.batch_size, len(tx)))
        for step in range(1, self.max_steps + 1):
            self.model.train()
            idx = torch.randint(0, len(tx), (batch_size,), generator=generator, device=self.device)
            optimizer.zero_grad(set_to_none=True)
            train_loss = loss_fn(self.model(tx[idx]), ty[idx])
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            if step % self.val_check_steps != 0 and step != self.max_steps:
                continue
            self.model.eval()
            with torch.no_grad():
                scaled = self.model(vx).detach().cpu().numpy()
            val_pred = self._inverse_y(scaled)
            qlike_h = qlike_by_horizon(val_y, val_pred)
            val_score = float(np.mean(qlike_h))
            val_mse = float(np.mean((val_y - val_pred) ** 2))
            improved = val_score < best_score - 1e-8
            if improved:
                best_score = val_score
                best_state = copy.deepcopy(self.model.state_dict())
                bad_checks = 0
            else:
                bad_checks += 1
            row = {
                "step": step,
                "train_loss": float(train_loss.detach().cpu()),
                "validation_qlike": val_score,
                "validation_qlike_h22": float(qlike_h[-1]),
                "validation_mse": val_mse,
                "best_so_far": best_score,
                "is_best": improved,
                "bad_checks": bad_checks,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "stopped_early": False,
            }
            self.history.append(row)
            if trial is not None:
                trial.report(val_score, step=step)
                if trial.should_prune():
                    import optuna

                    raise optuna.TrialPruned(f"Pruned at step {step}")
            if bad_checks >= self.early_stop_patience:
                self.history[-1]["stopped_early"] = True
                break
        if best_state is None:
            raise RuntimeError("Nenhum checkpoint de validação foi produzido.")
        self.model.load_state_dict(best_state)
        self.best_validation_qlike = best_score
        return self

    def predict(self, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Modelo não ajustado.")
        import torch

        values = torch.tensor(self._scale_x(x), device=self.device)
        chunks = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(values), batch_size):
                chunks.append(self.model(values[start:start + batch_size]).detach().cpu().numpy())
        return np.maximum(self._inverse_y(np.concatenate(chunks)), C.EPS)

    def history_frame(self, **metadata) -> pd.DataFrame:
        out = pd.DataFrame(self.history)
        for key, value in metadata.items():
            out[key] = value
        return out

    def config_dict(self) -> dict:
        return {
            "model_type": self.model_type,
            "input_size": self.input_size,
            "n_block": self.n_block,
            "ff_dim": self.ff_dim,
            "hidden_size": self.hidden_size,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_steps": self.max_steps,
            "loss": self.loss_name,
            "val_check_steps": self.val_check_steps,
            "early_stop_patience": self.early_stop_patience,
            "seed": self.seed,
        }

    def save(self, path: Path, features: list[str], metadata: dict | None = None) -> None:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format_version": 2,
            "config": self.config_dict(),
            "features": features,
            "x_center": self.x_center,
            "x_scale": self.x_scale,
            "y_center": self.y_center,
            "y_scale": self.y_scale,
            "state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "history": self.history,
            "metadata": metadata or {},
        }, path)

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> tuple["DirectVolatilityRegressor", dict]:
        import torch

        map_location = device or ("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.load(path, map_location=map_location, weights_only=False)
        config = dict(payload["config"])
        config["device"] = map_location
        model = cls(**config)
        model.x_center = np.asarray(payload["x_center"])
        model.x_scale = np.asarray(payload["x_scale"])
        model.y_center = np.asarray(payload["y_center"])
        model.y_scale = np.asarray(payload["y_scale"])
        model.device = torch.device(map_location)
        model.model = model._build(len(payload["features"])).to(model.device)
        model.model.load_state_dict(payload["state_dict"])
        model.model.eval()
        model.history = payload.get("history", [])
        return model, payload

