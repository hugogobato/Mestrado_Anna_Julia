"""TSMixerX nativo em PyTorch, busca de HP e rolling 4+1+1.

O painel é irregular em datas de calendário porque cada observação representa
exatamente 22 pregões. Por isso o script usa um índice operacional mensal
regular internamente, mantendo ``date`` original apenas para alinhamento e
relato. A arquitetura implementa os dois misturadores da TSMixerX, mistura
temporal e mistura de features, e aceita CUDA no Colab.

Uso local curto:
    python src/03_tsmixerx.py --trials 5 --max-steps 50 --smoke

No Colab:
    python src/03_tsmixerx.py --trials 200 --max-steps 1000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from utils import filter_collinear, hp_split, rolling_indices

warnings.filterwarnings("ignore")


def _set_seed(seed: int = C.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(max(1, min(8, (os.cpu_count() or 2) // 2)))
    except ImportError:
        pass


def _feature_columns(panel: pd.DataFrame) -> list[str]:
    excluded = {"date", "unique_id", "block_id", C.TARGET}
    return [c for c in panel.columns if c not in excluded and pd.api.types.is_numeric_dtype(panel[c])]


def _matrix(panel: pd.DataFrame, columns: list[str], fit_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = panel[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = raw.iloc[fit_idx].median(axis=0).fillna(0.0).to_numpy(dtype=float)
    scale = raw.iloc[fit_idx].std(axis=0, ddof=0).replace(0, 1.0).fillna(1.0).to_numpy(dtype=float)
    x = raw.to_numpy(dtype=float)
    x = np.nan_to_num((x - med) / scale, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return x, med, scale


def _add_target_lags(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy().sort_values("date").reset_index(drop=True)
    for lag in C.LAGS:
        out[f"target_lag_{lag}"] = out[C.TARGET].shift(lag)
    return out


def _build_tsmixer_net(n_steps: int, n_features: int, n_block: int,
                       ff_dim: int, dropout: float):
    """Cria a rede a partir da configuração, sem depender de pickle de classe local."""
    import torch
    from torch import nn

    class MixerBlock(nn.Module):
        def __init__(self, steps: int, features: int, width: int, p: float):
            super().__init__()
            self.norm_t = nn.LayerNorm(features)
            self.time = nn.Sequential(
                nn.Linear(steps, width), nn.GELU(), nn.Dropout(p), nn.Linear(width, steps)
            )
            self.norm_f = nn.LayerNorm(features)
            self.feature = nn.Sequential(
                nn.Linear(features, width), nn.GELU(), nn.Dropout(p), nn.Linear(width, features)
            )

        def forward(self, z):
            z = z + self.time(self.norm_t(z).transpose(1, 2)).transpose(1, 2)
            z = z + self.feature(self.norm_f(z))
            return z

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.Sequential(*[
                MixerBlock(n_steps, n_features, ff_dim, dropout) for _ in range(n_block)
            ])
            self.head = nn.Sequential(
                nn.LayerNorm(n_features), nn.Flatten(), nn.Linear(n_steps * n_features, 1)
            )

        def forward(self, z):
            return self.head(self.blocks(z))

    return Net()


def select_features(panel: pd.DataFrame, train_idx: np.ndarray, n_features: int | str,
                    candidates: list[str] | None = None) -> list[str]:
    """Pré-filtro de colinearidade seguido de SelectKBest no treino."""
    if candidates is None:
        candidates = _feature_columns(panel)
    # Reduz apenas exógenas; os cinco lags do alvo permanecem disponíveis.
    lag_cols = [c for c in candidates if c.startswith("target_lag_")]
    exog = [c for c in candidates if c not in lag_cols]
    exog_keep = filter_collinear(panel.loc[train_idx, exog], corr_threshold=0.99)
    candidates = exog_keep + lag_cols
    if n_features == "all":
        return candidates

    k = min(int(n_features), len(candidates))
    from sklearn.feature_selection import SelectKBest, f_regression

    x, _, _ = _matrix(panel, candidates, train_idx)
    y = panel[C.TARGET].to_numpy(dtype=float)
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(x[train_idx], y[train_idx])
    return [c for c, keep in zip(candidates, selector.get_support()) if keep]


class TSMixerXRegressor:
    """Implementação compacta da TSMixerX para uma série e exógenas."""

    def __init__(self, input_size: int, n_block: int, ff_dim: int, dropout: float,
                 learning_rate: float, max_steps: int, scaler_type: str = "robust",
                 revin: bool = True, loss: str = "huber", batch_size: int = 32,
                 early_stop_patience_steps: int = 10, val_check_steps: int = 20,
                 valid_loss: str = "qlike", seed: int = C.SEED):
        self.input_size = int(input_size)
        self.n_block = int(n_block)
        self.ff_dim = int(ff_dim)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.max_steps = int(max_steps)
        self.scaler_type = scaler_type
        self.revin = bool(revin)
        self.loss_name = loss
        self.batch_size = int(batch_size)
        self.early_stop_patience_steps = int(early_stop_patience_steps)
        self.val_check_steps = int(val_check_steps)
        self.valid_loss_name = valid_loss
        self.seed = seed
        self.model = None

    def _x_scale(self, x: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        center = np.zeros(x.shape[1], dtype=float)
        scale = np.ones(x.shape[1], dtype=float)
        if self.scaler_type == "standard":
            center = np.nanmean(x[positions], axis=0)
            scale = np.nanstd(x[positions], axis=0)
        elif self.scaler_type == "robust":
            center = np.nanmedian(x[positions], axis=0)
            scale = np.nanpercentile(x[positions], 75, axis=0) - np.nanpercentile(x[positions], 25, axis=0)
        center = np.where(np.isfinite(center), center, 0.0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
        return np.nan_to_num((x - center) / scale, nan=0.0, posinf=0.0, neginf=0.0), center, scale

    def _y_scale(self, y: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, float, float]:
        if not self.revin:
            return y, 0.0, 1.0
        if self.scaler_type == "robust":
            center = float(np.nanmedian(y[positions]))
            scale = float(np.nanpercentile(y[positions], 75) - np.nanpercentile(y[positions], 25))
        else:
            center = float(np.nanmean(y[positions]))
            scale = float(np.nanstd(y[positions]))
        scale = scale if np.isfinite(scale) and scale > 1e-8 else 1.0
        return (y - center) / scale, center, scale

    def _sequences(self, x: np.ndarray, y: np.ndarray, positions: np.ndarray,
                   context_positions: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        seq, targets = [], []
        if context_positions is None:
            context_positions = positions
        pos_set = set(int(i) for i in context_positions)
        for pos in positions:
            pos = int(pos)
            lo = pos - self.input_size + 1
            if lo < 0 or not all(i in pos_set for i in range(lo, pos + 1)):
                continue
            if np.isfinite(y[pos]) and np.isfinite(x[lo:pos + 1]).all():
                seq.append(x[lo:pos + 1])
                targets.append(y[pos])
        return np.asarray(seq, dtype=np.float32), np.asarray(targets, dtype=np.float32)

    def fit(self, x: np.ndarray, y: np.ndarray, positions: np.ndarray,
            validation_positions: np.ndarray | None = None) -> "TSMixerXRegressor":
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise RuntimeError("PyTorch não está instalado. No Colab, instale neuralforecast/torch.") from exc

        _set_seed(self.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        raw_x = x.astype(float)
        x, self.x_center, self.x_scale = self._x_scale(raw_x, positions)
        y_scaled, self.y_center, self.y_scale = self._y_scale(y.astype(float), positions)
        seq, target = self._sequences(x, y_scaled, positions)
        if len(seq) == 0:
            raise ValueError("Não há janelas válidas para treinar TSMixerX.")

        X = torch.tensor(seq, dtype=torch.float32, device=self.device)
        Y = torch.tensor(target, dtype=torch.float32, device=self.device).view(-1, 1)
        val_seq = val_target = None
        if validation_positions is not None and len(validation_positions):
            context = np.arange(0, int(np.max(validation_positions)) + 1, dtype=int)
            val_seq, val_target = self._sequences(x, y_scaled, validation_positions, context)
            if len(val_seq):
                val_X = torch.tensor(val_seq, dtype=torch.float32, device=self.device)
                val_Y = torch.tensor(val_target, dtype=torch.float32, device=self.device).view(-1, 1)
            else:
                val_X = val_Y = None
        n_steps, n_features = X.shape[1], X.shape[2]

        self.model = _build_tsmixer_net(n_steps, n_features, self.n_block, self.ff_dim, self.dropout).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        if self.loss_name == "mse":
            loss_fn = nn.MSELoss()
        elif self.loss_name == "mae":
            loss_fn = nn.L1Loss()
        else:
            loss_fn = nn.SmoothL1Loss()
        self.model.train()
        batch_size = max(1, min(self.batch_size, len(X)))
        best_state = None
        best_valid = np.inf
        bad_checks = 0
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed)
        for step in range(1, self.max_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            batch_idx = torch.randperm(len(X), generator=generator, device=self.device)[:batch_size]
            loss = loss_fn(self.model(X[batch_idx]), Y[batch_idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            if val_X is not None and step % self.val_check_steps == 0:
                self.model.eval()
                with torch.no_grad():
                    val_pred_scaled = self.model(val_X).squeeze(-1).detach().cpu().numpy()
                val_pred = val_pred_scaled * self.y_scale + self.y_center
                val_true = val_Y.squeeze(-1).detach().cpu().numpy() * self.y_scale + self.y_center
                if self.valid_loss_name.lower() == "qlike":
                    ratio = np.maximum(val_true, 1e-12) / np.maximum(val_pred, 1e-12)
                    valid_score = float(np.mean(ratio - np.log(ratio) - 1.0))
                elif self.valid_loss_name.lower() == "mae":
                    valid_score = float(np.mean(np.abs(val_true - val_pred)))
                else:
                    valid_score = float(np.mean((val_true - val_pred) ** 2))
                if valid_score < best_valid - 1e-8:
                    import copy
                    best_valid = valid_score
                    best_state = copy.deepcopy(self.model.state_dict())
                    bad_checks = 0
                else:
                    bad_checks += 1
                self.model.train()
                if self.early_stop_patience_steps >= 0 and bad_checks >= self.early_stop_patience_steps:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, x: np.ndarray, positions: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("TSMixerX ainda não foi ajustado.")
        import torch

        xs = np.nan_to_num((x - self.x_center) / self.x_scale, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        seq, valid = [], []
        for pos in positions:
            pos = int(pos)
            lo = pos - self.input_size + 1
            if lo >= 0:
                seq.append(xs[lo:pos + 1])
                valid.append(pos)
        pred = np.full(len(positions), np.nan, dtype=float)
        if seq:
            self.model.eval()
            with torch.no_grad():
                p = self.model(torch.tensor(np.asarray(seq), dtype=torch.float32, device=self.device))
            p = p.detach().cpu().numpy().ravel() * self.y_scale + self.y_center
            mapping = {pos: float(value) for pos, value in zip(valid, p)}
            pred = np.array([mapping.get(int(pos), np.nan) for pos in positions])
        return np.maximum(pred, 1e-10)

    def save_artifact(self, path: Path, metadata: dict | None = None) -> None:
        """Salva pesos e tudo que é necessário para recarregar o modelo."""
        if self.model is None:
            raise RuntimeError("Não é possível salvar um TSMixerX não ajustado.")
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state_dict = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        payload = {
            "format_version": 1,
            "model_config": {
                "input_size": self.input_size,
                "n_block": self.n_block,
                "ff_dim": self.ff_dim,
                "dropout": self.dropout,
                "learning_rate": self.learning_rate,
                "max_steps": self.max_steps,
                "scaler_type": self.scaler_type,
                "revin": self.revin,
                "loss": self.loss_name,
                "batch_size": self.batch_size,
                "early_stop_patience_steps": self.early_stop_patience_steps,
                "val_check_steps": self.val_check_steps,
                "valid_loss": self.valid_loss_name,
                "seed": self.seed,
                "n_features": int(self.x_scale.shape[0]),
            },
            "x_center": np.asarray(self.x_center),
            "x_scale": np.asarray(self.x_scale),
            "y_center": float(self.y_center),
            "y_scale": float(self.y_scale),
            "state_dict": state_dict,
            "metadata": metadata or {},
        }
        torch.save(payload, path)

    @classmethod
    def load_artifact(cls, path: Path, map_location: str = "cpu") -> "TSMixerXRegressor":
        """Recarrega um artefato salvo por :meth:`save_artifact`."""
        import torch

        payload = torch.load(path, map_location=map_location, weights_only=False)
        cfg = dict(payload["model_config"])
        cfg.pop("n_features", None)
        model = cls(**cfg)
        model.x_center = np.asarray(payload["x_center"])
        model.x_scale = np.asarray(payload["x_scale"])
        model.y_center = float(payload["y_center"])
        model.y_scale = float(payload["y_scale"])
        model.device = torch.device(map_location)
        model.model = _build_tsmixer_net(
            model.input_size, len(model.x_center), model.n_block, model.ff_dim, model.dropout
        ).to(model.device)
        model.model.load_state_dict(payload["state_dict"])
        model.model.eval()
        return model


def _trial_config(trial) -> dict:
    return {
        "input_size": trial.suggest_categorical("input_size", C.LAGS),
        "n_block": trial.suggest_categorical("n_block", [2, 3, 4, 6]),
        "ff_dim": trial.suggest_categorical("ff_dim", [32, 64, 128, 256]),
        "dropout": trial.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.3]),
        "learning_rate": trial.suggest_categorical("learning_rate", [1e-4, 3e-4, 1e-3, 3e-3]),
        "scaler_type": trial.suggest_categorical("scaler_type", ["identity", "robust", "standard"]),
        "loss": trial.suggest_categorical("loss", ["mae", "mse", "huber"]),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "early_stop_patience_steps": 10,
        "val_check_steps": 20,
        "valid_loss": "qlike",
        "n_features": trial.suggest_categorical("n_features", [30, 50, 70, "all"]),
        "revin": trial.suggest_categorical("revin", [True, False]),
    }


def _objective_factory(panel: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray,
                       candidates: list[str], max_steps: int):
    y = panel[C.TARGET].to_numpy(dtype=float)

    def objective(trial):
        config = _trial_config(trial)
        cols = select_features(panel, train_idx, config["n_features"], candidates)
        x, _, _ = _matrix(panel, cols, train_idx)
        model = TSMixerXRegressor(**{k: v for k, v in config.items() if k != "n_features"},
                                  max_steps=max_steps)
        model.fit(x, y, train_idx, validation_positions=val_idx)
        pred = model.predict(x, val_idx)
        valid = np.isfinite(pred) & np.isfinite(y[val_idx])
        if not valid.any():
            return 1e9
        yt, yp = y[val_idx][valid], pred[valid]
        # QLIKE é avaliada em variância positiva, com pequeno epsilon.
        ratio = np.maximum(yt, 1e-12) / np.maximum(yp, 1e-12)
        score = float(np.mean(ratio - np.log(ratio) - 1.0))
        trial.set_user_attr("selected_features", cols)
        return score

    return objective


def search_hp(panel: pd.DataFrame, trials: int, max_steps: int, smoke: bool = False) -> tuple[dict, list[str]]:
    train, val = hp_split(panel)
    train_idx = train.index.to_numpy(dtype=int)
    val_idx = val.index.to_numpy(dtype=int)
    candidates = _feature_columns(panel)
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Split de HP vazio.")

    try:
        import optuna

        C.HP_SEARCH_DIR.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{(C.HP_SEARCH_DIR / 'tsmixerx.db').resolve()}"
        study = optuna.create_study(direction="minimize", study_name="tsmixerx",
                                    storage=storage, load_if_exists=True,
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1))
        study.optimize(_objective_factory(panel, train_idx, val_idx, candidates, max_steps),
                       n_trials=trials, show_progress_bar=False)
        best = dict(study.best_trial.params)
    except ImportError:
        warnings.warn("Optuna não instalado; usando configuração inicial determinística.")
        best = {"input_size": 12, "n_block": 3, "ff_dim": 64, "dropout": 0.1,
                "learning_rate": 1e-3, "scaler_type": "robust", "loss": "huber",
                "batch_size": 32, "early_stop_patience_steps": 10,
                "val_check_steps": 20, "valid_loss": "qlike",
                "n_features": "all", "revin": True}

    selected = select_features(panel, train_idx, best.get("n_features", "all"), candidates)
    payload = {
        "best_params": best,
        "selected_features": selected,
        "hp_train_end": str(train.date.max()),
        "hp_val_end": str(val.date.max()),
        "hp_validation_scheme": "fixed chronological 8-year train / 2-year validation; no internal rolling 4+1+1",
    }
    C.HP_SEARCH_DIR.mkdir(parents=True, exist_ok=True)
    (C.HP_SEARCH_DIR / "best_config.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Melhor configuração HP: {best}")
    print(f"Features selecionadas: {len(selected)}")
    return best, selected


def run(panel_path: Path = C.DATA_PARQUET, output: Path = C.FORECAST_TSMIXERX,
        trials: int = C.N_TRIALS_DEFAULT, max_steps: int = C.MAX_STEPS_DEFAULT,
        smoke: bool = False, model_dir: Path = C.TSMIXERX_MODELS_DIR,
        save_models: bool = True) -> pd.DataFrame:
    panel = pd.read_parquet(panel_path).sort_values("date").reset_index(drop=True)
    panel = _add_target_lags(panel)
    if smoke:
        trials = min(trials, 3)
        max_steps = min(max_steps, 30)
    best, selected = search_hp(panel, trials, max_steps, smoke=smoke)
    model_params = {k: v for k, v in best.items() if k != "n_features"}
    rows = []
    manifest = []
    if save_models:
        model_dir.mkdir(parents=True, exist_ok=True)
    for window_id, info in enumerate(rolling_indices(panel)):
        tr0, tr1 = info["train_idx"]
        va0, va1 = info["val_idx"]
        te0, te1 = info["test_idx"]
        fit_idx = np.arange(tr0, tr1, dtype=int)
        validation_idx = np.arange(va0, va1, dtype=int)
        test_idx = np.arange(te0, te1, dtype=int)
        x, _, _ = _matrix(panel, selected, fit_idx)
        y = panel[C.TARGET].to_numpy(dtype=float)
        params = dict(model_params)
        params["max_steps"] = max_steps
        model = TSMixerXRegressor(**params)
        artifact_path = None
        try:
            model.fit(x, y, fit_idx, validation_positions=validation_idx)
            pred = model.predict(x, test_idx)
            if save_models:
                artifact = model_dir / f"tsmixerx_window_{window_id:02d}.pt"
                model.save_artifact(artifact, metadata={
                    "window_id": window_id,
                    "train_start": str(panel.loc[tr0, "date"]),
                    "train_end": str(panel.loc[va1 - 1, "date"]),
                    "test_start": str(panel.loc[te0, "date"]),
                    "test_end": str(panel.loc[te1 - 1, "date"]),
                    "selected_features": selected,
                })
                artifact_path = str(artifact.relative_to(C.ROOT)) if artifact.is_relative_to(C.ROOT) else str(artifact)
                manifest.append({"window_id": window_id, "path": artifact_path, "status": "saved"})
        except (ImportError, RuntimeError, ValueError) as exc:
            warnings.warn(f"TSMixerX indisponível na janela {window_id}: {exc}")
            pred = np.maximum(y[test_idx - 1], 1e-10)
            manifest.append({"window_id": window_id, "path": None, "status": f"fallback: {exc}"})
        frame = pd.DataFrame({
            "model": "TSMixerX",
            "unique_id": panel.loc[test_idx, "unique_id"].to_numpy(),
            "date": panel.loc[test_idx, "date"].to_numpy(),
            "y": y[test_idx],
            "y_hat": np.maximum(pred, 1e-10),
            "window_id": window_id,
        })
        frame["artifact_path"] = artifact_path
        rows.append(frame)

    forecasts = pd.concat(rows, ignore_index=True)
    # Alinhamento explícito por data, sem depender do índice criado pelo concat.
    vix_by_date = panel.set_index("date")["vix_mean"] if "vix_mean" in panel else pd.Series(dtype=float)
    forecasts["vix_mean"] = forecasts["date"].map(vix_by_date)
    forecasts["vrp_variance"] = (forecasts["vix_mean"] / 100.0) ** 2 - forecasts["y_hat"]
    forecasts["vol_error"] = (np.sqrt(forecasts["y_hat"]) - np.sqrt(forecasts["y"])) * 100.0
    output.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(output, index=False)
    forecasts.to_csv(output.with_suffix(".csv"), index=False)
    if save_models:
        (model_dir / "model_manifest.json").write_text(
            json.dumps({"model": "TSMixerX", "artifacts": manifest}, indent=2), encoding="utf-8"
        )
    print(f"Previsões TSMixerX salvas em: {output}; shape={forecasts.shape}")
    return forecasts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=C.DATA_PARQUET)
    parser.add_argument("--output", type=Path, default=C.FORECAST_TSMIXERX)
    parser.add_argument("--trials", type=int, default=C.N_TRIALS_DEFAULT)
    parser.add_argument("--max-steps", type=int, default=C.MAX_STEPS_DEFAULT)
    parser.add_argument("--models-dir", type=Path, default=C.TSMIXERX_MODELS_DIR)
    parser.add_argument("--no-save-models", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.panel, args.output, args.trials, args.max_steps, args.smoke,
        args.models_dir, save_models=not args.no_save_models)
    return 0


if __name__ == "__main__":
    sys.exit(main())
