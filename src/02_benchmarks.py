"""Benchmarks rolling para previsão da RV_YZ mensal.

Os modelos usam o mesmo painel e as mesmas janelas temporais. O script é
deliberadamente tolerante a dependências opcionais: ``arch`` e ``xgboost``
podem ser instalados no Colab, enquanto o smoke test usa fallbacks simples se
eles não estiverem disponíveis.

Uso:
    python src/02_benchmarks.py --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from utils import rolling_indices

warnings.filterwarnings("ignore")


def _panel_features(panel: pd.DataFrame) -> list[str]:
    excluded = {"date", "unique_id", "block_id", C.TARGET}
    return [c for c in panel.columns if c not in excluded and pd.api.types.is_numeric_dtype(panel[c])]


def _with_lags(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = panel.copy().sort_values("date").reset_index(drop=True)
    lag_cols = []
    for lag in C.LAGS:
        col = f"target_lag_{lag}"
        out[col] = out[C.TARGET].shift(lag)
        lag_cols.append(col)
    return out, lag_cols


def _numeric_matrix(panel: pd.DataFrame, columns: list[str], fit_idx: np.ndarray,
                    all_idx: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if all_idx is None:
        all_idx = np.arange(len(panel))
    raw = panel[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = raw.iloc[fit_idx].median(axis=0).fillna(0.0)
    scale = raw.iloc[fit_idx].std(axis=0, ddof=0).replace(0, 1.0).fillna(1.0)
    x = ((raw - med) / scale).fillna(0.0).to_numpy(dtype=np.float32)
    return x[all_idx], {"median": med.to_numpy(), "scale": scale.to_numpy()}


def _xgb_predict(panel: pd.DataFrame, fit_idx: np.ndarray, test_idx: np.ndarray,
                 feature_cols: list[str], smoke: bool,
                 return_model: bool = False):
    x_all, _ = _numeric_matrix(panel, feature_cols, fit_idx, np.arange(len(panel)))
    y = panel[C.TARGET].to_numpy(dtype=float)
    try:
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=40 if smoke else 250,
            max_depth=2 if smoke else 3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            eval_metric="mae",
            random_state=C.SEED,
            n_jobs=max(1, min(8, (os.cpu_count() or 2) // 2)),
        )
        model.fit(x_all[fit_idx], y[fit_idx])
        pred = model.predict(x_all[test_idx])
    except ImportError:
        # Fallback reproduzível para preparar o pipeline sem xgboost instalado.
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = HistGradientBoostingRegressor(
            max_iter=40 if smoke else 200, learning_rate=0.05, max_leaf_nodes=8,
            l2_regularization=1.0, random_state=C.SEED,
        )
        model.fit(x_all[fit_idx], y[fit_idx])
        pred = model.predict(x_all[test_idx])
    pred = np.maximum(np.asarray(pred, dtype=float), 1e-10)
    return (pred, model) if return_model else pred


class TorchLSTMRegressor:
    """LSTM pequeno e autocontido para o benchmark.

    O modelo usa a sequência de features disponíveis até o bloco de previsão,
    incluindo lags observados de ``y``. A implementação evita depender da
    API de versões específicas do NeuralForecast e funciona em CPU ou CUDA.
    """

    def __init__(self, input_size: int = 12, hidden_size: int = 32,
                 epochs: int = 15, learning_rate: float = 1e-3, seed: int = C.SEED):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.seed = seed
        self.model = None
        self.center = None
        self.scale = None

    def fit(self, x: np.ndarray, y: np.ndarray, positions: np.ndarray) -> "TorchLSTMRegressor":
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise RuntimeError("PyTorch não está instalado; use o runtime GPU do Colab.") from exc

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.center = np.nanmedian(x[positions], axis=0)
        self.center = np.where(np.isfinite(self.center), self.center, 0.0)
        self.scale = np.nanstd(x[positions], axis=0)
        self.scale = np.where(np.isfinite(self.scale) & (self.scale > 1e-8), self.scale, 1.0)
        xs = np.nan_to_num((x - self.center) / self.scale, nan=0.0, posinf=0.0, neginf=0.0)

        sequences, targets = [], []
        pos_set = set(int(i) for i in positions)
        for pos in positions:
            pos = int(pos)
            lo = pos - self.input_size + 1
            if lo < 0 or not all(i in pos_set for i in range(lo, pos + 1)):
                continue
            if np.isfinite(y[pos]) and np.isfinite(xs[lo:pos + 1]).all():
                sequences.append(xs[lo:pos + 1])
                targets.append(y[pos])
        if not sequences:
            raise ValueError("Não há janelas suficientes para treinar o LSTM.")

        X_t = torch.tensor(np.asarray(sequences), dtype=torch.float32, device=self.device)
        y_t = torch.tensor(np.asarray(targets), dtype=torch.float32, device=self.device).view(-1, 1)

        self.model = _build_lstm_net(X_t.shape[-1], self.hidden_size).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.SmoothL1Loss()
        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(self.model(X_t), y_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
        return self

    def predict(self, x: np.ndarray, positions: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LSTM ainda não foi ajustado.")
        import torch

        xs = np.nan_to_num((x - self.center) / self.scale, nan=0.0, posinf=0.0, neginf=0.0)
        seq, valid_pos = [], []
        for pos in positions:
            pos = int(pos)
            lo = pos - self.input_size + 1
            if lo >= 0:
                seq.append(xs[lo:pos + 1])
                valid_pos.append(pos)
        out = np.full(len(positions), np.nan, dtype=float)
        if seq:
            self.model.eval()
            with torch.no_grad():
                p = self.model(torch.tensor(np.asarray(seq), dtype=torch.float32, device=self.device))
            mapping = {p_: float(v) for p_, v in zip(valid_pos, p.detach().cpu().numpy().ravel())}
            out = np.array([mapping.get(int(pos), np.nan) for pos in positions])
        return np.maximum(out, 1e-10)

    def save_artifact(self, path: Path, metadata: dict | None = None) -> None:
        if self.model is None:
            raise RuntimeError("Não é possível salvar um LSTM não ajustado.")
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format_version": 1,
            "model_config": {"input_size": self.input_size, "hidden_size": self.hidden_size,
                             "learning_rate": self.learning_rate, "seed": self.seed,
                             "n_features": int(len(self.center))},
            "center": np.asarray(self.center), "scale": np.asarray(self.scale),
            "state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "metadata": metadata or {},
        }, path)


def _build_lstm_net(n_features: int, hidden_size: int):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden_size, batch_first=True)
            self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1), nn.Softplus())

        def forward(self, z):
            z, _ = self.lstm(z)
            return self.head(z[:, -1, :])

    return Net()


def _lstm_predict(panel: pd.DataFrame, fit_idx: np.ndarray, test_idx: np.ndarray,
                  feature_cols: list[str], smoke: bool,
                  return_model: bool = False):
    x, _ = _numeric_matrix(panel, feature_cols, fit_idx, np.arange(len(panel)))
    y = panel[C.TARGET].to_numpy(dtype=float)
    model = TorchLSTMRegressor(input_size=12, epochs=10 if smoke else 100)
    try:
        model.fit(x, y, fit_idx)
        pred = model.predict(x, test_idx)
        return (pred, model) if return_model else pred
    except (ImportError, RuntimeError, ValueError) as exc:
        warnings.warn(f"LSTM indisponível ({exc}); usando persistência como fallback.")
        pred = np.maximum(y[test_idx - 1], 1e-10)
        return (pred, None) if return_model else pred


def _har_predict(panel: pd.DataFrame, fit_idx: np.ndarray, test_idx: np.ndarray,
                 return_params: bool = False):
    y = panel[C.TARGET].to_numpy(dtype=float)
    z = pd.DataFrame(index=panel.index)
    z["const"] = 1.0
    z["daily"] = pd.Series(y).shift(1)
    z["weekly"] = pd.Series(y).shift(1).rolling(3, min_periods=3).mean()
    z["monthly"] = pd.Series(y).shift(1).rolling(12, min_periods=12).mean()
    X = z.to_numpy(dtype=float)
    valid = np.isfinite(X[fit_idx]).all(axis=1) & np.isfinite(y[fit_idx])
    fit = fit_idx[valid]
    if len(fit) < 10:
        pred = np.maximum(y[test_idx - 1], 1e-10)
        return (pred, None) if return_params else pred
    beta, *_ = np.linalg.lstsq(X[fit], y[fit], rcond=None)
    pred = X[test_idx] @ beta
    bad = ~np.isfinite(pred)
    pred[bad] = y[test_idx[bad] - 1]
    pred = np.maximum(pred, 1e-10)
    return (pred, beta) if return_params else pred


def _garch_params(returns: pd.Series, smoke: bool = False) -> tuple[float, float, float, float]:
    returns = returns.dropna().astype(float)
    if len(returns) < 100:
        var = float(returns.var()) if len(returns) else 1e-4
        return max(var / 10, 1e-12), 0.05, 0.9, var
    try:
        from arch import arch_model

        # Escala de porcentagem melhora a estabilidade numérica do arch.
        res = arch_model(returns * 100.0, mean="Zero", vol="GARCH", p=1, q=1,
                         dist="normal", rescale=False).fit(disp="off", show_warning=False,
                                                            options={"maxiter": 300 if smoke else 1000})
        p = res.params
        omega = max(float(p.get("omega", 0.01)) / 10000.0, 1e-12)
        alpha = float(p.get("alpha[1]", 0.05))
        beta = float(p.get("beta[1]", 0.9))
        if alpha + beta >= 0.999:
            beta = min(beta, 0.998 - alpha)
        last_var = float(res.conditional_volatility.iloc[-1] ** 2) / 10000.0
        return omega, alpha, beta, max(last_var, 1e-12)
    except (ImportError, ValueError, RuntimeError, np.linalg.LinAlgError):
        var = max(float(returns.var()), 1e-8)
        return max(var * 0.05, 1e-12), 0.05, 0.90, var


def _garch_predict(daily: pd.DataFrame, panel: pd.DataFrame, test_idx: np.ndarray,
                   fit_end: pd.Timestamp, smoke: bool = False,
                   return_params: bool = False):
    d = daily.copy().sort_values("date")
    returns = d["ret_spx_log"].dropna()
    # Ajusta os parâmetros até o bloco anterior. Os retornos do bloco atual
    # podem atualizar a variância condicional antes da previsão forward.
    first_pos = int(test_idx[0])
    previous_date = panel.loc[first_pos - 1, "date"] if first_pos > 0 else fit_end
    train_ret = returns.loc[d.loc[returns.index, "date"] <= previous_date]
    omega, alpha, beta, variance = _garch_params(train_ret, smoke=smoke)
    last_date = previous_date
    predictions = []
    for pos in test_idx:
        end_date = panel.loc[int(pos), "date"]
        new = d.loc[(d["date"] > last_date) & (d["date"] <= end_date), "ret_spx_log"].dropna()
        for ret in new.to_numpy(dtype=float):
            variance = omega + alpha * (ret ** 2) + beta * variance
        future_var = []
        state = variance
        for _ in range(C.BLOCK_SIZE):
            state = omega + beta * state
            future_var.append(state)
        predictions.append(float(np.mean(future_var) * 252.0))
        last_date = end_date
    pred = np.maximum(np.asarray(predictions), 1e-10)
    params = {"omega": omega, "alpha": alpha, "beta": beta,
              "last_variance": variance, "fit_end": str(previous_date)}
    return (pred, params) if return_params else pred


def _make_rows(panel: pd.DataFrame, model: str, positions: np.ndarray,
               pred: np.ndarray, window_id: int) -> pd.DataFrame:
    out = pd.DataFrame({
        "model": model,
        "unique_id": panel.loc[positions, "unique_id"].to_numpy(),
        "date": panel.loc[positions, "date"].to_numpy(),
        "y": panel.loc[positions, C.TARGET].to_numpy(dtype=float),
        "y_hat": np.asarray(pred, dtype=float),
        "window_id": window_id,
    })
    if "vix_mean" in panel:
        out["vix_mean"] = panel.loc[positions, "vix_mean"].to_numpy(dtype=float)
    if "vix3m_mean" in panel:
        out["vix3m_mean"] = panel.loc[positions, "vix3m_mean"].to_numpy(dtype=float)
    out["y_hat"] = out["y_hat"].clip(lower=1e-10)
    out["vrp_variance"] = (out["vix_mean"] / 100.0) ** 2 - out["y_hat"] if "vix_mean" in out else np.nan
    return out


def _read_daily_for_garch(path: Path = C.DATA_XLSX) -> pd.DataFrame:
    d = pd.read_excel(path, sheet_name="painel_diario", usecols=["date", "spx_close"])
    d["date"] = pd.to_datetime(d["date"])
    d["spx_close"] = pd.to_numeric(d["spx_close"], errors="coerce")
    d["ret_spx_log"] = np.log(d["spx_close"] / d["spx_close"].shift(1))
    return d


def run(panel_path: Path = C.DATA_PARQUET, output: Path = C.FORECAST_BENCH,
        smoke: bool = False, model_dir: Path = C.BENCHMARK_MODELS_DIR,
        save_models: bool = True) -> pd.DataFrame:
    panel = pd.read_parquet(panel_path).sort_values("date").reset_index(drop=True)
    panel, lag_cols = _with_lags(panel)
    feature_cols = _panel_features(panel)
    # O benchmark VIX usa a média diária agregada, convertida de nível (%) para
    # variância anualizada, mesma unidade do alvo Yang-Zhang.
    iterations = rolling_indices(panel)
    if not iterations:
        raise ValueError("Nenhuma janela rolling válida foi encontrada.")
    daily = _read_daily_for_garch() if "spx_close" in pd.read_excel(C.DATA_XLSX, sheet_name="painel_diario", nrows=1).columns else None
    rows: list[pd.DataFrame] = []
    manifest = []
    if save_models:
        model_dir.mkdir(parents=True, exist_ok=True)

    for window_id, info in enumerate(iterations):
        tr0, tr1 = info["train_idx"]
        va0, va1 = info["val_idx"]
        te0, te1 = info["test_idx"]
        fit_idx = np.arange(tr0, va1, dtype=int)
        test_idx = np.arange(te0, te1, dtype=int)
        y = panel[C.TARGET].to_numpy(dtype=float)

        rows.append(_make_rows(panel, "RW", test_idx, y[test_idx - 1], window_id))

        har_pred, har_beta = _har_predict(panel, fit_idx, test_idx, return_params=True)
        har_frame = _make_rows(panel, "HAR", test_idx, har_pred, window_id)
        rows.append(har_frame)
        if save_models and har_beta is not None:
            har_path = model_dir / f"har_window_{window_id:02d}.json"
            har_path.write_text(json.dumps({"beta": np.asarray(har_beta).tolist(),
                                            "features": ["const", "daily_lag_1", "weekly_lag_3", "monthly_lag_12"]}, indent=2), encoding="utf-8")
            manifest.append({"model": "HAR", "window_id": window_id,
                             "path": str(har_path.relative_to(C.ROOT)), "status": "saved"})

        xgb_pred, xgb_model = _xgb_predict(panel, fit_idx, test_idx, feature_cols, smoke, return_model=True)
        xgb_frame = _make_rows(panel, "XGBoost", test_idx, xgb_pred, window_id)
        rows.append(xgb_frame)
        if save_models:
            try:
                import joblib

                xgb_path = model_dir / f"xgboost_window_{window_id:02d}.joblib"
                joblib.dump(xgb_model, xgb_path)
                manifest.append({"model": "XGBoost", "window_id": window_id,
                                 "path": str(xgb_path.relative_to(C.ROOT)), "status": "saved"})
            except Exception as exc:
                manifest.append({"model": "XGBoost", "window_id": window_id,
                                 "path": None, "status": f"not_saved: {exc}"})

        lstm_pred, lstm_model = _lstm_predict(panel, fit_idx, test_idx, feature_cols, smoke, return_model=True)
        lstm_frame = _make_rows(panel, "LSTM", test_idx, lstm_pred, window_id)
        rows.append(lstm_frame)
        if save_models and lstm_model is not None:
            lstm_path = model_dir / f"lstm_window_{window_id:02d}.pt"
            lstm_model.save_artifact(lstm_path, metadata={"window_id": window_id,
                                                           "feature_columns": feature_cols})
            manifest.append({"model": "LSTM", "window_id": window_id,
                             "path": str(lstm_path.relative_to(C.ROOT)), "status": "saved"})

        if daily is not None:
            garch_pred, garch_params = _garch_predict(daily, panel, test_idx, panel.loc[te0, "date"],
                                                      smoke=smoke, return_params=True)
            rows.append(_make_rows(panel, "GARCH(1,1)", test_idx, garch_pred, window_id))
            if save_models:
                garch_path = model_dir / f"garch_window_{window_id:02d}.json"
                garch_path.write_text(json.dumps(garch_params, indent=2), encoding="utf-8")
                manifest.append({"model": "GARCH(1,1)", "window_id": window_id,
                                 "path": str(garch_path.relative_to(C.ROOT)), "status": "saved"})

        vix = panel.loc[test_idx, "vix_mean"].to_numpy(dtype=float) if "vix_mean" in panel else np.full(len(test_idx), np.nan)
        rows.append(_make_rows(panel, "VIX", test_idx, (vix / 100.0) ** 2, window_id))
        if "vix3m_mean" in panel:
            vix3m = panel.loc[test_idx, "vix3m_mean"].to_numpy(dtype=float)
            rows.append(_make_rows(panel, "VIX3M", test_idx, (vix3m / 100.0) ** 2, window_id))

    forecasts = pd.concat(rows, ignore_index=True)
    forecasts["vol_error"] = (np.sqrt(forecasts["y_hat"]) - np.sqrt(forecasts["y"])) * 100.0
    output.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(output, index=False)
    forecasts.to_csv(output.with_suffix(".csv"), index=False)
    if save_models:
        (model_dir / "model_manifest.json").write_text(
            json.dumps({"models": manifest,
                        "note": "RW, VIX e VIX3M são benchmarks sem pesos treináveis."}, indent=2),
            encoding="utf-8"
        )
    print(f"Previsões salvas em: {output}")
    print(f"Shape: {forecasts.shape}; modelos: {', '.join(forecasts.model.unique())}")
    return forecasts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=C.DATA_PARQUET)
    parser.add_argument("--output", type=Path, default=C.FORECAST_BENCH)
    parser.add_argument("--models-dir", type=Path, default=C.BENCHMARK_MODELS_DIR)
    parser.add_argument("--no-save-models", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.panel, args.output, args.smoke, args.models_dir, save_models=not args.no_save_models)
    return 0


if __name__ == "__main__":
    sys.exit(main())
