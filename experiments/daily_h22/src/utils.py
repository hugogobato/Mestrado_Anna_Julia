"""Alvos, splits, matrizes e métricas do experimento diário H=22."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

import config as C


def ensure_directories() -> None:
    for path in [C.DATA_DIR, C.RESULTS_DIR, C.FORECAST_DIR, C.INPUTS_DIR,
                 C.METRICS_DIR, C.FIGURES_DIR, C.MODELS_DIR, C.HP_DIR,
                 C.HISTORY_DIR, C.EXPLAIN_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = C.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def compute_yang_zhang(ohlc: pd.DataFrame, window: int = C.RV_WINDOW,
                       annualization: int = C.ANNUALIZATION) -> pd.Series:
    """Yang-Zhang padrão em janela móvel, retornando variância anualizada.

    A implementação combina variância overnight, variância open-to-close e o
    estimador Rogers-Satchell. O peso é
    ``k=0.34/(1.34+(n+1)/(n-1))``.
    """
    required = ["spx_open", "spx_high", "spx_low", "spx_close"]
    missing = [c for c in required if c not in ohlc]
    if missing:
        raise KeyError(f"OHLC ausente: {missing}")
    if window < 2:
        raise ValueError("Yang-Zhang requer window >= 2.")

    p = ohlc[required].apply(pd.to_numeric, errors="coerce").astype(float)
    valid = (p > 0).all(axis=1)
    p = p.where(valid)

    overnight = np.log(p["spx_open"] / p["spx_close"].shift(1))
    open_close = np.log(p["spx_close"] / p["spx_open"])
    high_open = np.log(p["spx_high"] / p["spx_open"])
    low_open = np.log(p["spx_low"] / p["spx_open"])
    rs_daily = high_open * (high_open - open_close) + low_open * (low_open - open_close)

    var_overnight = overnight.rolling(window, min_periods=window).var(ddof=1)
    var_open_close = open_close.rolling(window, min_periods=window).var(ddof=1)
    var_rs = rs_daily.rolling(window, min_periods=window).mean()
    k = 0.34 / (1.34 + (window + 1.0) / (window - 1.0))
    yz = var_overnight + k * var_open_close + (1.0 - k) * var_rs
    yz = (yz * annualization).where(np.isfinite(yz) & (yz > 0))
    return yz.rename(C.TARGET)


def add_forward_close_variance(frame: pd.DataFrame, horizon: int = C.HORIZON) -> pd.DataFrame:
    """Adiciona diagnóstico aditivo de variância close-to-close futura."""
    out = frame.copy()
    r = np.log(pd.to_numeric(out["spx_close"], errors="coerce") /
               pd.to_numeric(out["spx_close"], errors="coerce").shift(1))
    future_sum = sum(r.shift(-h).pow(2) for h in range(1, horizon + 1))
    variance = future_sum * C.ANNUALIZATION / horizon
    out["close_variance_forward_22"] = variance
    out["close_volatility_forward_22"] = np.sqrt(variance.clip(lower=0))
    return out


def valid_origin_mask(panel: pd.DataFrame, horizon: int = C.HORIZON) -> np.ndarray:
    y = panel[C.TARGET].to_numpy(float)
    valid = np.isfinite(y)
    mask = np.zeros(len(panel), dtype=bool)
    for pos in range(len(panel) - horizon):
        mask[pos] = valid[pos] and valid[pos + 1:pos + horizon + 1].all()
    return mask


def target_matrix(panel: pd.DataFrame, origins: np.ndarray,
                  horizon: int = C.HORIZON) -> np.ndarray:
    y = panel[C.TARGET].to_numpy(float)
    return np.asarray([y[int(pos) + 1:int(pos) + horizon + 1] for pos in origins], dtype=float)


def rolling_821_windows(panel: pd.DataFrame) -> list[dict]:
    """Janelas anuais com 8 anos de treino, 2 de validação e 1 de teste."""
    dates = pd.to_datetime(panel["date"])
    years = dates.dt.year.to_numpy()
    valid_origin = valid_origin_mask(panel)
    first_year, last_year = int(years.min()), int(years.max())
    first_test = first_year + C.TRAIN_YEARS + C.VAL_YEARS
    windows: list[dict] = []
    for window_id, test_year in enumerate(range(first_test, last_year + 1, C.STEP_YEARS)):
        train_start = test_year - C.VAL_YEARS - C.TRAIN_YEARS
        train_end = test_year - C.VAL_YEARS
        val_end = test_year
        train = np.flatnonzero((years >= train_start) & (years < train_end) & valid_origin)
        val = np.flatnonzero((years >= train_end) & (years < val_end) & valid_origin)
        test = np.flatnonzero((years == test_year) & valid_origin)
        if len(train) and len(val) and len(test):
            windows.append({
                "window_id": window_id,
                "test_year": test_year,
                "train_year_start": train_start,
                "train_year_end": train_end - 1,
                "val_year_start": train_end,
                "val_year_end": val_end - 1,
                "train_idx": train,
                "val_idx": val,
                "test_idx": test,
            })
    return windows


def write_split_manifest(panel: pd.DataFrame, windows: list[dict],
                         path: Path = C.SPLIT_MANIFEST) -> pd.DataFrame:
    rows = []
    for w in windows:
        row = {k: v for k, v in w.items() if not k.endswith("_idx")}
        for role in ["train", "val", "test"]:
            idx = w[f"{role}_idx"]
            row[f"{role}_start"] = panel.loc[idx[0], "date"]
            row[f"{role}_end"] = panel.loc[idx[-1], "date"]
            row[f"{role}_origins"] = len(idx)
        rows.append(row)
    manifest = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)
    return manifest


def feature_columns(panel: pd.DataFrame) -> list[str]:
    return [c for c in panel.columns
            if c not in C.NON_FEATURE_COLUMNS
            and c not in C.EXCLUDED_MAIN
            and pd.api.types.is_numeric_dtype(panel[c])]


def build_sequences(panel: pd.DataFrame, origins: np.ndarray, features: list[str],
                    input_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retorna X, Y e origens válidas sem cruzar o início da base."""
    raw = panel[features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = panel[C.TARGET].to_numpy(float)
    seq, targets, kept = [], [], []
    for origin in origins:
        origin = int(origin)
        lo = origin - input_size + 1
        if lo < 0 or origin + C.HORIZON >= len(panel):
            continue
        target = y[origin + 1:origin + C.HORIZON + 1]
        if np.isfinite(target).all():
            seq.append(raw[lo:origin + 1])
            targets.append(target)
            kept.append(origin)
    return np.asarray(seq, dtype=np.float32), np.asarray(targets, dtype=np.float32), np.asarray(kept, dtype=int)


def forecasts_long(panel: pd.DataFrame, origins: np.ndarray, predictions: np.ndarray,
                   model: str, window_id: int, artifact_path: str | None = None,
                   extra_origin_columns: list[str] | None = None) -> pd.DataFrame:
    rows = []
    extras = extra_origin_columns or []
    for i, origin in enumerate(origins):
        origin = int(origin)
        for h in range(1, C.HORIZON + 1):
            target_pos = origin + h
            row = {
                "model": model,
                "window_id": int(window_id),
                "origin_date": panel.loc[origin, "date"],
                "target_date": panel.loc[target_pos, "date"],
                "horizon": h,
                "y": float(panel.loc[target_pos, C.TARGET]),
                "y_hat": float(max(predictions[i, h - 1], C.EPS)),
                "artifact_path": artifact_path,
            }
            for col in extras:
                if col in panel:
                    row[f"x_{col}"] = panel.loc[origin, col]
            rows.append(row)
    out = pd.DataFrame(rows)
    out["error"] = out["y_hat"] - out["y"]
    out["squared_error"] = out["error"].pow(2)
    ratio = out["y"].clip(lower=C.EPS) / out["y_hat"].clip(lower=C.EPS)
    out["qlike_loss"] = ratio - np.log(ratio) - 1.0
    return out


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def qlike_by_horizon(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    ratio = np.maximum(y, C.EPS) / np.maximum(pred, C.EPS)
    return np.mean(ratio - np.log(ratio) - 1.0, axis=0)


def aggregate_qlike(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(qlike_by_horizon(y, pred)))

