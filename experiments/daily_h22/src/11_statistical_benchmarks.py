"""Benchmarks diários H=22 em janelas 8/2/1, com retomada por janela."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from feature_selection import select_features
from utils import (ensure_directories, forecasts_long, rolling_821_windows,
                   save_json, target_matrix)


MODEL_NAMES = ["RW", "HAR", "GARCH(1,1)", "VIX", "VIX_calibrated", "XGBoost"]


def _artifact_name(path: Path) -> str:
    return str(path.relative_to(C.EXPERIMENT_ROOT)) if path.is_relative_to(C.EXPERIMENT_ROOT) else str(path)


def _har_design(panel: pd.DataFrame, origins: np.ndarray) -> np.ndarray:
    y = panel[C.TARGET].to_numpy(float)
    rows = []
    for pos in origins:
        pos = int(pos)
        rows.append([1.0, np.log(max(y[pos], C.EPS)),
                     np.mean(np.log(np.maximum(y[max(0, pos - 4):pos + 1], C.EPS))),
                     np.mean(np.log(np.maximum(y[max(0, pos - 21):pos + 1], C.EPS)))])
    return np.asarray(rows, dtype=float)


def _har_predict(panel: pd.DataFrame, train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_train = _har_design(panel, train)
    x_test = _har_design(panel, test)
    log_y = np.log(np.maximum(target_matrix(panel, train), C.EPS))
    valid = np.isfinite(x_train).all(axis=1) & np.isfinite(log_y).all(axis=1)
    beta, *_ = np.linalg.lstsq(x_train[valid], log_y[valid], rcond=None)
    return np.exp(np.clip(x_test @ beta, -30, 10)), beta


def _fit_garch_params(returns: np.ndarray) -> dict:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    try:
        from arch import arch_model

        fit = arch_model(r * 100.0, mean="Zero", vol="GARCH", p=1, q=1,
                         dist="normal", rescale=False).fit(disp="off", show_warning=False)
        p = fit.params
        omega = max(float(p["omega"]) / 10000.0, 1e-12)
        alpha = float(p["alpha[1]"])
        beta = float(p["beta[1]"])
        last_variance = float(np.asarray(fit.conditional_volatility)[-1] ** 2) / 10000.0
    except (ImportError, ValueError, RuntimeError, np.linalg.LinAlgError):
        variance = max(float(np.var(r)), 1e-8)
        omega, alpha, beta, last_variance = variance * 0.05, 0.05, 0.90, variance
    if alpha + beta >= 0.999:
        beta = max(0.0, 0.998 - alpha)
    return {"omega": omega, "alpha": alpha, "beta": beta,
            "last_variance": max(last_variance, 1e-12)}


def _garch_predict(panel: pd.DataFrame, train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, dict]:
    returns = pd.to_numeric(panel["ret_spx_log"], errors="coerce").to_numpy(float)
    params = _fit_garch_params(returns[train])
    omega, alpha, beta = params["omega"], params["alpha"], params["beta"]

    # Reconstrói a variância condicional com parâmetros congelados, usando
    # retornos observados somente até cada origem.
    variance = params["last_variance"]
    cursor = int(train[-1])
    predictions = []
    for origin in test:
        origin = int(origin)
        for pos in range(cursor + 1, origin + 1):
            previous_return = returns[pos - 1] if pos > 0 else np.nan
            innovation = previous_return ** 2 if np.isfinite(previous_return) else variance
            variance = omega + alpha * innovation + beta * variance
        future = []
        state = variance
        last_return = returns[origin] if np.isfinite(returns[origin]) else 0.0
        state = omega + alpha * last_return ** 2 + beta * state
        future.append(state)
        for _ in range(1, C.HORIZON):
            state = omega + (alpha + beta) * state
            future.append(state)

        curve = []
        for h in range(1, C.HORIZON + 1):
            components = []
            for offset in range(h - C.RV_WINDOW + 1, h + 1):
                if offset <= 0:
                    pos = origin + offset
                    value = returns[pos] ** 2 if pos >= 0 and np.isfinite(returns[pos]) else variance
                else:
                    value = future[offset - 1]
                components.append(value)
            curve.append(C.ANNUALIZATION * float(np.mean(components)))
        predictions.append(curve)
        cursor = origin
    params["note"] = "GARCH daily variance combined with observed/predicted components in each trailing 22-day target window"
    return np.maximum(np.asarray(predictions), C.EPS), params


def _vix_predictions(panel: pd.DataFrame, origins: np.ndarray) -> np.ndarray:
    vix_variance = (pd.to_numeric(panel.loc[origins, "vix"], errors="coerce").to_numpy(float) / 100.0) ** 2
    return np.repeat(vix_variance[:, None], C.HORIZON, axis=1)


def _vix_calibrated(panel: pd.DataFrame, train: np.ndarray,
                    test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_train = _vix_predictions(panel, train)[:, 0]
    x_test = _vix_predictions(panel, test)[:, 0]
    design = np.column_stack([np.ones(len(x_train)), x_train])
    beta, *_ = np.linalg.lstsq(design, target_matrix(panel, train), rcond=None)
    pred = np.column_stack([np.ones(len(x_test)), x_test]) @ beta
    return np.maximum(pred, C.EPS), beta


def _xgb_matrix(panel: pd.DataFrame, origins: np.ndarray,
                selected: list[str]) -> tuple[np.ndarray, dict]:
    columns = [c for c in selected if c != C.TARGET]
    raw = panel[columns].apply(pd.to_numeric, errors="coerce")
    # Features contemporâneas mais lags autorregressivos fixos.
    matrix = raw.loc[origins].copy()
    y = panel[C.TARGET]
    for lag in [1, 5, 22, 66, 132]:
        matrix[f"target_lag_{lag}"] = y.shift(lag).loc[origins].to_numpy()
    return matrix.to_numpy(float), {"columns": matrix.columns.tolist()}


def _xgb_predict(panel: pd.DataFrame, train: np.ndarray, test: np.ndarray,
                 smoke: bool, window_id: int) -> tuple[np.ndarray, list, dict]:
    selected, ranking = select_features(panel, train, 20 if smoke else 50)
    x_train, metadata = _xgb_matrix(panel, train, selected)
    x_test, _ = _xgb_matrix(panel, test, selected)
    med = np.nanmedian(x_train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    x_train = np.where(np.isfinite(x_train), x_train, med)
    x_test = np.where(np.isfinite(x_test), x_test, med)
    y_train = np.log(np.maximum(target_matrix(panel, train), C.EPS))
    models, columns = [], []
    pred = np.empty((len(test), C.HORIZON), dtype=float)
    try:
        from xgboost import XGBRegressor

        for h in range(C.HORIZON):
            model = XGBRegressor(
                n_estimators=20 if smoke else 250,
                max_depth=2 if smoke else 3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=C.SEED + h,
                n_jobs=max(1, min(8, (os.cpu_count() or 2) // 2)),
                tree_method="hist",
            )
            model.fit(x_train, y_train[:, h])
            pred[:, h] = np.exp(np.clip(model.predict(x_test), -30, 10))
            models.append(model)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor

        for h in range(C.HORIZON):
            model = HistGradientBoostingRegressor(max_iter=20 if smoke else 200,
                                                   max_leaf_nodes=8, random_state=C.SEED + h)
            model.fit(x_train, y_train[:, h])
            pred[:, h] = np.exp(np.clip(model.predict(x_test), -30, 10))
            models.append(model)
    metadata.update({"selected_features": selected, "median": med.tolist(),
                     "ranking_records": ranking.to_dict("records")})
    return np.maximum(pred, C.EPS), models, metadata


def _partial_path(model: str, window_id: int) -> Path:
    slug = model.lower().replace("(1,1)", "_11").replace(",", "").replace(" ", "_")
    path = C.FORECAST_DIR / "partial" / f"{slug}_window_{window_id:02d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_forecast(frame: pd.DataFrame, model: str, window_id: int) -> None:
    path = _partial_path(model, window_id)
    frame.to_parquet(path, index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False)


def combine_outputs() -> None:
    for model in MODEL_NAMES:
        parts = sorted((_partial_path(model, 0).parent).glob(
            f"{model.lower().replace('(1,1)', '_11').replace(',', '').replace(' ', '_')}_window_*.parquet"))
        if not parts:
            continue
        frame = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        slug = model.lower().replace("(1,1)", "_11").replace(",", "").replace(" ", "_")
        frame.to_parquet(C.FORECAST_DIR / f"{slug}.parquet", index=False)
        frame.to_csv(C.FORECAST_DIR / f"{slug}.csv", index=False)


def run(smoke: bool = False, max_windows: int | None = None,
        window_ids: list[int] | None = None, resume: bool = True,
        time_budget_minutes: float | None = None) -> None:
    ensure_directories()
    panel = pd.read_parquet(C.DAILY_PANEL).sort_values("date").reset_index(drop=True)
    windows = rolling_821_windows(panel)
    if window_ids is not None:
        windows = [w for w in windows if w["window_id"] in window_ids]
    if max_windows is not None:
        windows = windows[:max_windows]
    started = time.monotonic()
    start_buffer = min(10.0, max(1.0, (time_budget_minutes or 0) * 0.20))
    safe_start_limit = ((time_budget_minutes - start_buffer)
                        if time_budget_minutes else None)
    extras = ["vix", "spx_close", "close_variance_forward_22", "close_volatility_forward_22"]

    for w in windows:
        if safe_start_limit is not None and (time.monotonic() - started) / 60 >= safe_start_limit:
            print("Limite seguro atingido; nenhuma nova janela será iniciada. "
                  "Os outputs concluídos já estão no armazenamento persistente.")
            break
        window_id, train, test = w["window_id"], w["train_idx"], w["test_idx"]
        print(f"Janela {window_id:02d}, teste {w['test_year']}, origens={len(test)}")

        if not (resume and _partial_path("RW", window_id).exists()):
            rw = np.repeat(panel.loc[test, C.TARGET].to_numpy(float)[:, None], C.HORIZON, axis=1)
            _save_forecast(forecasts_long(panel, test, rw, "RW", window_id,
                                          extra_origin_columns=extras), "RW", window_id)

        if not (resume and _partial_path("HAR", window_id).exists()):
            pred, beta = _har_predict(panel, train, test)
            artifact = C.MODELS_DIR / "har" / f"har_window_{window_id:02d}.json"
            window_meta = {k: v for k, v in w.items() if not k.endswith("_idx")}
            save_json({"beta": beta.tolist(),
                       "features": ["const", "log_rv_d", "log_rv_w", "log_rv_m"],
                       **window_meta}, artifact)
            _save_forecast(forecasts_long(panel, test, pred, "HAR", window_id,
                                          _artifact_name(artifact), extras), "HAR", window_id)

        if not (resume and _partial_path("GARCH(1,1)", window_id).exists()):
            pred, params = _garch_predict(panel, train, test)
            artifact = C.MODELS_DIR / "garch" / f"garch_window_{window_id:02d}.json"
            save_json({**params, **{k: v for k, v in w.items() if not k.endswith('_idx')}}, artifact)
            _save_forecast(forecasts_long(panel, test, pred, "GARCH(1,1)", window_id,
                                          _artifact_name(artifact), extras), "GARCH(1,1)", window_id)

        if not (resume and _partial_path("VIX", window_id).exists()):
            pred = _vix_predictions(panel, test)
            _save_forecast(forecasts_long(panel, test, pred, "VIX", window_id,
                                          extra_origin_columns=extras), "VIX", window_id)

        if not (resume and _partial_path("VIX_calibrated", window_id).exists()):
            pred, beta = _vix_calibrated(panel, train, test)
            artifact = C.MODELS_DIR / "vix_calibrated" / f"vix_calibrated_window_{window_id:02d}.json"
            save_json({"beta_by_horizon": beta.tolist(), **{k: v for k, v in w.items() if not k.endswith('_idx')}}, artifact)
            _save_forecast(forecasts_long(panel, test, pred, "VIX_calibrated", window_id,
                                          _artifact_name(artifact), extras), "VIX_calibrated", window_id)

        if not (resume and _partial_path("XGBoost", window_id).exists()):
            pred, models, metadata = _xgb_predict(panel, train, test, smoke, window_id)
            artifact = C.MODELS_DIR / "xgboost" / f"xgboost_window_{window_id:02d}.joblib"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            try:
                import joblib
                joblib.dump({"models": models, "metadata": metadata}, artifact)
                artifact_rel = _artifact_name(artifact)
            except ImportError:
                artifact_rel = None
            selected = metadata["selected_features"]
            xgb_values, xgb_columns_meta = _xgb_matrix(panel, test, selected)
            input_frame = pd.DataFrame(xgb_values, columns=xgb_columns_meta["columns"])
            input_frame.insert(0, "origin_date", panel.loc[test, "date"].to_numpy())
            input_frame.insert(0, "window_id", window_id)
            input_path = C.INPUTS_DIR / f"xgboost_window_{window_id:02d}.parquet"
            input_path.parent.mkdir(parents=True, exist_ok=True)
            input_frame.to_parquet(input_path, index=False)
            xgb_extras = [c for c in selected if c != C.TARGET]
            for common in extras:
                if common not in xgb_extras:
                    xgb_extras.append(common)
            _save_forecast(forecasts_long(panel, test, pred, "XGBoost", window_id,
                                          artifact_rel, xgb_extras), "XGBoost", window_id)
    combine_outputs()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--window-ids", type=int, nargs="*")
    parser.add_argument("--time-budget-minutes", type=float, default=60.0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    run(args.smoke, 1 if args.smoke and args.max_windows is None else args.max_windows,
        args.window_ids, not args.no_resume, args.time_budget_minutes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
