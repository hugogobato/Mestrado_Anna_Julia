"""Fit the corrected statistical forecasts and the controlled HAR specification comparison."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .panel import C, EXCLUDED_FEATURES, RESULTS, followup_windows
from feature_selection import select_features
from utils import save_json, target_matrix

legacy = import_module("11_statistical_benchmarks")
H = C.HORIZON
PRIMARY = ("RW", "HAR", "HAR-VIX", "GARCH(1,1)", "VIX", "VIX_calibrated", "XGBoost")


def har_design(panel: pd.DataFrame) -> dict[str, np.ndarray]:
    """Compute HAR predictors at the close with trailing 1, 5 and 22 sessions."""
    oc = np.log(panel.spx_close / panel.spx_open)
    ho = np.log(panel.spx_high / panel.spx_open)
    lo = np.log(panel.spx_low / panel.spx_open)
    rs = (ho * (ho - oc) + lo * (lo - oc)).clip(lower=0) * C.ANNUALIZATION
    daily_range = [np.log(rs.rolling(n).mean().clip(lower=C.EPS)) for n in (1, 5, 22)]
    smooth_yz = [np.log(panel[C.TARGET].clip(lower=C.EPS)).rolling(n).mean() for n in (1, 5, 22)]
    intercept = np.ones(len(panel))
    implied = np.square(panel.vix / 100)
    return {
        "HAR": np.column_stack([intercept, *daily_range]),
        "HAR-VIX": np.column_stack([intercept, *daily_range, np.log(implied)]),
        "RS-HAR-VIX-level": np.column_stack([intercept, *daily_range, implied]),
        "YZsmooth-HAR": np.column_stack([intercept, *smooth_yz]),
        "YZsmooth-HAR-VIX-level": np.column_stack([intercept, *smooth_yz, implied]),
        "YZsmooth-HAR-VIX-log": np.column_stack([intercept, *smooth_yz, np.log(implied)]),
    }


def smearing(log_y: np.ndarray, fitted_log_y: np.ndarray) -> np.ndarray:
    """Duan's pooled empirical residual multiplier, separately by horizon."""
    residual = log_y - fitted_log_y
    return np.exp(logsumexp(residual, axis=0) - np.log(len(residual)))


def forecast_frame(
    panel: pd.DataFrame,
    window: dict,
    model: str,
    point_estimate: np.ndarray,
    factor: float | np.ndarray = 1.0,
    log_scale: bool = False,
) -> pd.DataFrame:
    origins = window["test_idx"]
    dates = panel.date.to_numpy(dtype="datetime64[ns]")
    targets = panel[C.TARGET].to_numpy(float)
    idx = (origins[:, None] + np.arange(1, H + 1)).ravel()
    valid = idx < len(panel)
    realized = np.full(len(idx), np.nan)
    realized[valid] = targets[idx[valid]]
    target_dates = np.full(len(idx), np.datetime64("NaT", "ns"), dtype=dates.dtype)
    target_dates[valid] = dates[idx[valid]]
    raw = np.exp(point_estimate) if log_scale else point_estimate
    factors = np.broadcast_to(factor, raw.shape)
    if (not np.isfinite(raw).all() or np.any(raw <= 0) or not np.isfinite(factors).all()
            or np.any(factors <= 0) or not np.isfinite(raw * factors).all()):
        raise ValueError(f"Invalid forecast or smearing factor from {model}")
    return pd.DataFrame({
        "model": model,
        "window_id": window["window_id"],
        "origin_date": dates[origins].repeat(H),
        "target_date": target_dates,
        "horizon": np.tile(np.arange(1, H + 1), len(origins)),
        "y": realized,
        "y_hat_raw": raw.ravel(),
        "y_hat_log": point_estimate.ravel() if log_scale else np.log(raw.ravel()),
        "smearing_factor": factors.ravel(),
        "y_hat": (raw * factors).ravel(),
    })


def fit_xgboost(panel, train, test, log_y, window_id, output, threads):
    import xgboost

    selected, ranking = select_features(panel, train, 50)
    x_train, metadata = legacy._xgb_matrix(panel, train, selected)
    x_test, _ = legacy._xgb_matrix(panel, test, selected)
    median = np.nanmedian(x_train, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    x_train = np.where(np.isfinite(x_train), x_train, median)
    x_test = np.where(np.isfinite(x_test), x_test, median)
    fitted = np.empty((len(train), H))
    predictions = np.empty((len(test), H))
    model_dir = output / "xgboost_models"
    model_dir.mkdir(exist_ok=True)
    artifact = model_dir / f"window_{window_id:02d}.zip"
    with ZipFile(artifact, "w", compression=ZIP_DEFLATED) as archive:
        for h in range(H):
            model = xgboost.XGBRegressor(
                n_estimators=250,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=C.SEED + h,
                n_jobs=threads,
                tree_method="hist",
            )
            model.fit(x_train, log_y[:, h])
            fitted[:, h] = model.predict(x_train)
            predictions[:, h] = model.predict(x_test)
            archive.writestr(f"horizon_{h + 1:02d}.ubj", bytes(model.get_booster().save_raw(raw_format="ubj")))
        report = ranking.to_csv(index=False)
        archive.writestr("feature_ranking.csv", report)
    return predictions, smearing(log_y, fitted), {
        **metadata, "median": median.tolist(), "artifact": str(artifact.relative_to(output)),
        "xgboost_version": xgboost.__version__,
    }


def run(panel: pd.DataFrame, output: Path, workers: int = 2, threads: int = 2) -> None:
    if any(column in panel for column in EXCLUDED_FEATURES):
        raise ValueError("Panel contains excluded features")
    output.mkdir(parents=True, exist_ok=True)
    design = har_design(panel)

    def fit_window(window):
        train, test, wid = window["train_idx"], window["test_idx"], window["window_id"]
        log_y = np.log(target_matrix(panel, train))
        scorable = np.isfinite(log_y).all(axis=1)
        # Same origins for all six HAR variants, including both members of each nested pair.
        for x in design.values():
            scorable &= np.isfinite(x[train]).all(axis=1)
            if not np.isfinite(x[test]).all():
                raise ValueError(f"HAR predictors missing at test origins in fold {wid}")
        primary, comparison, metadata = [], [], {}
        for name, x in design.items():
            beta, *_ = np.linalg.lstsq(x[train][scorable], log_y[scorable], rcond=None)
            factor = smearing(log_y[scorable], x[train][scorable] @ beta)
            result = forecast_frame(panel, window, name, x[test] @ beta, factor, log_scale=True)
            comparison.append(result)
            if name in PRIMARY:
                primary.append(result)
            metadata[name] = {
                "coefficients": beta.tolist(), "smearing": factor.tolist(), "training_rows": int(scorable.sum()),
            }
        for name, prediction in (
            ("RW", np.repeat(panel.loc[test, C.TARGET].to_numpy()[:, None], H, axis=1)),
            ("VIX", legacy._vix_predictions(panel, test)),
        ):
            primary.append(forecast_frame(panel, window, name, prediction))
        garch, params = legacy._garch_predict(panel, train, test)
        calibrated, coefficients = legacy._vix_calibrated(panel, train, test)
        primary.append(forecast_frame(panel, window, "GARCH(1,1)", garch))
        primary.append(forecast_frame(panel, window, "VIX_calibrated", calibrated))
        metadata["GARCH(1,1)"] = params
        metadata["VIX_calibrated"] = {"coefficients": coefficients.tolist()}
        xgb, factor, xgb_metadata = fit_xgboost(panel, train, test, log_y, wid, output, threads)
        primary.append(forecast_frame(panel, window, "XGBoost", xgb, factor, log_scale=True))
        metadata["XGBoost"] = {**xgb_metadata, "smearing": factor.tolist()}
        save_json(metadata, output / "fit_parameters" / f"window_{wid:02d}.json")
        print(f"Statistical fold {wid:02d}, test year {window['test_year']}, {len(test)} origins", flush=True)
        return pd.concat(primary, ignore_index=True), pd.concat(comparison, ignore_index=True)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(fit_window, followup_windows(panel)))
    stats, har = zip(*results)
    pd.concat(stats, ignore_index=True).to_parquet(output / "forecasts.parquet", index=False)
    pd.concat(har, ignore_index=True).to_parquet(output / "har_comparison.parquet", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=RESULTS / "panel.parquet")
    parser.add_argument("--output", type=Path, default=RESULTS / "statistical")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    run(pd.read_parquet(args.panel), args.output, args.workers, args.threads)
