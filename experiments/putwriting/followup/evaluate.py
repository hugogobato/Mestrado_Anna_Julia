"""Score aligned out-of-sample forecasts with and without retransformation."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from .statistical import PRIMARY

legacy = import_module("15_evaluate_daily")
KEYS = ["origin_date", "horizon"]
NEURAL = ("LSTM", "TSMixerX")


def load_forecasts(panel, statistical_dir: Path, neural_paths: list[Path]) -> pd.DataFrame:
    paths = [statistical_dir / "forecasts.parquet", *neural_paths]
    columns = ["model", *KEYS, "target_date", "window_id", "y", "y_hat", "y_hat_raw",
               "y_hat_log", "smearing_factor"]
    all_models = pd.concat([pd.read_parquet(path)[columns] for path in paths], ignore_index=True)
    expected = set(PRIMARY) | (set(NEURAL) if neural_paths else set())
    if set(all_models.model) != expected or all_models.duplicated(["model", *KEYS]).any():
        raise ValueError("Coverage or duplicate forecast key error")
    for date in ("origin_date", "target_date"):
        all_models[date] = pd.to_datetime(all_models[date]).astype("datetime64[ns]")
    ref = all_models.loc[all_models.model.eq("RW")].sort_values(KEYS)
    from .panel import C, followup_windows

    expected_origins = np.concatenate([w["test_idx"] for w in followup_windows(panel)])
    positions = expected_origins[:, None] + np.arange(1, C.HORIZON + 1)
    ref_origins = panel.date.to_numpy(dtype="datetime64[ns]")[expected_origins]
    dates = pd.to_datetime(panel.date).to_numpy(dtype="datetime64[ns]")
    target_positions = positions.ravel()
    available = target_positions < len(panel)
    expected_target_dates = np.full(len(target_positions), np.datetime64("NaT", "ns"), dtype=dates.dtype)
    expected_target_dates[available] = dates[target_positions[available]]
    expected_y = np.full(len(target_positions), np.nan)
    expected_y[available] = panel[C.TARGET].to_numpy()[target_positions[available]]
    if (not np.array_equal(ref.origin_date.to_numpy(), ref_origins.repeat(C.HORIZON))
            or not np.array_equal(ref.horizon.to_numpy(), np.tile(np.arange(1, C.HORIZON + 1), len(ref_origins)))
            or not np.array_equal(ref.target_date.to_numpy().view("i8"), expected_target_dates.view("i8"))
            or not np.allclose(ref.y, expected_y, rtol=0, atol=1e-12, equal_nan=True)):
        raise ValueError("RW forecast calendar or target mismatch with the panel")
    for model, group in all_models.groupby("model"):
        group = group.sort_values(KEYS).reset_index(drop=True)
        keys = [*KEYS, "target_date", "window_id"]
        if (not group[keys].equals(ref[keys].reset_index(drop=True))
                or not np.allclose(group.y, ref.y, rtol=1e-7, atol=1e-12, equal_nan=True)):
            raise ValueError(f"{model}: origins, targets, horizons or folds differ from reference")
        point = group[["y_hat", "y_hat_raw", "smearing_factor"]].to_numpy(float)
        if (not np.isfinite(point).all() or not (point > 0).all()
                or not np.allclose(point[:, 0], point[:, 1] * point[:, 2], rtol=1e-7, atol=0)):
            raise ValueError(f"{model}: forecast or smearing identity mismatch")
    return all_models.sort_values(["model", *KEYS], ignore_index=True)


def forecast_metrics(forecasts: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    naive = reference.loc[reference.model.eq("RW"), KEYS + ["y_hat"]].rename(columns={"y_hat": "rw"})
    joined = forecasts.merge(naive, on=KEYS, validate="many_to_one")
    rows = []
    for period, data in (("all", joined), ("2025_origins", joined[joined.origin_date.dt.year.eq(2025)])):
        for (model, h), group in data.groupby(["model", "horizon"]):
            valid = group.dropna(subset=["y"])
            y = valid.y.to_numpy(float)
            mse_rw = np.mean(np.square(y - valid.rw))
            for point, name in (("y_hat", "primary"), ("y_hat_raw", "without_smearing")):
                pred = valid[point].to_numpy(float)
                ratio = y / pred
                mse = np.mean(np.square(y - pred))
                rows.append({
                    "period": period, "model": model, "horizon": h, "retransformation": name,
                    "forecast_origins": len(group), "scored_origins": len(valid),
                    "QLIKE": (ratio - np.log(ratio) - 1).mean(), "MSE": mse,
                    "MAE": np.abs(pred - y).mean(), "R2_vs_RW": 1 - mse / mse_rw,
                    "mean_y": y.mean(), "mean_y_hat": pred.mean(), "bias": (pred - y).mean(),
                })
    return pd.DataFrame(rows)


def clark_west(har: pd.DataFrame) -> pd.DataFrame:
    """Use the original fitted linear log forecasts for the nested comparison."""
    rows = []
    for restricted, unrestricted in (
        ("HAR", "HAR-VIX"),
        ("HAR", "RS-HAR-VIX-level"),
        ("YZsmooth-HAR", "YZsmooth-HAR-VIX-log"),
        ("YZsmooth-HAR", "YZsmooth-HAR-VIX-level"),
    ):
        a = har.loc[har.model.eq(restricted)].set_index(KEYS).sort_index()
        b = har.loc[har.model.eq(unrestricted)].set_index(KEYS).sort_index()
        if not a.index.equals(b.index) or not a.y.equals(b.y) or not a.target_date.equals(b.target_date):
            raise ValueError("Mismatched nested samples")
        for h, group in a.dropna(subset=["y"]).groupby(level="horizon"):
            y = np.log(group.y)
            f0, f1 = group.y_hat_log, b.loc[group.index, "y_hat_log"]
            adjusted = (y - f0)**2 - (y - f1)**2 + (f0 - f1)**2
            for lags in (21, 42):
                stat, _ = legacy._hac_mean_test(adjusted.to_numpy(), maxlags=lags)
                rows.append({
                    "restricted": restricted, "unrestricted": unrestricted, "horizon": h,
                    "hac_lags": lags, "n": len(group), "mean_adjusted_log_loss_difference": adjusted.mean(),
                    "CW_log_stat": stat, "CW_log_one_sided_p": norm.sf(stat),
                })
    result = pd.DataFrame(rows)
    result["CW_log_BH_22_horizons"] = result.groupby(
        ["restricted", "unrestricted", "hac_lags"]
    ).CW_log_one_sided_p.transform(legacy._fdr_bh)
    return result


def write_forecast_evaluation(panel, statistical_dir, neural_paths, output):
    forecasts = load_forecasts(panel, statistical_dir, neural_paths)
    har = pd.read_parquet(statistical_dir / "har_comparison.parquet")
    forecast_metrics(forecasts, forecasts).to_csv(output / "forecast_metrics.csv", index=False)
    forecast_metrics(har, forecasts).to_csv(output / "har_specification_metrics.csv", index=False)
    clark_west(har).to_csv(output / "clark_west_log.csv", index=False)
    crisis = har.loc[
        har.horizon.eq(22) & har.origin_date.between("2020-03-02", "2020-04-30"),
        ["model", "origin_date", "target_date", "y", "y_hat_raw", "y_hat", "smearing_factor"],
    ].merge(panel[["date", "vix"]], left_on="origin_date", right_on="date").drop(columns="date")
    crisis.to_csv(output / "har_crisis_h22.csv", index=False)
    forecasts.loc[forecasts.horizon.eq(22)].to_parquet(output / "forecasts_h22.parquet", index=False)
    return forecasts
