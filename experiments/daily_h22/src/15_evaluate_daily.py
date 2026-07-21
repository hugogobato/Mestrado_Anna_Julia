"""Avaliação por horizonte, testes com HAC/MCS e figuras do experimento v2."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from utils import ensure_directories, save_json

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_daily_h22"
try:
    from statsmodels.tools.sm_exceptions import ValueWarning

    warnings.filterwarnings("ignore", category=ValueWarning)
except ImportError:
    pass


def _load_forecasts() -> pd.DataFrame:
    files = sorted(C.FORECAST_DIR.glob("*.parquet"))
    files = [p for p in files if p.name != "all_models.parquet"]
    if not files:
        raise FileNotFoundError("Nenhum forecast consolidado em results/forecasts.")
    frames = [pd.read_parquet(p) for p in files]
    out = pd.concat(frames, ignore_index=True)
    out["origin_date"] = pd.to_datetime(out["origin_date"])
    out["target_date"] = pd.to_datetime(out["target_date"])
    out = out.drop_duplicates(["model", "origin_date", "target_date", "horizon"], keep="last")
    return out.sort_values(["model", "origin_date", "horizon"]).reset_index(drop=True)


def _fdr_bh(pvalues: pd.Series) -> pd.Series:
    p = pvalues.to_numpy(float)
    valid = np.isfinite(p)
    result = np.full(len(p), np.nan)
    if not valid.any():
        return pd.Series(result, index=pvalues.index)
    values = p[valid]
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0, 1)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    result[np.flatnonzero(valid)] = restored
    return pd.Series(result, index=pvalues.index)


def _hac_mean_test(d: np.ndarray, maxlags: int = 21) -> tuple[float, float]:
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 30:
        return np.nan, np.nan
    try:
        import statsmodels.api as sm

        fit = sm.OLS(d, np.ones((len(d), 1))).fit(
            cov_type="HAC", cov_kwds={"maxlags": min(maxlags, len(d) - 2)})
        return float(fit.tvalues[0]), float(fit.pvalues[0])
    except (ImportError, ValueError, np.linalg.LinAlgError):
        return np.nan, np.nan


def _gw_test(loss_diff: np.ndarray, model_pred: np.ndarray,
             rw_pred: np.ndarray, lag_y: np.ndarray) -> tuple[float, float]:
    try:
        import statsmodels.api as sm

        z = np.column_stack([np.ones(len(loss_diff)), model_pred, rw_pred, lag_y])
        valid = np.isfinite(loss_diff) & np.isfinite(z).all(axis=1)
        d, z = loss_diff[valid], z[valid]
        independent, rank = [], 0
        for j in range(z.shape[1]):
            candidate = z[:, independent + [j]]
            new_rank = np.linalg.matrix_rank(candidate)
            if new_rank > rank:
                independent.append(j)
                rank = new_rank
        fit = sm.OLS(d, z[:, independent]).fit(cov_type="HAC", cov_kwds={"maxlags": 21})
        test = fit.wald_test(np.eye(len(independent)), scalar=True)
        return float(np.asarray(test.statistic).squeeze()), float(np.asarray(test.pvalue).squeeze())
    except (ImportError, ValueError, np.linalg.LinAlgError):
        return np.nan, np.nan


def metrics_by_horizon(forecasts: pd.DataFrame) -> pd.DataFrame:
    rw = forecasts.loc[forecasts.model == "RW",
                       ["origin_date", "target_date", "horizon", "y_hat"]].rename(columns={"y_hat": "rw_hat"})
    rows = []
    for (model, horizon), group in forecasts.groupby(["model", "horizon"], sort=True):
        joined = group.merge(rw, on=["origin_date", "target_date", "horizon"], how="left")
        joined = joined.dropna(subset=["y", "y_hat", "rw_hat"])
        y, pred, naive = joined.y.to_numpy(float), joined.y_hat.to_numpy(float), joined.rw_hat.to_numpy(float)
        ratio = np.maximum(y, C.EPS) / np.maximum(pred, C.EPS)
        qloss = ratio - np.log(ratio) - 1.0
        mse_model = float(np.mean((y - pred) ** 2))
        mse_rw = float(np.mean((y - naive) ** 2))
        r2 = 1.0 - mse_model / mse_rw if mse_rw > 0 else np.nan
        rows.append({
            "model": model,
            "horizon": int(horizon),
            "n": len(joined),
            "QLIKE": float(np.mean(qloss)),
            "MSE": mse_model,
            "MAE": float(np.mean(np.abs(y - pred))),
            "R2_oos": r2,
            "mean_y": float(np.mean(y)),
            "mean_y_hat": float(np.mean(pred)),
            "bias": float(np.mean(pred - y)),
        })
    return pd.DataFrame(rows)


def predictive_tests(forecasts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rw = forecasts.loc[forecasts.model == "RW",
                       ["origin_date", "target_date", "horizon", "y_hat"]].rename(columns={"y_hat": "rw_hat"})
    dm_rows, gw_rows = [], []
    for (model, horizon), group in forecasts.groupby(["model", "horizon"]):
        if model == "RW":
            continue
        joined = group.merge(rw, on=["origin_date", "target_date", "horizon"], how="inner").dropna(
            subset=["y", "y_hat", "rw_hat"])
        y, pred, naive = joined.y.to_numpy(float), joined.y_hat.to_numpy(float), joined.rw_hat.to_numpy(float)
        ratio_model = np.maximum(y, C.EPS) / np.maximum(pred, C.EPS)
        ratio_rw = np.maximum(y, C.EPS) / np.maximum(naive, C.EPS)
        loss_model = ratio_model - np.log(ratio_model) - 1.0
        loss_rw = ratio_rw - np.log(ratio_rw) - 1.0
        # Positivo favorece o modelo porque loss_RW - loss_model > 0.
        d = loss_rw - loss_model
        stat, pvalue = _hac_mean_test(d, maxlags=21)
        dm_rows.append({"model": model, "horizon": horizon, "DM_stat_vs_RW": stat,
                        "DM_pvalue_vs_RW": pvalue, "n": len(joined)})
        lag_y = np.roll(y, 1)
        lag_y[0] = y[0]
        gw_stat, gw_p = _gw_test(d, pred, naive, lag_y)
        gw_rows.append({"model": model, "horizon": horizon, "GW_stat_vs_RW": gw_stat,
                        "GW_pvalue_vs_RW": gw_p, "n": len(joined)})
    dm = pd.DataFrame(dm_rows)
    gw = pd.DataFrame(gw_rows)
    if len(dm):
        dm["DM_pvalue_FDR_all_horizons"] = _fdr_bh(dm.DM_pvalue_vs_RW)
    if len(gw):
        gw["GW_pvalue_FDR_all_horizons"] = _fdr_bh(gw.GW_pvalue_vs_RW)
    return dm, gw


def run_mcs_h22(forecasts: pd.DataFrame, reps: int = 1000,
                 block_size: int = 22, size: float = 0.05) -> tuple[pd.DataFrame, dict]:
    from arch.bootstrap import MCS

    h22 = forecasts.loc[forecasts.horizon == 22]
    losses = h22.assign(loss=(h22.y - h22.y_hat) ** 2).pivot_table(
        index="origin_date", columns="model", values="loss", aggfunc="first").dropna()
    if losses.shape[0] < 30 or losses.shape[1] < 2:
        return pd.DataFrame(), {}
    mcs = MCS(losses, size=size, reps=reps, block_size=block_size,
              method="R", seed=C.SEED)
    mcs.compute()
    pvalues = mcs.pvalues.copy()
    mapping = pvalues["Pvalue"].to_dict() if "Pvalue" in pvalues else {}
    membership = pd.DataFrame({"model": losses.columns})
    membership["included"] = membership.model.isin(mcs.included)
    membership["pvalue"] = membership.model.map(mapping)
    config = {"implementation": "arch.bootstrap.MCS", "loss": "MSE", "horizon": 22,
              "size": size, "reps": reps, "block_size": block_size,
              "method": "R", "n_common_origins": len(losses)}
    return membership, config


def _make_figures(forecasts: pd.DataFrame, metrics: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for metric in ["QLIKE", "MSE", "MAE", "R2_oos"]:
        fig, ax = plt.subplots(figsize=(11, 6))
        for model, group in metrics.groupby("model"):
            ax.plot(group.horizon, group[metric], marker="o", markersize=2.5, label=model)
        ax.set_xlabel("Horizonte (pregões)")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} por horizonte")
        ax.grid(alpha=0.25)
        ax.legend(ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(C.FIGURES_DIR / f"{metric.lower()}_by_horizon.png", dpi=180)
        plt.close(fig)

    h22 = forecasts.loc[forecasts.horizon == 22].copy()
    actual = h22.drop_duplicates("origin_date").sort_values("origin_date")
    if "x_vix" in actual:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(actual.origin_date, np.sqrt(actual.y) * 100, color="black", label="RV Yang-Zhang futura")
        ax.plot(actual.origin_date, actual.x_vix, color="tab:blue", alpha=0.8, label="VIX na origem")
        ax.set_ylabel("Volatilidade anualizada (%)")
        ax.set_title("VIX e volatilidade realizada Yang-Zhang, horizonte 22")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(C.FIGURES_DIR / "vix_vs_rv_volatility.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(actual.x_vix, np.sqrt(actual.y) * 100, s=8, alpha=0.35)
        bounds = [0, max(actual.x_vix.max(), (np.sqrt(actual.y) * 100).max())]
        ax.plot(bounds, bounds, color="black", linestyle="--", label="45 graus")
        ax.set_xlabel("VIX na origem (%)")
        ax.set_ylabel("RV futura (%)")
        ax.set_title("VIX versus RV futura, h=22")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(C.FIGURES_DIR / "vix_rv_scatter.png", dpi=180)
        plt.close(fig)

        corr = actual.set_index("origin_date")
        rolling = corr.x_vix.rolling(252).corr(np.sqrt(corr.y) * 100)
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(rolling.index, rolling, color="tab:purple")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Correlação móvel de 252 pregões: VIX versus RV futura")
        ax.set_ylabel("Correlação")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(C.FIGURES_DIR / "vix_rv_rolling_correlation.png", dpi=180)
        plt.close(fig)

    pivot = h22.pivot_table(index="origin_date", columns="model", values="y_hat", aggfunc="first")
    y = h22.drop_duplicates("origin_date").set_index("origin_date")["y"]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(y.index, y, color="black", linewidth=1.4, label="Realizado")
    for model in pivot:
        ax.plot(pivot.index, pivot[model], alpha=0.7, linewidth=0.8, label=model)
    ax.set_title("Previsões diárias para h=22")
    ax.set_ylabel("Variância anualizada")
    ax.grid(alpha=0.25)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "forecasts_h22.png", dpi=180)
    plt.close(fig)


def _convergence_figure() -> None:
    import matplotlib.pyplot as plt

    files = sorted(C.HISTORY_DIR.glob("*.csv"))
    if not files:
        return
    histories = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    for model, group in histories.groupby("model"):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for window_id, window in group.groupby("window_id"):
            axes[0].plot(window.step, window.train_loss, alpha=0.55, label=f"w{int(window_id):02d}")
            axes[1].plot(window.step, window.validation_qlike, alpha=0.55, label=f"w{int(window_id):02d}")
            best = window.loc[window.validation_qlike.idxmin()]
            axes[1].scatter(best.step, best.validation_qlike, s=12)
        axes[0].set_title(f"{model}: loss de treino")
        axes[1].set_title(f"{model}: QLIKE de validação")
        for ax in axes:
            ax.set_xlabel("Passo")
            ax.grid(alpha=0.25)
        axes[1].set_yscale("log")
        axes[1].legend(ncol=2, fontsize=7)
        fig.tight_layout()
        fig.savefig(C.FIGURES_DIR / f"convergence_{model}.png", dpi=180)
        plt.close(fig)


def run(smoke: bool = False, require_complete: bool = False,
        mcs_reps: int = 1000) -> None:
    ensure_directories()
    forecasts = _load_forecasts()
    expected_windows = 16
    coverage = forecasts.groupby("model").window_id.nunique().rename("completed_windows").reset_index()
    coverage["expected_windows"] = expected_windows
    coverage["complete"] = coverage.completed_windows == expected_windows
    coverage.to_csv(C.METRICS_DIR / "model_coverage.csv", index=False)
    if require_complete and not coverage.complete.all():
        raise RuntimeError("Há modelos com janelas incompletas. Consulte model_coverage.csv no notebook 2.")

    forecasts.to_parquet(C.FORECAST_DIR / "all_models.parquet", index=False)
    metrics = metrics_by_horizon(forecasts)
    metrics.to_csv(C.METRICS_DIR / "metrics_by_horizon.csv", index=False)
    metrics.loc[metrics.horizon == 22].to_csv(C.METRICS_DIR / "metrics_h22.csv", index=False)
    aggregate = metrics.groupby("model").agg(
        horizons=("horizon", "nunique"), n_mean=("n", "mean"),
        QLIKE_macro=("QLIKE", "mean"), MSE_macro=("MSE", "mean"),
        MAE_macro=("MAE", "mean"), R2_oos_macro=("R2_oos", "mean"),
    ).reset_index()
    aggregate.to_csv(C.METRICS_DIR / "metrics_aggregate.csv", index=False)
    by_window = forecasts.groupby(["model", "window_id", "horizon"]).agg(
        n=("y", "size"), MSE=("squared_error", "mean"),
        MAE=("error", lambda s: float(np.mean(np.abs(s)))),
        QLIKE=("qlike_loss", "mean"),
    ).reset_index()
    by_window.to_csv(C.METRICS_DIR / "metrics_by_window.csv", index=False)

    dm, gw = predictive_tests(forecasts)
    dm.to_csv(C.METRICS_DIR / "dm_vs_rw_by_horizon.csv", index=False)
    gw.to_csv(C.METRICS_DIR / "gw_vs_rw_by_horizon.csv", index=False)
    membership, mcs_config = run_mcs_h22(forecasts, reps=100 if smoke else mcs_reps)
    if len(membership):
        membership.to_csv(C.METRICS_DIR / "mcs_h22_membership.csv", index=False)
        save_json(mcs_config, C.METRICS_DIR / "mcs_h22_config.json")

    h22 = forecasts.loc[forecasts.horizon == 22].drop_duplicates("origin_date")
    correlation = {}
    if "x_vix" in h22:
        correlation = {
            "n": len(h22),
            "pearson_vix_vs_volatility": float(h22.x_vix.corr(np.sqrt(h22.y) * 100)),
            "spearman_vix_vs_volatility": float(h22.x_vix.corr(np.sqrt(h22.y) * 100, method="spearman")),
            "pearson_vix2_vs_variance": float(((h22.x_vix / 100) ** 2).corr(h22.y)),
            "mean_vix_percent": float(h22.x_vix.mean()),
            "mean_realized_volatility_percent": float((np.sqrt(h22.y) * 100).mean()),
            "mean_variance_risk_premium": float((((h22.x_vix / 100) ** 2) - h22.y).mean()),
        }
        save_json(correlation, C.METRICS_DIR / "vix_rv_correlation.json")
    _make_figures(forecasts, metrics)
    _convergence_figure()
    save_json({"models": sorted(forecasts.model.unique()), "rows": len(forecasts),
               "coverage": coverage.to_dict("records"), "vix_correlation": correlation,
               "horizon": C.HORIZON, "window": "8/2/1", "mcs": mcs_config},
              C.RESULTS_DIR / "evaluation_manifest.json")
    print(metrics.loc[metrics.horizon == 22].sort_values("MSE").to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--mcs-reps", type=int, default=1000)
    args = parser.parse_args()
    run(args.smoke, args.require_complete, args.mcs_reps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
