"""Avalia previsões, testes de superioridade e produz figuras do paper."""

from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from utils import mae, mse, qlike, r2_oos

warnings.filterwarnings("ignore")


def pointwise_loss(y: np.ndarray, pred: np.ndarray, kind: str = "QLIKE") -> np.ndarray:
    y = np.asarray(y, dtype=float)
    pred = np.maximum(np.asarray(pred, dtype=float), 1e-12)
    if kind.upper() == "QLIKE":
        ratio = np.maximum(y, 1e-12) / pred
        return ratio - np.log(ratio) - 1.0
    if kind.upper() == "MSE":
        return (y - pred) ** 2
    return np.abs(y - pred)


def _hac_variance(x: np.ndarray, max_lag: int | None = None) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return np.nan
    max_lag = min(max_lag if max_lag is not None else int(n ** (1 / 3)), n - 1)
    centered = x - x.mean()
    gamma0 = float(np.mean(centered * centered))
    var = gamma0
    for lag in range(1, max_lag + 1):
        gamma = float(np.mean(centered[lag:] * centered[:-lag]))
        weight = 1.0 - lag / (max_lag + 1.0)
        var += 2.0 * weight * gamma
    return max(var, 1e-18)


def dm_test(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
            loss: str = "QLIKE", max_lag: int | None = None) -> dict:
    """Diebold-Mariano com variância HAC; positivo favorece A."""
    d = pointwise_loss(y, pred_b, loss) - pointwise_loss(y, pred_a, loss)
    d = d[np.isfinite(d)]
    if len(d) < 3:
        return {"stat": np.nan, "pvalue": np.nan, "n": len(d)}
    var = _hac_variance(d, max_lag)
    stat = float(d.mean() / np.sqrt(var / len(d)))
    try:
        from scipy.stats import norm

        pvalue = float(2.0 * norm.sf(abs(stat)))
    except ImportError:
        pvalue = np.nan
    return {"stat": stat, "pvalue": pvalue, "n": len(d)}


def gw_test(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
            max_lag: int | None = None) -> dict:
    """Giacomini-White via regressão HAC da diferença de QLIKE.

    Os instrumentos são constante, previsão do modelo A, previsão do modelo
    B e alvo defasado. O teste conjunto verifica superioridade condicional.
    """
    y = np.asarray(y, dtype=float)
    a = np.asarray(pred_a, dtype=float)
    b = np.asarray(pred_b, dtype=float)
    d = pointwise_loss(y, b) - pointwise_loss(y, a)
    z = np.column_stack([np.ones(len(y)), a, b, np.roll(y, 1)])
    z[0, -1] = y[0]
    valid = np.isfinite(d) & np.isfinite(z).all(axis=1)
    d, z = d[valid], z[valid]
    # VIX e VIX3M, ou previsões muito persistentes, podem tornar alguns
    # instrumentos linearmente dependentes. Mantemos uma base de colunas
    # independente antes da regressão HAC.
    independent: list[int] = []
    current_rank = 0
    for j in range(z.shape[1]):
        candidate = z[:, independent + [j]]
        rank = np.linalg.matrix_rank(candidate)
        if rank > current_rank:
            independent.append(j)
            current_rank = rank
    z = z[:, independent]
    if len(d) <= z.shape[1] + 2:
        return {"stat": np.nan, "pvalue": np.nan, "n": len(d)}
    try:
        import statsmodels.api as sm

        maxlags = min(max_lag if max_lag is not None else int(len(d) ** (1 / 3)), len(d) // 3)
        fit = sm.OLS(d, z).fit(cov_type="HAC", cov_kwds={"maxlags": max(1, maxlags)})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test = fit.wald_test(np.eye(z.shape[1]), scalar=True)
        return {"stat": float(np.asarray(test.statistic).squeeze()),
                "pvalue": float(np.asarray(test.pvalue).squeeze()), "n": len(d)}
    except (ImportError, ValueError, np.linalg.LinAlgError):
        return {"stat": np.nan, "pvalue": np.nan, "n": len(d)}


def clark_west(y: np.ndarray, larger_pred: np.ndarray, nested_pred: np.ndarray) -> dict:
    """Clark-West para o modelo maior contra um benchmark aninhado."""
    y = np.asarray(y, dtype=float)
    e_larger = y - np.asarray(larger_pred, dtype=float)
    e_nested = y - np.asarray(nested_pred, dtype=float)
    adjusted = e_nested ** 2 - (e_larger ** 2 - (np.asarray(larger_pred) - np.asarray(nested_pred)) ** 2)
    adjusted = adjusted[np.isfinite(adjusted)]
    if len(adjusted) < 3:
        return {"stat": np.nan, "pvalue": np.nan, "n": len(adjusted)}
    stat = float(adjusted.mean() / np.sqrt(_hac_variance(adjusted) / len(adjusted)))
    try:
        from scipy.stats import norm

        pvalue = float(norm.sf(stat))
    except ImportError:
        pvalue = np.nan
    return {"stat": stat, "pvalue": pvalue, "n": len(adjusted)}


def run_arch_mcs(losses: pd.DataFrame, size: float = 0.05, reps: int = 1000,
                 block_size: int = 1000, method: str = "R", seed: int = C.SEED) -> dict:
    """Executa o MCS oficial de Hansen, Lunde e Nason via ``arch``.

    ``losses`` deve conter uma coluna por modelo e uma linha por data. Como
    no exemplo de referência, usamos perdas MSE, em que valores menores são
    melhores. A função não possui fallback aproximado: uma execução final sem
    ``arch`` instalado deve falhar explicitamente, evitando resultados que
    pareçam um MCS oficial.
    """
    from arch.bootstrap import MCS

    losses = losses.dropna(axis=0, how="any").astype(float)
    if losses.shape[0] < 10 or losses.shape[1] < 2:
        raise ValueError("MCS requer pelo menos 10 observações e 2 modelos completos.")
    mcs = MCS(losses, size=size, reps=reps, block_size=block_size, method=method, seed=seed)
    mcs.compute()
    pvalues = mcs.pvalues.copy()
    return {
        "pvalues": pvalues,
        "included": list(mcs.included),
        "excluded": list(mcs.excluded),
        "losses": losses,
        "size": size,
        "reps": reps,
        "block_size": block_size,
        "method": method,
    }


def _load_forecasts() -> pd.DataFrame:
    paths = [C.FORECAST_BENCH, C.FORECAST_TSMIXERX]
    frames = [pd.read_parquet(p) for p in paths if p.exists()]
    if not frames:
        raise FileNotFoundError("Nenhum arquivo de previsões foi encontrado.")
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["model", "date"]).reset_index(drop=True)


def _metric_table(forecasts: pd.DataFrame, mcs_reps: int = 1000,
                  mcs_block_size: int = 1000, mcs_size: float = 0.05) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    models = sorted(forecasts.model.dropna().unique())
    rw = forecasts.loc[forecasts.model == "RW", ["date", "y_hat"]].rename(columns={"y_hat": "rw_hat"})
    rows = []
    loss_frames = {}
    mcs_loss_frames = {}
    for model in models:
        f = forecasts.loc[forecasts.model == model, ["date", "y", "y_hat"]].copy()
        f = f.merge(rw, on="date", how="left")
        valid = f[["y", "y_hat"]].notna().all(axis=1)
        f = f.loc[valid]
        if f.empty:
            continue
        y, p = f.y.to_numpy(float), f.y_hat.to_numpy(float)
        naive = np.array(f.rw_hat.to_numpy(float), dtype=float, copy=True) if f.rw_hat.notna().all() else np.roll(y, 1)
        if len(naive):
            naive[0] = y[0] if not np.isfinite(naive[0]) else naive[0]
        naive = np.where(np.isfinite(naive), naive, y)
        losses = pointwise_loss(y, p)
        loss_frames[model] = pd.Series(losses, index=f.date.to_numpy())
        # MCS conforme ao exemplo fornecido: perda quadrática por observação.
        mcs_loss_frames[model] = pd.Series((y - p) ** 2, index=f.date.to_numpy())
        rows.append({"model": model, "n": len(y), "QLIKE": qlike(y, p), "MSE": mse(y, p),
                     "MAE": mae(y, p), "R2_oos": r2_oos(y, p, naive),
                     "mean_y_hat": float(np.mean(p))})

    loss_df = pd.concat(loss_frames, axis=1).sort_index() if loss_frames else pd.DataFrame()
    mcs_loss_df = pd.concat(mcs_loss_frames, axis=1).sort_index() if mcs_loss_frames else pd.DataFrame()
    mcs = run_arch_mcs(mcs_loss_df, size=mcs_size, reps=mcs_reps,
                       block_size=mcs_block_size) if not mcs_loss_df.empty else None
    table = pd.DataFrame(rows)
    if mcs is not None:
        table["MCS_arch_included"] = table.model.isin(mcs["included"])
        pvalues = mcs["pvalues"]
        if isinstance(pvalues, pd.DataFrame) and "Pvalue" in pvalues:
            table["MCS_arch_pvalue"] = table.model.map(pvalues["Pvalue"])
        else:
            table["MCS_arch_pvalue"] = np.nan
    else:
        table["MCS_arch_included"] = False
        table["MCS_arch_pvalue"] = np.nan
    return table, loss_df, mcs


def _pairwise_tables(forecasts: pd.DataFrame, models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dm_stat = pd.DataFrame(np.nan, index=models, columns=models)
    dm_p = dm_stat.copy()
    gw_p = dm_stat.copy()
    cw_p = dm_stat.copy()
    for a, b in itertools.permutations(models, 2):
        left = forecasts.loc[forecasts.model == a, ["date", "y", "y_hat"]].rename(columns={"y_hat": "a"})
        right = forecasts.loc[forecasts.model == b, ["date", "y_hat"]].rename(columns={"y_hat": "b"})
        joined = left.merge(right, on="date", how="inner").dropna()
        if len(joined) < 3:
            continue
        y = joined.y.to_numpy(float)
        dm = dm_test(y, joined.a.to_numpy(float), joined.b.to_numpy(float))
        gw = gw_test(y, joined.a.to_numpy(float), joined.b.to_numpy(float))
        cw = clark_west(y, joined.a.to_numpy(float), joined.b.to_numpy(float))
        dm_stat.loc[a, b], dm_p.loc[a, b] = dm["stat"], dm["pvalue"]
        gw_p.loc[a, b], cw_p.loc[a, b] = gw["pvalue"], cw["pvalue"]
    return dm_stat, dm_p, gw_p, cw_p


def _make_figures(forecasts: pd.DataFrame, dm_p: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    pivot = forecasts.pivot_table(index="date", columns="model", values="y_hat", aggfunc="first")
    actual = forecasts.drop_duplicates("date").set_index("date")["y"]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(actual.index, actual, color="black", linewidth=1.6, label="RV_YZ realizado")
    for col in pivot:
        ax.plot(pivot.index, pivot[col], linewidth=0.9, alpha=0.75, label=col)
    ax.set_ylabel("Variância anualizada")
    ax.set_title("Previsões rolling versus RV_YZ")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "forecasts_vs_realized.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    arr = dm_p.to_numpy(float)
    im = ax.imshow(np.nan_to_num(arr, nan=0.5), cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(dm_p.columns)), dm_p.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(dm_p.index)), dm_p.index)
    ax.set_title("p-valores DM, QLIKE")
    fig.colorbar(im, ax=ax, label="p-value")
    fig.tight_layout()
    fig.savefig(output_dir / "dm_heatmap.png", dpi=180)
    plt.close(fig)

    for label, (start, end) in {"gfc_2008": ("2008-09-01", "2009-03-31"),
                                "covid_2020": ("2020-02-01", "2020-12-31")}.items():
        sub = forecasts[forecasts.date.between(start, end)]
        if sub.empty:
            continue
        p = sub.pivot_table(index="date", columns="model", values="y_hat", aggfunc="first")
        y = sub.drop_duplicates("date").set_index("date")["y"]
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(y.index, y, color="black", label="RV_YZ")
        for col in p:
            ax.plot(p.index, p[col], label=col, alpha=0.8)
        ax.set_title(label)
        ax.legend(ncol=3, fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / f"{label}.png", dpi=180)
        plt.close(fig)


def run(output_csv: Path = C.METRICS_CSV, make_figures: bool = True,
        mcs_reps: int = 1000, mcs_block_size: int = 1000,
        mcs_size: float = 0.05) -> tuple[pd.DataFrame, pd.DataFrame]:
    forecasts = _load_forecasts()
    metrics, loss_df, mcs = _metric_table(forecasts, mcs_reps, mcs_block_size, mcs_size)
    models = metrics.model.tolist()
    dm_stat, dm_p, gw_p, cw_p = _pairwise_tables(forecasts, models)
    if mcs is None:
        raise RuntimeError("O MCS oficial não foi calculado.")
    metrics["MCS_arch_included_set"] = ", ".join(map(str, mcs["included"]))
    metrics["MCS_arch_excluded_set"] = ", ".join(map(str, mcs["excluded"]))
    metrics["DM_p_vs_RW"] = metrics.model.map(dm_p["RW"] if "RW" in dm_p else pd.Series(dtype=float))
    metrics["GW_p_vs_RW"] = metrics.model.map(gw_p["RW"] if "RW" in gw_p else pd.Series(dtype=float))
    metrics["CW_p_vs_RW"] = metrics.model.map(cw_p["RW"] if "RW" in cw_p else pd.Series(dtype=float))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_csv, index=False)
    dm_p.to_csv(output_csv.parent / "dm_pvalues.csv")
    gw_p.to_csv(output_csv.parent / "gw_pvalues.csv")
    cw_p.to_csv(output_csv.parent / "cw_pvalues.csv")
    mcs["pvalues"].to_csv(output_csv.parent / "mcs_pvalues.csv")
    pvalues = mcs["pvalues"]
    pvalue_map = pvalues["Pvalue"].to_dict() if isinstance(pvalues, pd.DataFrame) and "Pvalue" in pvalues else {}
    membership = pd.DataFrame({"model": list(mcs["losses"].columns)})
    membership["included"] = membership["model"].isin(mcs["included"])
    membership["status"] = np.where(membership["included"], "included", "excluded")
    membership["pvalue"] = membership["model"].map(pvalue_map)
    membership.to_csv(output_csv.parent / "mcs_membership.csv", index=False)
    (output_csv.parent / "mcs_config.json").write_text(
        pd.Series({"size": mcs["size"], "reps": mcs["reps"],
                   "block_size": mcs["block_size"], "method": mcs["method"],
                   "loss": "MSE", "implementation": "arch.bootstrap.MCS"}).to_json(indent=2),
        encoding="utf-8"
    )
    if make_figures:
        _make_figures(forecasts, dm_p, C.FIGURES_DIR)
    print(f"Métricas salvas em: {output_csv}")
    print(f"MCS arch incluído: {mcs['included']}")
    print(f"MCS arch excluído: {mcs['excluded']}")
    print(metrics.to_string(index=False))
    return metrics, forecasts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=C.METRICS_CSV)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--mcs-reps", type=int, default=1000)
    parser.add_argument("--mcs-block-size", type=int, default=1000)
    parser.add_argument("--mcs-size", type=float, default=0.05)
    args = parser.parse_args()
    run(args.output, make_figures=not args.no_figures,
        mcs_reps=args.mcs_reps, mcs_block_size=args.mcs_block_size,
        mcs_size=args.mcs_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
