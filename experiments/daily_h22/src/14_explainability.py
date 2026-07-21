"""Atribuições temporais e visualizações da seleção de features."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from models import DirectVolatilityRegressor
from utils import build_sequences, ensure_directories, rolling_821_windows, save_json

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_daily_h22"


def _family(feature: str) -> str:
    name = feature.lower()
    if name.startswith("ret_"):
        return "retornos"
    if any(key in name for key in ["rv_", "parkinson", "garman", "rogers", "atr", "true_range", "vix"]):
        return "volatilidade"
    if any(key in name for key in ["rsi", "macd", "roc", "adx", "bollinger", "obv", "volume"]):
        return "técnicas"
    if any(key in name for key in ["cpi", "unemployment", "industrial", "epu", "stress"]):
        return "macro"
    if any(key in name for key in ["ust", "fed_funds", "baa", "aaa", "yield"]):
        return "juros/crédito"
    if any(key in name for key in ["ndx", "indu", "nky", "ukx", "wti", "gold", "dxy"]):
        return "internacionais"
    return "mercado doméstico"


def feature_selection_figures() -> None:
    import matplotlib.pyplot as plt

    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    best_path = C.HP_DIR / "feature_ranking_best.csv"
    if best_path.exists():
        ranking = pd.read_csv(best_path)
        top = ranking.head(30).sort_values("f_score_h22")
        fig, ax = plt.subplots(figsize=(10, 9))
        colors = np.where(top.selected, "tab:blue", "lightgray")
        ax.barh(top.feature, top.f_score_h22, color=colors)
        ax.set_title("Ranking de features no treino inicial, target h=22")
        ax.set_xlabel("F-score")
        fig.tight_layout()
        fig.savefig(C.FIGURES_DIR / "feature_ranking_top30.png", dpi=180)
        plt.close(fig)

        selected = ranking.loc[ranking.selected].copy()
        selected["family"] = selected.feature.map(_family)
        counts = selected.family.value_counts().sort_values()
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(counts.index, counts.values, color="tab:green")
        ax.set_title("Features selecionadas por família")
        ax.set_xlabel("Quantidade")
        fig.tight_layout()
        fig.savefig(C.FIGURES_DIR / "feature_family_counts.png", dpi=180)
        plt.close(fig)

        if C.DAILY_PANEL.exists():
            panel = pd.read_parquet(C.DAILY_PANEL).sort_values("date").reset_index(drop=True)
            first = rolling_821_windows(panel)[0]
            corr_features = ranking.head(min(30, len(ranking))).feature.tolist()
            corr = panel.loc[first["train_idx"], corr_features].corr().fillna(0.0)
            try:
                from scipy.cluster.hierarchy import leaves_list, linkage
                from scipy.spatial.distance import squareform

                distance = (1.0 - corr.abs()).clip(0, 1)
                np.fill_diagonal(distance.values, 0.0)
                order = leaves_list(linkage(squareform(distance.values, checks=False), method="average"))
                corr_features = [corr_features[i] for i in order]
                corr = corr.loc[corr_features, corr_features]
            except (ImportError, ValueError):
                pass
            fig, ax = plt.subplots(figsize=(13, 11))
            image = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr_features)), corr_features, rotation=90, fontsize=7)
            ax.set_yticks(range(len(corr_features)), corr_features, fontsize=7)
            ax.set_title("Correlação das 30 features mais bem ranqueadas, treino inicial")
            fig.colorbar(image, ax=ax, label="Correlação")
            fig.tight_layout()
            fig.savefig(C.FIGURES_DIR / "feature_correlation_clusters.png", dpi=180)
            plt.close(fig)

    count_path = C.HP_DIR / "feature_selection_counts.json"
    if count_path.exists():
        counts = json.loads(count_path.read_text())
        if counts:
            labels = ["Numéricas", "Após missing/variância", "Após colinearidade"]
            values = [counts.get("raw_numeric", 0), counts.get("after_missing_variance", 0),
                      counts.get("after_collinearity", 0)]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(labels, values, marker="o", linewidth=2)
            for x, value in enumerate(values):
                ax.text(x, value, str(value), ha="center", va="bottom")
            ax.set_ylabel("Número de features")
            ax.set_title("Fluxo do pré-filtro de features")
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(C.FIGURES_DIR / "feature_selection_flow.png", dpi=180)
            plt.close(fig)

    ranking_files = sorted((C.RESULTS_DIR / "feature_selection").glob("ranking_tsmixerx_window_*.csv"))
    if ranking_files:
        records = []
        for path in ranking_files:
            frame = pd.read_csv(path)
            window = int(path.stem.rsplit("_", 1)[-1])
            records.extend({"feature": row.feature, "window_id": window,
                            "selected": bool(row.selected)} for row in frame.itertuples())
        stability = pd.DataFrame(records).groupby("feature").agg(
            windows_available=("window_id", "nunique"), selected_windows=("selected", "sum")).reset_index()
        stability["selection_rate"] = stability.selected_windows / stability.windows_available
        stability = stability.sort_values(["selection_rate", "selected_windows"], ascending=False)
        stability.to_csv(C.RESULTS_DIR / "feature_selection" / "selection_stability.csv", index=False)
        top = stability.head(30).sort_values("selection_rate")
        fig, ax = plt.subplots(figsize=(10, 9))
        ax.barh(top.feature, top.selection_rate, color="tab:orange")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Fração das janelas")
        ax.set_title("Estabilidade da seleção de features")
        fig.tight_layout()
        fig.savefig(C.FIGURES_DIR / "feature_selection_stability.png", dpi=180)
        plt.close(fig)


def _integrated_gradients(estimator: DirectVolatilityRegressor, x: np.ndarray,
                          features: list[str], origins: np.ndarray,
                          panel: pd.DataFrame, horizons: list[int],
                          n_steps: int) -> pd.DataFrame:
    import torch
    from captum.attr import IntegratedGradients

    scaled_x = estimator._scale_x(x)
    rows = []

    class Wrapper(torch.nn.Module):
        def __init__(self, horizon_idx: int):
            super().__init__()
            self.horizon_idx = horizon_idx

        def forward(self, values):
            scaled = estimator.model(values)[:, self.horizon_idx]
            center = torch.as_tensor(estimator.y_center[self.horizon_idx], device=values.device)
            scale = torch.as_tensor(estimator.y_scale[self.horizon_idx], device=values.device)
            return torch.exp(torch.clamp(scaled * scale + center, -30, 10))

    for sample_idx, origin in enumerate(origins):
        values = torch.tensor(scaled_x[sample_idx:sample_idx + 1], device=estimator.device,
                              requires_grad=True)
        baseline = torch.zeros_like(values)
        for horizon in horizons:
            explainer = IntegratedGradients(Wrapper(horizon - 1))
            attribution, delta = explainer.attribute(values, baselines=baseline,
                                                      n_steps=n_steps,
                                                      return_convergence_delta=True)
            arr = attribution.detach().cpu().numpy()[0]
            for time_idx in range(arr.shape[0]):
                lag = arr.shape[0] - 1 - time_idx
                for feature_idx, feature in enumerate(features):
                    rows.append({
                        "method": "IntegratedGradients",
                        "origin_date": panel.loc[int(origin), "date"],
                        "horizon": horizon,
                        "lag": lag,
                        "feature": feature,
                        "attribution": float(arr[time_idx, feature_idx]),
                        "abs_attribution": float(abs(arr[time_idx, feature_idx])),
                        "convergence_delta": float(delta.detach().cpu().numpy().ravel()[0]),
                    })
    return pd.DataFrame(rows)


def _grouped_shapley(estimator: DirectVolatilityRegressor, x: np.ndarray,
                     features: list[str], origin_date, horizons: list[int],
                     n_samples: int) -> pd.DataFrame:
    """Shapley Value Sampling agrupado separadamente por feature e por lag."""
    import torch
    from captum.attr import ShapleyValueSampling

    values = torch.tensor(estimator._scale_x(x[:1]), device=estimator.device)
    baseline = torch.zeros_like(values)
    rows = []

    class Wrapper(torch.nn.Module):
        def __init__(self, horizon_idx: int):
            super().__init__()
            self.horizon_idx = horizon_idx

        def forward(self, z):
            scaled = estimator.model(z)[:, self.horizon_idx]
            center = torch.as_tensor(estimator.y_center[self.horizon_idx], device=z.device)
            scale = torch.as_tensor(estimator.y_scale[self.horizon_idx], device=z.device)
            return torch.exp(torch.clamp(scaled * scale + center, -30, 10))

    feature_mask = torch.arange(len(features), device=estimator.device).view(1, 1, -1).expand_as(values).clone()
    lag_mask = torch.arange(values.shape[1], device=estimator.device).view(1, -1, 1).expand_as(values).clone()
    for horizon in horizons:
        explainer = ShapleyValueSampling(Wrapper(horizon - 1))
        feature_attr = explainer.attribute(values, baselines=baseline,
                                           feature_mask=feature_mask, n_samples=n_samples)
        # Todos os elementos do mesmo grupo recebem o mesmo valor; usamos o
        # primeiro elemento para não multiplicar a contribuição pelo grupo.
        for j, feature in enumerate(features):
            value = float(feature_attr[0, 0, j].detach().cpu())
            rows.append({"method": "ShapleyValueSampling_feature_group",
                         "origin_date": origin_date, "horizon": horizon,
                         "lag": np.nan, "feature": feature,
                         "attribution": value, "abs_attribution": abs(value),
                         "convergence_delta": np.nan})
        lag_attr = explainer.attribute(values, baselines=baseline,
                                       feature_mask=lag_mask, n_samples=n_samples)
        for t in range(values.shape[1]):
            lag = values.shape[1] - 1 - t
            value = float(lag_attr[0, t, 0].detach().cpu())
            rows.append({"method": "ShapleyValueSampling_lag_group",
                         "origin_date": origin_date, "horizon": horizon,
                         "lag": lag, "feature": "ALL_FEATURES",
                         "attribution": value, "abs_attribution": abs(value),
                         "convergence_delta": np.nan})
    return pd.DataFrame(rows)


def _explanation_figures(attributions: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    ig = attributions.loc[attributions.method == "IntegratedGradients"]
    if ig.empty:
        return
    feature_global = ig.groupby(["horizon", "feature"]).abs_attribution.mean().reset_index()
    feature_global.to_csv(C.EXPLAIN_DIR / "integrated_gradients_feature_horizon.csv", index=False)
    for horizon in sorted(feature_global.horizon.unique()):
        top = feature_global.loc[feature_global.horizon == horizon].nlargest(20, "abs_attribution").sort_values("abs_attribution")
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.barh(top.feature, top.abs_attribution, color="tab:blue")
        ax.set_title(f"Importância temporal global, horizonte {horizon}")
        ax.set_xlabel("|Integrated Gradients| médio")
        fig.tight_layout()
        fig.savefig(C.FIGURES_DIR / f"temporal_importance_h{int(horizon):02d}.png", dpi=180)
        plt.close(fig)

    lag_global = ig.groupby(["horizon", "lag"]).abs_attribution.mean().reset_index()
    lag_global.to_csv(C.EXPLAIN_DIR / "integrated_gradients_lag_horizon.csv", index=False)
    pivot = lag_global.pivot(index="lag", columns="horizon", values="abs_attribution").sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 10))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="magma")
    ax.set_xticks(range(len(pivot.columns)), [f"h={int(h)}" for h in pivot.columns])
    step = max(1, len(pivot) // 12)
    ticks = np.arange(0, len(pivot), step)
    ax.set_yticks(ticks, pivot.index.to_numpy()[ticks])
    ax.set_ylabel("Lag (pregões)")
    ax.set_title("Importância temporal por lag e horizonte")
    fig.colorbar(image, ax=ax, label="|atribuição| média")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "temporal_lag_horizon_heatmap.png", dpi=180)
    plt.close(fig)


def run(window_ids: list[int] | None = None, origins_per_window: int = 3,
        horizons: list[int] | None = None, n_steps: int = 32,
        run_shapley: bool = False, shapley_samples: int = 25,
        smoke: bool = False) -> None:
    ensure_directories()
    feature_selection_figures()
    panel = pd.read_parquet(C.DAILY_PANEL).sort_values("date").reset_index(drop=True)
    windows = {w["window_id"]: w for w in rolling_821_windows(panel)}
    artifacts = sorted((C.MODELS_DIR / "tsmixerx").glob("tsmixerx_window_*.pt"))
    if window_ids is not None:
        wanted = set(window_ids)
        artifacts = [p for p in artifacts if int(p.stem.rsplit("_", 1)[-1]) in wanted]
    if smoke:
        artifacts = artifacts[:1]
        origins_per_window = 1
        n_steps = min(n_steps, 8)
    horizons = horizons or [1, 5, 10, 22]
    all_frames = []
    for artifact in artifacts:
        window_id = int(artifact.stem.rsplit("_", 1)[-1])
        w = windows[window_id]
        estimator, payload = DirectVolatilityRegressor.load(artifact)
        features = payload["features"]
        test_x, test_y, origins = build_sequences(panel, w["test_idx"], features, estimator.input_size)
        if not len(origins):
            continue
        # Primeiro ponto, maior RV h=22 e ponto mediano formam uma amostra
        # simples de regimes sem usar os valores para ajustar o modelo.
        candidates = [0, int(np.argmax(test_y[:, -1])), len(origins) // 2, len(origins) - 1]
        selected_idx = np.unique(candidates)[:origins_per_window]
        selected_x, selected_origins = test_x[selected_idx], origins[selected_idx]
        ig = _integrated_gradients(estimator, selected_x, features, selected_origins,
                                   panel, horizons, n_steps)
        ig["window_id"] = window_id
        all_frames.append(ig)
        if run_shapley:
            shap = _grouped_shapley(estimator, selected_x[:1], features,
                                    panel.loc[int(selected_origins[0]), "date"],
                                    [1, 22], shapley_samples)
            shap["window_id"] = window_id
            all_frames.append(shap)
    if not all_frames:
        print("Nenhum artefato TSMixerX disponível; visualizações de seleção foram mantidas.")
        return
    attributions = pd.concat(all_frames, ignore_index=True)
    attributions.to_parquet(C.EXPLAIN_DIR / "temporal_attributions.parquet", index=False)
    attributions.to_csv(C.EXPLAIN_DIR / "temporal_attributions.csv", index=False)
    _explanation_figures(attributions)
    save_json({"methods": sorted(attributions.method.unique()),
               "horizons_integrated_gradients": horizons,
               "shapley_horizons": [1, 22] if run_shapley else [],
               "windows": sorted(attributions.window_id.unique().tolist()),
               "origins": int(attributions.origin_date.nunique()),
               "integrated_gradients_steps": n_steps,
               "shapley_samples": shapley_samples if run_shapley else 0},
              C.EXPLAIN_DIR / "explainability_manifest.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-ids", type=int, nargs="*")
    parser.add_argument("--origins-per-window", type=int, default=3)
    parser.add_argument("--horizons", type=int, nargs="*", default=[1, 5, 10, 22])
    parser.add_argument("--n-steps", type=int, default=32)
    parser.add_argument("--run-shapley", action="store_true")
    parser.add_argument("--shapley-samples", type=int, default=25)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.window_ids, args.origins_per_window, args.horizons, args.n_steps,
        args.run_shapley, args.shapley_samples, args.smoke)
    return 0


if __name__ == "__main__":
    sys.exit(main())
