"""Seleção de features ajustada exclusivamente nas origens de treino."""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
from utils import feature_columns, target_matrix


def rank_features(panel: pd.DataFrame, train_origins: np.ndarray,
                  missing_threshold: float = 0.25,
                  corr_threshold: float = 0.99) -> pd.DataFrame:
    candidates = [c for c in feature_columns(panel) if c != C.TARGET]
    x = panel.loc[train_origins, candidates].apply(pd.to_numeric, errors="coerce")
    missing = x.isna().mean()
    variance = x.var(ddof=0)
    eligible = [c for c in candidates if missing[c] <= missing_threshold
                and np.isfinite(variance[c]) and variance[c] > 1e-12]

    # Remove pares quase determinísticos, mantendo a maior variância.
    corr = x[eligible].corr().abs()
    drop: set[str] = set()
    for i, left in enumerate(eligible):
        if left in drop:
            continue
        for right in eligible[i + 1:]:
            if right in drop:
                continue
            value = corr.loc[left, right]
            if np.isfinite(value) and value > corr_threshold:
                drop.add(left if variance[left] < variance[right] else right)
    filtered = [c for c in eligible if c not in drop]

    y_h22 = target_matrix(panel, train_origins)[:, -1]
    raw = x[filtered]
    medians = raw.median().fillna(0.0)
    matrix = raw.fillna(medians).to_numpy(float)
    try:
        from sklearn.feature_selection import f_regression

        score, pvalue = f_regression(matrix, y_h22)
    except ImportError:
        score = np.zeros(len(filtered))
        pvalue = np.full(len(filtered), np.nan)
    abs_corr = []
    for j in range(matrix.shape[1]):
        value = np.corrcoef(matrix[:, j], y_h22)[0, 1] if np.std(matrix[:, j]) > 0 else 0.0
        abs_corr.append(abs(value) if np.isfinite(value) else 0.0)

    ranking = pd.DataFrame({
        "feature": filtered,
        "f_score_h22": score,
        "pvalue_h22": pvalue,
        "abs_corr_h22": abs_corr,
        "missing_rate_train": [missing[c] for c in filtered],
        "variance_train": [variance[c] for c in filtered],
    }).sort_values(["f_score_h22", "abs_corr_h22"], ascending=False).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    ranking.attrs["counts"] = {
        "raw_numeric": len(candidates),
        "after_missing_variance": len(eligible),
        "after_collinearity": len(filtered),
    }
    return ranking


def select_features(panel: pd.DataFrame, train_origins: np.ndarray,
                    top_k: int | str) -> tuple[list[str], pd.DataFrame]:
    ranking = rank_features(panel, train_origins)
    if top_k == "all":
        selected_exog = ranking.feature.tolist()
    else:
        selected_exog = ranking.head(min(int(top_k), len(ranking))).feature.tolist()
    ranking["selected"] = ranking.feature.isin(selected_exog)
    # A própria RV histórica é sempre o primeiro canal autorregressivo.
    return [C.TARGET, *selected_exog], ranking

