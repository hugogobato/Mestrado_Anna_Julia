"""Diagnóstico opcional de colinearidade e ranking supervisionado.

A seleção usada pelo TSMixerX continua integrada à busca de HP em
``03_tsmixerx.py``. Este arquivo apenas gera uma tabela auditável para o
paper, sempre ajustando o ranking no primeiro bloco temporal de treino.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from utils import filter_collinear, hp_split


def run(panel_path: Path = C.DATA_PARQUET, output: Path = C.RESULTS_DIR / "feature_ranking.csv") -> pd.DataFrame:
    panel = pd.read_parquet(panel_path).sort_values("date").reset_index(drop=True)
    train, _ = hp_split(panel)
    train_idx = train.index.to_numpy(dtype=int)
    excluded = {"date", "unique_id", "block_id", C.TARGET}
    numeric = [c for c in panel if c not in excluded and pd.api.types.is_numeric_dtype(panel[c])]
    keep = filter_collinear(panel.loc[train_idx, numeric])
    raw = panel[keep].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = raw.iloc[train_idx].median().fillna(0.0)
    x = raw.fillna(med).to_numpy(float)
    y = panel[C.TARGET].to_numpy(float)
    try:
        from sklearn.feature_selection import f_regression

        score, pvalue = f_regression(x[train_idx], y[train_idx])
    except ImportError:
        score = np.zeros(len(keep))
        pvalue = np.full(len(keep), np.nan)
    ranking = pd.DataFrame({"feature": keep, "f_score": score, "pvalue": pvalue})
    ranking["abs_corr_target"] = [abs(np.corrcoef(x[train_idx, i], y[train_idx])[0, 1])
                                   if np.std(x[train_idx, i]) > 0 else 0.0
                                   for i in range(x.shape[1])]
    ranking = ranking.sort_values(["f_score", "abs_corr_target"], ascending=False).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output, index=False)
    print(f"Ranking salvo em: {output}; {len(ranking)} features após pré-filtro")
    print(ranking.head(20).to_string(index=False))
    return ranking


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=C.DATA_PARQUET)
    parser.add_argument("--output", type=Path, default=C.RESULTS_DIR / "feature_ranking.csv")
    args = parser.parse_args()
    run(args.panel, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
