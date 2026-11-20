"""Rebuild the 8/2/1 forecast panel after auditing the candidate features."""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DAILY_SRC = ROOT / "experiments/daily_h22/src"
if str(DAILY_SRC) not in sys.path:
    sys.path.insert(0, str(DAILY_SRC))

import config as C
from utils import rolling_821_windows, write_split_manifest

RESULTS = HERE / "results"
EXCLUDED = ("ret_nky_simple", "ret_ukx_simple", "ret_wti_simple")
EXCLUDED_FEATURES = (*EXCLUDED, "vix_term_spread")


def followup_windows(panel: pd.DataFrame) -> list[dict]:
    """Purge train/validation targets and generate test predictions through the last close."""
    dates = pd.to_datetime(panel["date"])
    y = panel[C.TARGET].to_numpy(float)
    windows = rolling_821_windows(panel, purge_boundaries=True)
    for window in windows:
        window["test_idx"] = np.flatnonzero(
            (dates.dt.year.to_numpy() == window["test_year"]) & np.isfinite(y) & (y > 0)
        )
    if not windows or windows[-1]["test_year"] != int(dates.dt.year.max()):
        raise ValueError("Last calendar year has no admissible 8/2/1 fold")
    return windows


def prepare(workbook: Path, output: Path) -> pd.DataFrame:
    prep = import_module("10_data_prep_daily")
    daily = pd.read_excel(workbook, sheet_name="painel_diario")
    macro = pd.read_excel(workbook, sheet_name="macro")
    suspect = pd.concat(
        [
            daily.loc[daily[column].le(-1), ["date", column]]
            .rename(columns={column: "value"})
            .assign(feature=column)
            for column in EXCLUDED
        ],
        ignore_index=True,
    )
    # The term spread exactly reconstructs the excluded VIX3M from the VIX.
    clean = prep._clean_daily(daily.drop(columns=list(EXCLUDED_FEATURES)))
    panel = prep._merge_macro_conservative(clean, macro)
    panel[C.TARGET] = prep.compute_yang_zhang(panel)
    panel = prep.add_forward_close_variance(panel)
    panel.insert(1, "unique_id", "sp500")
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output, index=False)
    split = write_split_manifest(panel, followup_windows(panel), output.parent / "split_manifest.csv")
    suspect.groupby("feature").agg(
        values_le_minus_one=("value", "size"),
        values_equal_minus_one=("value", lambda x: x.eq(-1).sum()),
        minimum=("value", "min"),
    ).to_csv(output.parent / "invalid_simple_returns.csv")
    print(f"{len(panel)} sessions, {len(split)} folds, through {panel.date.max().date()}", flush=True)
    return panel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=C.SOURCE_XLSX)
    parser.add_argument("--output", type=Path, default=RESULTS / "panel.parquet")
    args = parser.parse_args()
    prepare(args.workbook, args.output)
