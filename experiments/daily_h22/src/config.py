"""Configuração isolada do experimento diário H=22."""

import os
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]

SOURCE_XLSX = REPO_ROOT / "base de dados dissertação.xlsx"
DATA_DIR = Path(os.environ.get("DAILY_H22_DATA_DIR", EXPERIMENT_ROOT / "data"))
DAILY_PANEL = DATA_DIR / "daily_panel.parquet"
SPLIT_MANIFEST = DATA_DIR / "split_manifest.csv"

RESULTS_DIR = Path(os.environ.get("DAILY_H22_RESULTS_DIR", EXPERIMENT_ROOT / "results"))
FORECAST_DIR = RESULTS_DIR / "forecasts"
INPUTS_DIR = RESULTS_DIR / "inputs"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"
HP_DIR = RESULTS_DIR / "hp_search"
HISTORY_DIR = RESULTS_DIR / "training_history"
EXPLAIN_DIR = RESULTS_DIR / "explainability"

TARGET = "rv_yz_22"
HORIZON = 22
RV_WINDOW = 22
ANNUALIZATION = 252

TRAIN_YEARS = 8
VAL_YEARS = 2
TEST_YEARS = 1
STEP_YEARS = 1

# Conversão dos lags mensais acordados {1,3,6,12,24} para pregões.
INPUT_SIZES = [22, 66, 132, 264, 528]

SEED = 42
EPS = 1e-10

PRICE_RETURN_PAIRS = {
    "spx_close": "ret_spx_log",
    "ndx_close": "ret_ndx_log",
    "indu_close": "ret_indu_log",
    "nky_close": "ret_nky_log",
    "ukx_close": "ret_ukx_log",
    "wti_close": "ret_wti_log",
    "gold_close": "ret_gold_log",
    "dxy": "ret_dxy_log",
}

MONTHLY_MACRO = ["industrial_production", "cpi", "unemployment"]
WEEKLY_MACRO = ["stress_index"]

# VIX3M é deliberadamente excluído da análise principal 2000-2025.
EXCLUDED_MAIN = {
    "vix3m",
    "retorno_acum_22d",
    "rv_semanal",
    "rv_mensal",
    "rolling_std_22d",
}

NON_FEATURE_COLUMNS = {
    "date",
    "unique_id",
    TARGET,
    "close_variance_forward_22",
    "close_volatility_forward_22",
}
