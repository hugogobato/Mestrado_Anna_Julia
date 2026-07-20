"""Constantes compartilhadas do pipeline de previsao de volatilidade."""

from pathlib import Path

# ----- Paths -----
ROOT = Path(__file__).resolve().parent.parent
DATA_XLSX = ROOT / "base de dados dissertação.xlsx"
DATA_PARQUET = ROOT / "data" / "monthly_panel.parquet"
RESULTS_DIR = ROOT / "results"
HP_SEARCH_DIR = RESULTS_DIR / "hp_search"
FIGURES_DIR = ROOT / "results" / "figures"
MODELS_DIR = RESULTS_DIR / "models"
TSMIXERX_MODELS_DIR = MODELS_DIR / "tsmixerx"
BENCHMARK_MODELS_DIR = MODELS_DIR / "benchmarks"
FORECAST_BENCH = RESULTS_DIR / "forecasts_benchmarks.parquet"
FORECAST_TSMIXERX = RESULTS_DIR / "forecasts_tsmixerx.parquet"
METRICS_CSV = RESULTS_DIR / "metrics.csv"

# ----- Horizonte e janela -----
BLOCK_SIZE = 22
HORIZON = 1
RV_WINDOW = 22

# ----- Lags do TSMixerX (em blocos mensais) -----
LAGS = [1, 3, 6, 12, 24]

# ----- Esquema de janelas -----
TRAIN_YEARS = 8
VAL_YEARS = 2
HP_TOTAL_YEARS = TRAIN_YEARS + VAL_YEARS

ROLL_TRAIN_YEARS = 4
ROLL_VAL_YEARS = 1
ROLL_TEST_YEARS = 1
ROLL_STEP_YEARS = 1
# Apenas uma aproximação para mensagens e compatibilidade. Os cortes do
# pipeline usam anos-calendário diretamente, pois 22 pregões não equivalem
# exatamente a 1/12 de um ano.
BLOCKS_PER_YEAR = 252 // BLOCK_SIZE

# ----- Colunas -----
OHLC = ["spx_open", "spx_high", "spx_low", "spx_close"]

EXOG_CONTINUOUS = [
    "spx_close", "spy_volume", "vix", "vix3m",
    "fed_funds", "ust_3m", "ust_10y", "baa", "aaa",
    "ndx_close", "indu_close", "nky_close", "ukx_close",
    "wti_close", "gold_close", "dxy",
    "ret_spx_log", "ret_ndx_log", "ret_indu_log", "ret_nky_log", "ret_ukx_log",
    "ret_wti_log", "ret_gold_log", "ret_dxy_log",
    "rv_diaria", "parkinson", "garman_klass", "rogers_satchell",
    "true_range", "atr_14", "volume_medio_22d",
    "rsi_14", "macd", "roc_10", "adx_14", "bollinger_width_20",
    "vix_term_spread", "yield_slope", "baa_aaa_spread",
    "epu",
]

EXOG_FLOW = ["obv"]

MACRO_MONTHLY = ["industrial_production", "cpi", "unemployment"]
MACRO_WEEKLY = ["stress_index"]

TARGET = "rv_yz_forward"

SEED = 42

N_TRIALS_DEFAULT = 100
MAX_STEPS_DEFAULT = 1000
EARLY_STOP_PATIENCE = 10
