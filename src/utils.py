"""Funções compartilhadas do pipeline de previsão de volatilidade."""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


# ------------------------------ RV de Yang-Zhang ------------------------------
def compute_yz_rv(ohlc: pd.DataFrame, window: int = C.RV_WINDOW) -> pd.Series:
    """Calcula a variância realizada Yang-Zhang anualizada.

    A implementação segue a parametrização definida para este projeto:

    ``YZ = O + k*C + (1-k)*N`` e
    ``k = 1 / (1 + O/N + C/N)``,

    em que ``O`` é a variância open-to-open, ``C`` é a variância
    close-to-close e ``N`` é a variância overnight. A série é anualizada por
    252. Valores inválidos de OHLC produzem ``NaN`` e são tratados no
    agregador, nunca substituídos por zero.
    """
    required = ["spx_open", "spx_high", "spx_low", "spx_close"]
    missing = [c for c in required if c not in ohlc]
    if missing:
        raise KeyError(f"Colunas OHLC ausentes: {missing}")

    df = ohlc[required].astype(float).copy()
    positive = (df > 0).all(axis=1)
    df.loc[~positive, required] = np.nan

    # Retornos definidos na mesma convenção usada no painel.
    log_oo = np.log(df["spx_open"] / df["spx_open"].shift(1))
    log_cc = np.log(df["spx_close"] / df["spx_close"].shift(1))
    log_no = np.log(df["spx_open"] / df["spx_close"].shift(1))

    var_o = log_oo.rolling(window, min_periods=window).var(ddof=1)
    var_c = log_cc.rolling(window, min_periods=window).var(ddof=1)
    var_n = log_no.rolling(window, min_periods=window).var(ddof=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        k = 1.0 / (1.0 + var_o / var_n + var_c / var_n)
        yz = var_o + k * var_c + (1.0 - k) * var_n

    yz = yz.where(np.isfinite(yz) & (yz >= 0))
    return (yz * 252.0).rename("rv_yz")


# ------------------------------ Agregacao diaria -> mensal (22 pregões) --------
def aggregate_monthly(
    daily: pd.DataFrame,
    block_size: int = C.BLOCK_SIZE,
    macro: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Agrega o painel diário em blocos não sobrepostos de 22 pregões.

    O bloco ``t`` usa apenas observações até sua última data. O alvo é a
    média da RV_YZ dos 22 pregões estritamente posteriores ao bloco. A
    utilização de ``merge_asof`` para macroeconômicas garante que uma
    observação futura não seja usada como feature.
    """
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    daily["rv_yz"] = compute_yz_rv(daily)

    # Fills autorizados somente para níveis de séries externas. Não fazemos
    # fill no OHLC, pois isso poderia criar retornos artificiais no alvo.
    fill_cols = ["nky_close", "ukx_close", "baa", "aaa"]
    for col in fill_cols:
        if col in daily:
            daily[col] = daily[col].ffill()

    daily["block_id"] = np.arange(len(daily), dtype=int) // block_size
    blocks: list[dict] = []
    for bid, g in daily.groupby("block_id", sort=True):
        end_idx = int(g.index[-1])
        fwd = daily["rv_yz"].iloc[end_idx + 1 : end_idx + 1 + block_size]
        row: dict = {
            "date": g["date"].iloc[-1],
            "block_id": int(bid),
            C.TARGET: float(fwd.mean()) if len(fwd) == block_size and fwd.notna().all() else np.nan,
        }

        for col in C.EXOG_CONTINUOUS:
            if col not in g:
                continue
            vals = pd.to_numeric(g[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else np.nan
            # Mantemos std como NaN para um bloco com uma única observação.
            row[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
        for col in C.EXOG_FLOW:
            if col not in g:
                continue
            vals = pd.to_numeric(g[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else np.nan
        blocks.append(row)

    monthly = pd.DataFrame(blocks)
    monthly = monthly.dropna(subset=[C.TARGET]).reset_index(drop=True)

    if macro is not None and len(macro):
        monthly = merge_macro(monthly, macro)

    monthly.insert(1, "unique_id", "sp500")
    return monthly


def merge_macro(monthly: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Mescla macro mensal/semanal por última observação disponível."""
    out = monthly.sort_values("date").reset_index(drop=True).copy()
    m = macro.copy()
    for col in ["date_ref_mensal", "date_ref_semanal"]:
        if col in m:
            m[col] = pd.to_datetime(m[col], errors="coerce")

    if "date_ref_mensal" in m:
        cols = ["date_ref_mensal", *[c for c in C.MACRO_MONTHLY if c in m]]
        mm = m[cols].dropna(subset=["date_ref_mensal"]).sort_values("date_ref_mensal")
        mm = mm.drop_duplicates("date_ref_mensal", keep="last")
        value_cols = [c for c in C.MACRO_MONTHLY if c in mm]
        if value_cols:
            # Alguns meses têm publicação ausente na planilha. Carregamos a
            # última informação disponível, sem usar qualquer data futura.
            mm[value_cols] = mm[value_cols].ffill()
        out = pd.merge_asof(out, mm, left_on="date", right_on="date_ref_mensal", direction="backward")
        out = out.drop(columns=["date_ref_mensal"], errors="ignore")

    if "date_ref_semanal" in m and "stress_index" in m:
        mw = m[["date_ref_semanal", "stress_index"]].dropna(subset=["date_ref_semanal"])
        mw = mw.sort_values("date_ref_semanal").drop_duplicates("date_ref_semanal", keep="last")
        out = pd.merge_asof(out.sort_values("date"), mw, left_on="date", right_on="date_ref_semanal", direction="backward")
        out = out.drop(columns=["date_ref_semanal"], errors="ignore")

    return out


# ------------------------------ Splits temporais ------------------------------
def hp_split(monthly: pd.DataFrame, train_years: int = C.TRAIN_YEARS,
             val_years: int = C.VAL_YEARS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide por anos-calendário, sem assumir 11 blocos por ano."""
    monthly = monthly.sort_values("date").reset_index(drop=True)
    first_year = int(monthly["date"].dt.year.min())
    train_year_end = first_year + train_years
    val_year_end = train_year_end + val_years
    years = monthly["date"].dt.year
    train = monthly.loc[years < train_year_end].copy()
    val = monthly.loc[(years >= train_year_end) & (years < val_year_end)].copy()
    return train, val


def rolling_indices(monthly: pd.DataFrame) -> list[dict]:
    """Gera indices para rolling 4+1+1 step anual no periodo de teste.

    Cada iteracao:
      train: 4 anos de blocos anteriores
      val:   1 ano de blocos seguintes
      test:  1 ano de blocos seguintes
      step:  1 ano (desloca t_start)
    Retorna lista de dicts {train_idx, val_idx, test_idx, test_start_date}.
    """
    monthly = monthly.sort_values("date").reset_index(drop=True)
    first_year = int(monthly["date"].dt.year.min())
    last_year = int(monthly["date"].dt.year.max())
    first_val_year = first_year + C.HP_TOTAL_YEARS
    iters = []
    for val_year in range(first_val_year, last_year - C.ROLL_TEST_YEARS + 1, C.ROLL_STEP_YEARS):
        train_start = val_year - C.ROLL_TRAIN_YEARS
        val_end = val_year + C.ROLL_VAL_YEARS
        test_end = val_end + C.ROLL_TEST_YEARS
        years = monthly["date"].dt.year
        train_pos = np.flatnonzero((years >= train_start) & (years < val_year))
        val_pos = np.flatnonzero((years >= val_year) & (years < val_end))
        test_pos = np.flatnonzero((years >= val_end) & (years < test_end))
        if len(train_pos) and len(val_pos) and len(test_pos):
            iters.append({
                "train_idx": (int(train_pos[0]), int(train_pos[-1]) + 1),
                "val_idx": (int(val_pos[0]), int(val_pos[-1]) + 1),
                "test_idx": (int(test_pos[0]), int(test_pos[-1]) + 1),
                "val_year": val_year,
                "test_start": monthly["date"].iloc[test_pos[0]],
            })
    return iters


# ------------------------------ Metricas de previsao --------------------------
def qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """QLIKE (log scoring rule) para RV, Patton (2011).
    QLIKE = mean( y_true/y_pred - log(y_true/y_pred) - 1 )
    Robusto a zero (desde que y_pred > 0)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    eps = 1e-12
    y_pred = np.maximum(y_pred, eps)
    y_true = np.maximum(y_true, eps)
    return float(np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_oos(y_true: np.ndarray, y_pred: np.ndarray, y_naive: np.ndarray | None = None) -> float:
    """R2 out-of-sample vs random walk naive (y_t = y_{t-1}).
    R2_oos = 1 - sum((y - yhat)^2) / sum((y - y_naive)^2)."""
    if y_naive is None:
        y_naive = np.roll(y_true, 1)
        y_naive[0] = y_true[0]
    num = np.sum((y_true - y_pred) ** 2)
    den = np.sum((y_true - y_naive) ** 2)
    if den == 0:
        return float("nan")
    return float(1 - num / den)


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, y_naive: np.ndarray | None = None) -> dict:
    return {
        "QLIKE": qlike(y_true, y_pred),
        "MSE": mse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "R2_oos": r2_oos(y_true, y_pred, y_naive),
    }


# ------------------------------ Feature pre-filter -----------------------------
def filter_collinear(features: pd.DataFrame, corr_threshold: float = 0.99,
                     var_threshold: float = 1e-8) -> list[str]:
    """Remove features quase-constantes e quase-deterministicas (corr > threshold)."""
    var = features.var(axis=0)
    keep = var[var > var_threshold].index.tolist()
    if not keep:
        return []
    sub = features[keep]
    corr = sub.corr().abs()
    # Cluster por correlacao: dropar a de menor variancia dentro de pares > threshold
    drop = set()
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if cols[j] in drop or cols[i] in drop:
                continue
            if corr.iloc[i, j] > corr_threshold:
                # dropar a de menor variancia
                if var[cols[i]] < var[cols[j]]:
                    drop.add(cols[i])
                else:
                    drop.add(cols[j])
    return [c for c in keep if c not in drop]
