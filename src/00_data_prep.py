"""Prepara o painel diário e cria o painel mensal operacional.

Uso:
    python src/00_data_prep.py
    python src/00_data_prep.py --smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from utils import aggregate_monthly


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


def _read_input(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"Base de dados não encontrada: {path}")
    daily = pd.read_excel(path, sheet_name="painel_diario")
    macro = pd.read_excel(path, sheet_name="macro")
    return daily, macro


def _clean_daily(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily.columns = [str(c).strip() for c in daily.columns]
    daily = daily.drop(columns=[c for c in daily if c.lower().startswith("unnamed")], errors="ignore")
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily = daily.dropna(subset=["date"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)

    missing_ohlc = [c for c in C.OHLC if c not in daily]
    if missing_ohlc:
        raise KeyError(f"OHLC ausentes no painel diário: {missing_ohlc}")
    for col in C.OHLC:
        daily[col] = pd.to_numeric(daily[col], errors="coerce")
    daily = daily.loc[(daily[C.OHLC] > 0).all(axis=1)].reset_index(drop=True)

    # Preços internacionais têm feriados próprios. O forward-fill é feito
    # apenas nos níveis, como definido no desenho do experimento.
    for col in ["nky_close", "ukx_close", "baa", "aaa"]:
        if col in daily:
            daily[col] = pd.to_numeric(daily[col], errors="coerce").ffill()

    # Recalcula retornos log a partir dos níveis limpos. Isso evita que um
    # código sentinela, como -1 em um feriado, entre na média do bloco.
    for price_col, return_col in PRICE_RETURN_PAIRS.items():
        if price_col in daily:
            # WTI pode assumir preço spot negativo em abril de 2020; nesse
            # ponto não existe retorno log bem definido.
            price = pd.to_numeric(daily[price_col], errors="coerce").where(lambda s: s > 0)
            daily[return_col] = np.log(price / price.shift(1)).replace([np.inf, -np.inf], np.nan)

    # A base contém algumas colunas inteiras e indicadores já calculados. A
    # agregação converte cada coluna selecionada para numérico e remove NaN.
    return daily


def prepare(output: Path = C.DATA_PARQUET, input_path: Path = C.DATA_XLSX) -> pd.DataFrame:
    daily, macro = _read_input(input_path)
    daily = _clean_daily(daily)
    monthly = aggregate_monthly(daily, block_size=C.BLOCK_SIZE, macro=macro)
    monthly = monthly.sort_values("date").reset_index(drop=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_parquet(output, index=False)
    return monthly


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=C.DATA_XLSX)
    parser.add_argument("--output", type=Path, default=C.DATA_PARQUET)
    parser.add_argument("--smoke", action="store_true", help="imprime verificações adicionais")
    args = parser.parse_args()

    monthly = prepare(args.output, args.input)
    print(f"Painel mensal salvo em: {args.output}")
    print(f"Shape: {monthly.shape}")
    print(f"Período: {monthly.date.min().date()} a {monthly.date.max().date()}")
    print(f"Alvo positivo: {(monthly[C.TARGET] > 0).all()}")
    print(f"NaN no alvo: {monthly[C.TARGET].isna().sum()}")
    if args.smoke:
        print(f"Blocos: {monthly.block_id.nunique()}")
        print(f"Features: {monthly.shape[1] - 4}")
        print(f"RV_YZ máximo: {monthly[C.TARGET].max():.6f}")
        stress = monthly.loc[monthly.date.between("2008-09-01", "2008-12-31"), C.TARGET]
        print(f"RV_YZ forward médio em 2008-09--12: {stress.mean():.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
