"""Cria o painel diário H=22 com Yang-Zhang corrigido e splits 8/2/1."""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

import config as C
from utils import (add_forward_close_variance, compute_yang_zhang,
                   ensure_directories, feature_columns, rolling_821_windows,
                   write_split_manifest)


def _clean_daily(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).strip() for c in out]
    out = out.drop(columns=[c for c in out if c.lower().startswith("unnamed")], errors="ignore")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)

    for col in ["spx_open", "spx_high", "spx_low", "spx_close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.loc[(out[["spx_open", "spx_high", "spx_low", "spx_close"]] > 0).all(axis=1)].reset_index(drop=True)

    for col in ["nky_close", "ukx_close", "baa", "aaa"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce").ffill()
    for price_col, return_col in C.PRICE_RETURN_PAIRS.items():
        if price_col in out:
            price = pd.to_numeric(out[price_col], errors="coerce").where(lambda s: s > 0)
            out[return_col] = np.log(price / price.shift(1)).replace([np.inf, -np.inf], np.nan)

    # Estas séries serão reconstruídas com uma data de disponibilidade
    # conservadora, em vez de usar o mês de referência como se fosse release.
    out = out.drop(columns=[*C.MONTHLY_MACRO, *C.WEEKLY_MACRO], errors="ignore")
    if "epu" in out:
        out["epu"] = pd.to_numeric(out["epu"], errors="coerce").shift(22)
    return out


def _merge_macro_conservative(daily: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    out = daily.sort_values("date").copy()
    m = macro.copy()
    m.columns = [str(c).strip() for c in m]
    if "date_ref_mensal" in m:
        mm_cols = [c for c in C.MONTHLY_MACRO if c in m]
        mm = m[["date_ref_mensal", *mm_cols]].dropna(subset=["date_ref_mensal"]).copy()
        mm["date_ref_mensal"] = pd.to_datetime(mm["date_ref_mensal"])
        # Sem release dates individuais na planilha, adotamos um mês completo
        # de atraso. A hipótese fica registrada no manifesto.
        mm["availability_date"] = mm["date_ref_mensal"] + pd.offsets.MonthEnd(1)
        mm = mm.sort_values("availability_date").drop_duplicates("availability_date", keep="last")
        out = pd.merge_asof(out, mm[["availability_date", *mm_cols]],
                            left_on="date", right_on="availability_date", direction="backward")
        out = out.drop(columns="availability_date")
    if "date_ref_semanal" in m and "stress_index" in m:
        mw = m[["date_ref_semanal", "stress_index"]].dropna(subset=["date_ref_semanal"]).copy()
        mw["date_ref_semanal"] = pd.to_datetime(mw["date_ref_semanal"])
        mw["availability_date"] = mw["date_ref_semanal"] + pd.Timedelta(days=7)
        mw = mw.sort_values("availability_date").drop_duplicates("availability_date", keep="last")
        out = pd.merge_asof(out.sort_values("date"), mw[["availability_date", "stress_index"]],
                            left_on="date", right_on="availability_date", direction="backward")
        out = out.drop(columns="availability_date")
    return out


def prepare(input_path=C.SOURCE_XLSX, output=C.DAILY_PANEL) -> pd.DataFrame:
    ensure_directories()
    daily = pd.read_excel(input_path, sheet_name="painel_diario")
    macro = pd.read_excel(input_path, sheet_name="macro")
    daily = _merge_macro_conservative(_clean_daily(daily), macro)
    daily[C.TARGET] = compute_yang_zhang(daily)
    daily = add_forward_close_variance(daily)
    daily.insert(1, "unique_id", "sp500")
    daily.to_parquet(output, index=False)
    windows = rolling_821_windows(daily)
    write_split_manifest(daily, windows)
    return daily


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=C.SOURCE_XLSX)
    parser.add_argument("--output", default=C.DAILY_PANEL)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    panel = prepare(args.input, args.output)
    windows = rolling_821_windows(panel)
    print(f"Painel salvo em {args.output}; shape={panel.shape}")
    print(f"Período: {panel.date.min().date()} a {panel.date.max().date()}")
    print(f"RV_YZ válida: {panel[C.TARGET].notna().sum()}; janelas 8/2/1: {len(windows)}")
    print(f"Features principais: {len(feature_columns(panel))}; VIX3M excluído: {'vix3m' not in feature_columns(panel)}")
    if args.smoke:
        assert len(windows) == 16, f"Esperadas 16 janelas, obtidas {len(windows)}"
        assert (panel[C.TARGET].dropna() > 0).all()
        first = windows[0]
        print({k: v for k, v in first.items() if not k.endswith("_idx")})
        print(panel.loc[panel.date.between("2020-02-01", "2020-05-31"), ["date", C.TARGET]].nlargest(5, C.TARGET).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

