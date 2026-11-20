"""Audit the inputs and run the daily backtest from verified forecast tables."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from ..src.bootstrap import bootstrap_index_paths, paired_block_bootstrap
from ..src.strategy import build_signals
from .evaluate import write_forecast_evaluation
from .market import audit, load_followup_market
from .panel import RESULTS
from .strategy import _daily_metrics, expiry_metrics, run_daily_strategy
from utils import save_json

MARKS = ("observed_mid", "mid_else_ask_stale", "ask_stale")
NAMES = ("strategy", "put", "cash", "official_put")
CURVE_COLUMNS = [
    "date", "mark_source", "entered", "expired", "signal_date", "weight",
    *[f"{portfolio}_{kind}" for portfolio in NAMES for kind in ("nav", "return")],
]


def sha256(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def validate_market(market, panel, output):
    spx = market["spx"].loc[lambda d: d.close.gt(0), ["date", "open", "high", "low", "close"]]
    aligned = spx.merge(panel[["date", "spx_open", "spx_high", "spx_low", "spx_close", "vix"]],
                        on="date", how="outer", validate="one_to_one", indicator=True)
    if not aligned["_merge"].eq("both").all():
        raise ValueError("Model and market sessions differ")
    for col in ("open", "high", "low", "close"):
        if not aligned[col].eq(aligned[f"spx_{col}"]).all():
            raise ValueError(f"Model and market SPX {col} differ")
    vix = aligned.merge(market["vix"], on="date", how="left", validate="one_to_one")
    if not vix.vix.eq(vix.value_pct).all():
        raise ValueError("Model and market VIX differ")
    roll = market["rolls"]
    quotes = market["quotes"]
    entries = roll.merge(quotes, on=["option_ticker", "roll_date", "expiry", "strike"], validate="one_to_many")
    entries = entries.loc[entries.quote_date_y.eq(entries.roll_date)]
    level = entries.merge(market["iv"], left_on="roll_date", right_on="date", validate="one_to_one")
    difference = level.ivol_mid_pct - level.value_pct
    if len(level) != len(roll) or difference.isna().any():
        raise ValueError("Entry IV comparison lacks complete pairs")
    spot = roll.merge(spx[["date", "close"]], left_on="roll_date", right_on="date", validate="one_to_one")
    abs_moneyness_pct = 100 * (spot.strike / spot.close - 1).abs()
    # A conservative indicative lower bound, using spot without dividends.
    q = quotes.merge(spx[["date", "close"]], left_on="quote_date", right_on="date", validate="many_to_one")
    rates = market["tbill"][["date", "yield_decimal"]]
    q = pd.merge_asof(q.sort_values("date"), rates, on="date", allow_exact_matches=False)
    days = (q.expiry - q.date).dt.days
    lower = (q.strike / (1 + q.yield_decimal * days / 360) - q.close).clip(lower=0)
    checks, flagged_quotes = {}, []
    for field in ("mid", "ask"):
        shortfall = lower - q[field]
        flagged = shortfall.gt(0.01)
        checks[field] = {
            "valid_pairs": int(shortfall.notna().sum()), "below_indicative_bound": int(flagged.sum()),
            "max_shortfall_index_points": float(shortfall.max()),
        }
        flagged_quotes.append(q.loc[flagged, ["quote_date", "option_ticker"]].assign(
            field=field, shortfall_index_points=shortfall[flagged],
        ))
    pd.concat(flagged_quotes, ignore_index=True).sort_values(["quote_date", "field"]).to_csv(
        output / "data_audit" / "quote_bound_screen.csv", index=False,
    )
    save_json({
        "aligned_spx_and_vix_model_sessions": len(aligned),
        "aligned_2025_sessions": int(aligned.date.dt.year.eq(2025).sum()),
        "max_absolute_model_market_ohlc_and_vix_difference": 0.0,
        "entry_iv_contract_minus_live_pp": difference.describe().to_dict(),
        "entry_absolute_strike_distance_from_spx_close_pct": abs_moneyness_pct.describe().to_dict(),
        "quote_lower_bound_screen": checks,
        "bound": "max(K/(1+lagged USGG1M × calendar days/360) − S_close, 0); threshold 0.01 point. "
                 "Indicative only: quote timestamps and yield convention are unresolved.",
    }, output / "data_audit" / "panel_and_marks.json")


def validate_ledger(daily, roll, mark_policy):
    tr = daily.loc[daily.entered]
    settled = daily.loc[daily.expired]
    if (len(tr) != len(roll) or not tr.date.reset_index(drop=True).equals(roll.roll_date.reset_index(drop=True))
            or len(settled) != len(tr) or not tr.signal_available.all()
            or not tr.signal_date.lt(tr.date).all()):
        raise ValueError("Missing or improperly timed entry/settlement")
    pay = (settled.settled_strike - settled.settlement_soq).clip(lower=0)
    if not np.allclose(settled.settlement_payoff_per_unit, pay):
        raise ValueError("AM payoff mismatch")
    realized_growth = daily.cash_nav / daily.cash_nav.shift(1)
    for name in ("strategy", "put"):
        required = tr[f"{name}_units"] * tr.active_strike
        allocated_cash = tr[f"{name}_pretrade_cash"] * (tr.weight if name == "strategy" else 1)
        cash = daily[f"{name}_cash"]
        units = daily[f"{name}_units"]
        received = units * daily.execution_price.fillna(0)
        paid = units.shift(1).fillna(0) * daily.settlement_payoff_per_unit.fillna(0)
        account = cash.shift(1) * realized_growth + received - paid
        nav = cash - units * daily.option_mid.where(units.ne(0), 0)
        if (not np.allclose(required, allocated_cash)
                or not np.allclose(cash.iloc[1:], account.iloc[1:])
                or not np.allclose(daily[f"{name}_nav"], nav, equal_nan=True)):
            raise ValueError(f"{name}: collateral, cash or NAV reconciliation failed")
        if mark_policy != "observed_mid" and not np.isfinite(nav).all():
            raise ValueError(f"{mark_policy}: expected a complete sensitivity curve")


def add_period_metrics(daily, full_metrics):
    prior_close = pd.Timestamp("2024-12-31")
    recent = daily.loc[daily.date.ge(prior_close)].copy()
    if recent.empty or recent.date.iloc[0] != prior_close:
        raise ValueError("Missing calendar anchor for the 2025 comparison")
    recent.loc[recent.index[0], [f"{name}_return" for name in NAMES]] = np.nan
    frames = []
    for period, curve, metrics in (
        ("full", daily, full_metrics), ("2025_to_last_expiry", recent, _daily_metrics(recent)),
    ):
        elapsed = (curve.date.iloc[-1] - curve.date.iloc[0]).days
        metrics = metrics.copy()
        metrics["CAGR"] = (1 + metrics.anchored_total_return)**(365.25 / elapsed) - 1
        metrics["period"] = period
        metrics["start_date"] = curve.date.iloc[0]
        metrics["end_date"] = curve.date.iloc[-1]
        after_anchor = curve.iloc[1:]
        metrics["entries"] = int(after_anchor.entered.sum())
        metrics["trades"] = int((after_anchor.entered & after_anchor.weight.gt(0)).sum())
        metrics["settlements"] = int(after_anchor.expired.sum())
        frames.append(metrics)
    return pd.concat(frames, ignore_index=True)


def run_strategy(market, forecasts, output, bootstrap_reps):
    valid_spx = market["spx"].loc[lambda s: s.close.gt(0)].copy()
    roll = market["rolls"]
    start = valid_spx.loc[valid_spx.date.lt(roll.roll_date.min()), "date"].max()
    end = roll.expiry.max()
    signals_all, curves, performance, expiry_all, expiry_performance, bootstrap, diagnostics = (
        [], [], [], [], [], [], [],
    )
    mark_rules = {}
    models = forecasts[forecasts.horizon.eq(22)]
    for model, fc in models.groupby("model"):
        point = fc[["origin_date", "y_hat"]].rename(columns={"origin_date": "date", "y_hat": "h22"})
        for iv_id, label in (("iv", "LIVE"), ("vix", "VIX")):
            iv = market[iv_id][["date", "value_pct"]].rename(columns={"value_pct": "iv_percent"})
            signal = build_signals(valid_spx[["date"]].merge(point, on="date", validate="one_to_one")
                                   .merge(iv, on="date", how="left", validate="one_to_one"))
            keys = dict(model=model, iv_source=label)
            signals_all.append(signal.assign(**keys))
            for execution in ("bid", "mid"):
                for policy in MARKS:
                    tags = dict(**keys, execution=execution, mark_policy=policy)
                    result = run_daily_strategy(market, signal, execution=execution, start=start, end=end,
                                                mark_policy=policy)
                    mark_rules[policy] = result["metadata"]["marks"]
                    daily = result["daily"]
                    if not result["excluded"].empty:
                        raise ValueError(result["excluded"].to_string(index=False))
                    validate_ledger(daily, roll, policy)
                    indexed_signal = signal.set_index("date")
                    decisions = daily.loc[daily.entered]
                    observed = signal.loc[signal.date.between(start, end)]
                    if policy == "observed_mid" and execution == "bid":
                        diagnostics.append({
                            **keys, "positive_VRP_sessions": int(observed.vrp.gt(0).sum()),
                            "min_VRP": observed.vrp.min(), "max_VRP": observed.vrp.max(),
                            "monthly_opportunities": len(decisions), "monthly_trades": int(decisions.weight.gt(0).sum()),
                            "positive_VRP_at_entry_signals": int(indexed_signal.loc[decisions.signal_date, "vrp"].gt(0).sum()),
                            "positive_VRP_sessions_2025": int(observed.loc[observed.date.dt.year.eq(2025), "vrp"].gt(0).sum()),
                        })
                    metrics = add_period_metrics(daily, result["daily_metrics"])
                    performance.append(metrics.assign(**tags))
                    for name in NAMES:
                        daily[f"{name}_nav"] /= float(daily[f"{name}_nav"].iloc[0])
                    curves.append(daily[CURVE_COLUMNS].assign(**tags))
                    if policy == "observed_mid":
                        expiry = result["expiry"]
                        expiry_all.append(expiry[["start_date", "date", *[f"{n}_return" for n in NAMES]]].assign(**tags))
                        expiry_performance.append(expiry_metrics(expiry).assign(**tags))
                        for comparator in ("put_return", "official_put_return"):
                            summary = paired_block_bootstrap(expiry, block_sizes=(1, 3, 6),
                                                             n_bootstrap=bootstrap_reps, seed=42,
                                                             comparator_col=comparator)
                            bootstrap.append(summary.assign(**tags, comparator=comparator))
        print(f"Daily backtest complete: {model}", flush=True)
    pd.concat(signals_all, ignore_index=True).to_parquet(output / "daily_signals.parquet", index=False)
    pd.concat(curves, ignore_index=True).to_parquet(output / "daily_curves.parquet", index=False)
    pd.concat(performance, ignore_index=True).to_csv(output / "performance_daily.csv", index=False)
    pd.concat(expiry_all, ignore_index=True).to_parquet(output / "expiry_returns.parquet", index=False)
    pd.concat(expiry_performance, ignore_index=True).to_csv(output / "performance_expiry.csv", index=False)
    pd.concat(bootstrap, ignore_index=True).to_csv(output / "bootstrap_paired_expiry.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output / "signal_diagnostics.csv", index=False)
    draws, paths, metadata = bootstrap_index_paths(valid_spx, n_bootstrap=bootstrap_reps, seed=42)
    draws.to_csv(output / "bootstrap_index_draws.csv", index=False)
    paths.to_parquet(output / "bootstrap_index_paths.parquet")
    save_json({**metadata, "start_date": str(valid_spx.date.min().date()),
               "end_date": str(valid_spx.date.max().date())}, output / "bootstrap_index_metadata.json")
    metadata = {key: result["metadata"][key] for key in ("clock", "fees", "rates", "official_put", "expiry")}
    return {"start": str(start.date()), "end": str(end.date()), "entries": len(roll),
            "scenario_count": int(models.model.nunique() * 2 * 2), "mark_policies": mark_rules,
            "execution_prices": ["bid", "mid"], "accounting_reconciled_each_scenario": True, **metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-zip", type=Path, required=True)
    parser.add_argument("--followup-zip", type=Path, required=True)
    parser.add_argument("--panel", type=Path, default=RESULTS / "panel.parquet")
    parser.add_argument("--statistical", type=Path, default=RESULTS / "statistical")
    parser.add_argument("--neural-forecasts", type=Path, nargs=2, required=True,
                        metavar=("LSTM_PARQUET", "TSMIXERX_PARQUET"))
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(args.panel)
    market = load_followup_market(args.original_zip, args.followup_zip)
    data_audit = audit(args.original_zip, args.followup_zip, args.output / "data_audit")
    if (data_audit["quotes"]["entry_matches"] != len(market["rolls"])
            or any(data_audit["quotes"][f"entry_{price}_mismatches"] for price in ("bid", "ask", "mid"))):
        raise ValueError("Daily option entries differ from the original roll file")
    if any(data_audit["settlement"][key] for key in ("missing", "mismatches")):
        raise ValueError("Official SOQ reconciliation failed")
    validate_market(market, panel, args.output)
    forecasts = write_forecast_evaluation(panel, args.statistical, args.neural_forecasts, args.output)
    report = run_strategy(market, forecasts, args.output, args.bootstrap_reps)
    inputs = [args.original_zip, args.followup_zip, args.panel, *args.neural_forecasts,
              args.statistical / "forecasts.parquet", args.statistical / "har_comparison.parquet"]
    files = Path(__file__).parent
    save_json({
        **report, "bootstrap_reps": args.bootstrap_reps,
        "inputs": [{"name": file.name, "sha256": sha256(file)} for file in inputs],
        "source_sha256": {file.name: sha256(file) for file in sorted(files.glob("*.py"))},
        "packages": {package: version(package) for package in ("pandas", "numpy", "scipy", "statsmodels", "pyarrow")},
    }, args.output / "run_manifest.json")


if __name__ == "__main__":
    main()
