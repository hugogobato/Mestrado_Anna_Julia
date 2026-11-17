"""Causal variance-risk-premium signals and observed-quote put accounting.

Inputs are observations, not model-implied option prices. Each contract row is one
non-overlapping holding period, selected independently of its subsequent payoff.
All monetary quantities use the same index-point units; fractional contracts are
allowed. Rates are annual simple-interest percentage quotes on the specified basis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_signals(frame: pd.DataFrame, window: int = 504) -> pd.DataFrame:
    """Compute the empirical rank against exactly `window` prior sessions.

    A missing input invalidates every window containing it. Dates must represent
    consecutive observed market sessions (the caller is responsible for the
    exchange calendar). Ties count as less than or equal to the current value.
    """
    if window < 1:
        raise ValueError("window must be positive")
    out = frame.copy()
    out['date'] = pd.to_datetime(out['date'])
    if out['date'].isna().any() or out['date'].duplicated().any() or not out['date'].is_monotonic_increasing:
        raise ValueError('dates must be valid, unique, and increasing')
    iv = pd.to_numeric(out['iv_percent'], errors='raise')
    forecast = pd.to_numeric(out['h22'], errors='raise')
    if ((iv.dropna() < 0).any() or (forecast.dropna() < 0).any() or
            np.isinf(iv).any() or np.isinf(forecast).any()):
        raise ValueError('IV and annual variance forecasts must be finite and nonnegative')
    out['vrp'] = (iv / 100.0) ** 2 - forecast
    values = out['vrp'].to_numpy(dtype=float)
    rank = np.full(len(out), np.nan)
    for i in range(window, len(out)):
        prior = values[i-window:i]
        if np.isfinite(values[i]) and np.isfinite(prior).all():
            rank[i] = np.mean(prior <= values[i])
    out['quantile'] = rank
    out['weight'] = np.where(np.isnan(rank), np.nan,
                             np.where(values > 0, np.minimum(rank, 1.0), 0.0))
    return out


def backtest_puts(signals: pd.DataFrame, contracts: pd.DataFrame,
                  initial_cash: float = 1.0, annual_day_basis: float = 360.0,
                  allow_same_day_am_roll: bool = False) -> pd.DataFrame:
    """Account for sequential observed put contracts and matched cash benchmark.

    contracts columns: signal_date, entry_date, expiry_date, contract_id, strike,
    premium, settlement, settlement_type (AM or PM), settlement_source (SOQ or
    OFFICIAL_CLOSE respectively), tbill_annual_pct, and rate_date. Premium is a
    per-unit *observed* execution price. Rates must be known by entry. Entry and
    expiry timestamps are calendar dates, with simple actual-days/basis accrual.
    Signal weights are applied once per contract; unavailable warm-up signals are
    held in cash. Quote, settlement and selected contract must be obtained and
    validated externally. No fill or selection is simulated here.

    Output rows are expiry-to-expiry period observations, not daily marks. The
    three portfolios start equally and independently reinvest their own wealth.
    Gaps earn no return unless an externally computed gap_growth_factor is supplied.
    An optional cash_growth_factor column allows externally computed causal
    day-by-day accrual. Same-day rolls require an explicit assumption that the
    replacement fill follows AM settlement; date-only data cannot verify this.
    """
    if not np.isfinite(initial_cash) or initial_cash <= 0 or not np.isfinite(annual_day_basis) or annual_day_basis <= 0:
        raise ValueError('initial cash and day basis must be positive')
    required = ['signal_date','entry_date','expiry_date','contract_id','strike','premium',
                'settlement','settlement_type','settlement_source','tbill_annual_pct','rate_date']
    missing = set(required) - set(contracts.columns)
    if missing:
        raise ValueError(f'missing contract columns: {sorted(missing)}')
    s = signals[['date','weight']].copy()
    s['date'] = pd.to_datetime(s['date'])
    if s['date'].isna().any() or s['date'].duplicated().any():
        raise ValueError('signal dates must be valid and unique')
    c = contracts.copy()
    for col in ('signal_date','entry_date','expiry_date','rate_date'):
        c[col] = pd.to_datetime(c[col])
    if c.empty:
        return pd.DataFrame()
    if c[['signal_date','entry_date','expiry_date','rate_date']].isna().any().any():
        raise ValueError('all dates are required')
    if not c['entry_date'].is_monotonic_increasing:
        raise ValueError('contracts must be ordered by entry date')
    if ((c.signal_date >= c.entry_date) | (c.entry_date >= c.expiry_date) | (c.rate_date > c.entry_date)).any():
        raise ValueError('require signal before entry, entry before expiry, and rate known at entry')
    if len(c) > 1:
        next_entry = c.entry_date.iloc[1:].to_numpy()
        previous_expiry = c.expiry_date.iloc[:-1].to_numpy()
        same_day = next_entry == previous_expiry
        previous_is_am = (c.settlement_type.iloc[:-1].to_numpy() == 'AM')
        if (next_entry < previous_expiry).any() or (same_day & (~previous_is_am | (not allow_same_day_am_roll))).any():
            raise ValueError('overlap or unverified same-day roll')
    if c['contract_id'].isna().any() or (c['contract_id'].astype(str).str.strip() == '').any() or c['contract_id'].duplicated().any():
        raise ValueError('unique contract identities are required')
    for col in ('strike','premium','settlement','tbill_annual_pct'):
        c[col] = pd.to_numeric(c[col], errors='raise')
        if not np.isfinite(c[col]).all():
            raise ValueError(f'{col} must be observed and finite')
    if (c.strike <= 0).any() or (c.premium < 0).any() or (c.settlement < 0).any():
        raise ValueError('invalid strike, premium, or settlement')
    valid_settlement = ((c.settlement_type == 'AM') & (c.settlement_source == 'SOQ')) | ((c.settlement_type == 'PM') & (c.settlement_source == 'OFFICIAL_CLOSE'))
    if not valid_settlement.all():
        raise ValueError('AM settlement requires SOQ; PM settlement requires official close')
    c = c.merge(s, left_on='signal_date', right_on='date', how='left', validate='many_to_one', indicator=True)
    if (c['_merge'] != 'both').any():
        raise ValueError('contract signal date must exist in signals')
    if ((c.weight.dropna() < 0) | (c.weight.dropna() > 1) | ~np.isfinite(c.weight.dropna())).any():
        raise ValueError('weights must be in [0,1] or missing')
    wealth = dict(strategy=float(initial_cash), put=float(initial_cash), cash=float(initial_cash))
    rows = []
    for row in c.itertuples(index=False):
        days = (row.expiry_date - row.entry_date).days
        factor = (row.cash_growth_factor if 'cash_growth_factor' in c.columns
                  else 1 + row.tbill_annual_pct / 100 * days / annual_day_basis)
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError('cash accrual factor must be positive')
        gap_factor = row.gap_growth_factor if 'gap_growth_factor' in c.columns else 1.0
        if not np.isfinite(gap_factor) or gap_factor <= 0:
            raise ValueError('gap growth factor must be positive')
        weight = 0.0 if pd.isna(row.weight) else float(row.weight)
        intrinsic = max(row.strike - row.settlement, 0.0)
        result = dict(signal_date=row.signal_date, entry_date=row.entry_date,
                      expiry_date=row.expiry_date, contract_id=row.contract_id,
                      weight=weight, signal_available=not pd.isna(row.weight),
                      strike=row.strike, premium=row.premium, settlement=row.settlement,
                      settlement_type=row.settlement_type, settlement_source=row.settlement_source,
                      tbill_annual_pct=row.tbill_annual_pct, rate_date=row.rate_date,
                      days=days, intrinsic=intrinsic, cash_growth_factor=factor, gap_growth_factor=gap_factor)
        for name, allocation in (('strategy', weight), ('put', 1.0), ('cash', 0.0)):
            start = wealth[name]
            entry_wealth = start * gap_factor
            units = allocation * entry_wealth / row.strike
            collateral = units * row.strike
            # Cash receives premium at execution; all cash earns the quoted rate.
            end = (entry_wealth + units * row.premium) * factor - units * intrinsic
            if end < -1e-12:
                raise ValueError('negative terminal wealth; check rate/contract inputs')
            end = max(end, 0.0)
            result.update({f'{name}_start': start, f'{name}_units': units,
                           f'{name}_collateral': collateral, f'{name}_end': end,
                           f'{name}_return': end / start - 1 if start else np.nan})
            wealth[name] = end
        rows.append(result)
    return pd.DataFrame(rows)


def expiry_metrics(periods: pd.DataFrame, periods_per_year: float = 12.0) -> pd.DataFrame:
    """Expiry-observed statistics, never daily risk measures.

    Sharpe and Sortino use paired period cash returns as the risk-free series and
    sqrt(periods_per_year) scaling. Drawdown observes expiry wealth only, so can
    miss larger intraperiod losses. Annualization assumes comparable periods.
    """
    if periods_per_year <= 0 or periods.empty:
        raise ValueError('nonempty periods and positive annualization required')
    result = []
    cash = periods['cash_return'].to_numpy(dtype=float)
    for name in ('strategy','put','cash'):
        returns = periods[f'{name}_return'].to_numpy(dtype=float)
        excess = returns - cash
        vol = np.std(excess, ddof=1) if len(excess) > 1 else np.nan
        downside = np.sqrt(np.mean(np.minimum(excess, 0) ** 2))
        curve = np.r_[1.0, np.cumprod(1 + returns)]
        peak = np.maximum.accumulate(curve)
        dd = np.min(curve / peak - 1)
        result.append({'portfolio': name, 'periods': len(returns),
                       'total_return': float(curve[-1] - 1),
                       'mean_period_return': float(np.mean(returns)),
                       'expiry_sharpe_annualized': float(np.mean(excess) / vol * np.sqrt(periods_per_year)) if vol > 0 else np.nan,
                       'expiry_sortino_annualized': float(np.mean(excess) / downside * np.sqrt(periods_per_year)) if downside > 0 else np.nan,
                       'expiry_max_drawdown': float(dd)})
    return pd.DataFrame(result)
