"""Daily close-to-close cash-secured put accounting using observed quotes.

AM settlement and replacement at that calendar day's close are assumed, since
input timestamps do not prove execution order. Mid marks are indicative, and
bid/mid execution excludes fees and market impact. Missing marks are never
forward-filled. Cash earns a provisional simple ACT/360 yield, not a bill index.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _daily_metrics(daily):
    rows = []
    for name in ('strategy', 'put', 'cash', 'official_put'):
        ret = daily[f'{name}_return'].to_numpy(float)
        cash = daily['cash_return'].to_numpy(float)
        valid = np.isfinite(ret) & np.isfinite(cash)
        excess = ret[valid] - cash[valid]
        sd = np.std(excess, ddof=1) if len(excess) > 1 else np.nan
        down = np.sqrt(np.mean(np.minimum(excess, 0)**2)) if len(excess) else np.nan
        # Drawdown across observed closes is a lower bound if there are gaps.
        nav = daily[f'{name}_nav'].to_numpy(float)
        nav = nav[np.isfinite(nav)]
        dd = np.min(nav / np.maximum.accumulate(nav) - 1) if len(nav) else np.nan
        common = valid & np.isfinite(daily['strategy_return'].to_numpy(float))
        common_excess = ret[common] - cash[common]
        common_sd = np.std(common_excess, ddof=1) if len(common_excess) > 1 else np.nan
        anchored_total = nav[-1] / nav[0] - 1 if len(nav) and np.isfinite(daily[f'{name}_nav'].iloc[0]) and np.isfinite(daily[f'{name}_nav'].iloc[-1]) else np.nan
        rows.append(dict(portfolio=name, anchored_total_return=anchored_total,
                         common_strategy_days=int(common.sum()),
                         common_days_sharpe_annualized=np.mean(common_excess)/common_sd*np.sqrt(252) if common_sd > 0 else np.nan,
                         paired_daily_returns=int(valid.sum()),
                         expected_daily_returns=len(daily)-1,
                         missing_daily_returns=int((~valid[1:]).sum()),
                         daily_sharpe_annualized=np.mean(excess)/sd*np.sqrt(252) if sd > 0 else np.nan,
                         daily_sortino_annualized=np.mean(excess)/down*np.sqrt(252) if down > 0 else np.nan,
                         observed_close_max_drawdown=dd,
                         drawdown_is_partial=bool(daily[f'{name}_nav'].isna().any()),
                         drawdown_uses_fallback_marks=bool((daily['mark_source'].isin(['ask fallback', 'mid fallback', 'stale fallback']) & (daily[f'{name}_units'] > 0)).any()) if name in ('strategy', 'put', 'cash') else False,
                         annualization_sessions=252))
    return pd.DataFrame(rows)


def run_daily_strategy(market, signals, execution='bid', initial_cash=1.,
                       start=None, end=None, max_rate_age_days=7, mark_policy='observed_mid'):
    """Return daily, expiry and exclusion frames plus metadata.

    `market` is the follow-up loader dictionary. `signals` has date and weight
    columns; at each entry only the immediately preceding SPX session is used.
    Missing weight means cash for the strategy, while the matched put benchmark
    still writes. Quotes must identify the selected roll ticker and have exact
    session dates. Missing entry bid/mid excludes that trade for all portfolios;
    missing intermediate mid yields missing NAV and daily returns, not stale marks.
    Expiry returns are close-to-close, including any replacement execution cost.
    Open final contracts are marked at the last session and not counted as an
    expiry. The official PUT series is a date-matched close-to-close comparator.
    """
    if mark_policy not in ('observed_mid', 'ask_stale', 'mid_else_ask_stale'):
        raise ValueError('invalid mark policy')
    if execution not in ('bid', 'mid') or not np.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError('execution must be bid or mid and initial cash positive')
    spx = market['spx'].copy().sort_values('date')
    spx['date'] = pd.to_datetime(spx.date)
    if spx.date.isna().any() or spx.date.duplicated().any():
        raise ValueError('invalid SPX calendar')
    spx['close'] = pd.to_numeric(spx['close'], errors='coerce')
    spx = spx.loc[np.isfinite(spx.close) & (spx.close > 0)].copy()
    dates = pd.DatetimeIndex(spx.date)
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]
    if len(dates) < 2:
        raise ValueError('at least two SPX sessions required')
    sig = signals[['date', 'weight']].copy()
    sig.date = pd.to_datetime(sig.date)
    if sig.date.isna().any() or sig.date.duplicated().any():
        raise ValueError('signal dates must be unique')
    sig.weight = pd.to_numeric(sig.weight, errors='raise')
    if ((sig.weight.dropna() < 0) | (sig.weight.dropna() > 1) | ~np.isfinite(sig.weight.dropna())).any():
        raise ValueError('invalid signal weight')
    sig = sig.set_index('date').weight
    rates = market['tbill'][['date', 'yield_pct']].copy().sort_values('date')
    rates.date = pd.to_datetime(rates.date)
    if rates.date.isna().any() or rates.date.duplicated().any():
        raise ValueError('invalid rate dates')
    quotes = market['quotes'].copy()
    quotes.quote_date = pd.to_datetime(quotes.quote_date)
    if quotes.duplicated(['quote_date', 'option_ticker']).any():
        raise ValueError('duplicate daily option quote')
    quotes = quotes.set_index(['quote_date', 'option_ticker'])
    rolls = market['rolls'].copy().sort_values('roll_date')
    for col in ('roll_date', 'expiry', 'quote_date'):
        rolls[col] = pd.to_datetime(rolls[col])
    if rolls.roll_date.duplicated().any():
        raise ValueError('duplicate roll dates')
    settlements = market['settlement'].copy()
    settlements.settle_date = pd.to_datetime(settlements.settle_date)
    if settlements.settle_date.duplicated().any():
        raise ValueError('duplicate settlement dates')
    settlements = settlements.set_index('settle_date').SOQ
    put = market['put_index'].copy()
    put.date = pd.to_datetime(put.date)
    if put.date.duplicated().any():
        raise ValueError('duplicate PUT dates')
    put = put.set_index('date').close
    # Accrual is evaluated for every calendar day, including non-trading days.
    accrual = pd.DataFrame({'day': pd.date_range(dates[0], dates[-1], inclusive='left')})
    matched = pd.merge_asof(accrual, rates, left_on='day', right_on='date',
                            allow_exact_matches=False, tolerance=pd.Timedelta(days=max_rate_age_days))
    if matched.yield_pct.isna().any():
        raise ValueError('missing causal rate within maximum age')
    factors = 1 + matched.yield_pct.to_numpy(float) / 100 / 360
    if not np.isfinite(factors).all() or (factors <= 0).any():
        raise ValueError('invalid cash rate')
    cumulative = np.r_[1., np.cumprod(factors)]
    day_offsets = (dates - dates[0]).days.to_numpy()
    interval_factors = cumulative[day_offsets[1:]] / cumulative[day_offsets[:-1]]
    holdings = {name: {'cash': float(initial_cash), 'units': 0.} for name in ('strategy', 'put', 'cash')}
    active = None
    last_mark = np.nan
    exclusions, records = [], []
    rmap = rolls.set_index('roll_date')
    full_calendar = pd.DatetimeIndex(spx.date)
    for i, date in enumerate(dates):
        if i:
            factor = interval_factors[i-1]
            for h in holdings.values():
                h['cash'] *= factor
        expired = False
        entered = False
        settled_id, settled_strike, soq, payoff = None, np.nan, np.nan, np.nan
        execution_price = np.nan
        pretrade_cash = {name: np.nan for name in holdings}
        entry_premium = {name: 0. for name in holdings}
        settlement_payments = {name: 0. for name in holdings}
        if active is not None and active['expiry'] <= date:
            if active['expiry'] != date or date not in settlements.index or not np.isfinite(settlements.loc[date]):
                raise ValueError(f'missing AM settlement at {active["expiry"].date()}')
            settled_id, settled_strike = active['ticker'], active['strike']
            soq = float(settlements.loc[date])
            payoff = max(active['strike'] - soq, 0.)
            for name, h in holdings.items():
                settlement_payments[name] = h['units'] * payoff
                h['cash'] -= settlement_payments[name]
                h['units'] = 0.
            active = None
            expired = True
        if date in rmap.index:
            roll = rmap.loc[date]
            reason = None
            prior = full_calendar[full_calendar < date]
            ticker = roll.option_ticker
            quote = quotes.loc[(date, ticker)] if (date, ticker) in quotes.index else None
            if active is not None:
                reason = 'overlapping previous contract'
            elif not len(prior) or (date-prior[-1]).days > 7:
                reason = 'no recent prior trading session'
            elif roll.quote_date != date:
                reason = 'roll quote date differs from entry'
            elif quote is None or not np.isfinite(quote.get('mid', np.nan)) or not np.isfinite(quote.get(execution, np.nan)):
                reason = 'missing same-day execution or mid mark'
            elif roll.expiry <= date or not np.isfinite(roll.strike) or roll.strike <= 0:
                reason = 'invalid contract'
            elif roll.expiry <= dates[-1] and (roll.expiry not in settlements.index or not np.isfinite(settlements.loc[roll.expiry])):
                reason = 'missing AM SOQ'
            elif quote[execution] < 0 or quote['mid'] < 0:
                reason = 'invalid quote'
            if reason:
                exclusions.append(dict(entry_date=date, contract_id=ticker, reason=reason))
            else:
                signal_date = prior[-1]
                w = sig.get(signal_date, np.nan)
                weight = 0. if pd.isna(w) else float(w)
                last_mark = np.nan
                entered = True
                execution_price = float(quote[execution])
                active = dict(ticker=ticker, strike=float(roll.strike), expiry=roll.expiry,
                              signal_date=signal_date, weight=weight, signal_available=not pd.isna(w))
                for name, allocation in (('strategy', weight), ('put', 1.), ('cash', 0.)):
                    h = holdings[name]
                    pretrade_cash[name] = h['cash']
                    h['units'] = allocation * h['cash'] / active['strike']
                    entry_premium[name] = h['units'] * execution_price
                    h['cash'] += entry_premium[name]
        mark = np.nan
        mark_source = 'no position'
        if active is not None:
            mark_source = 'missing'
            q = quotes.loc[(date, active['ticker'])] if (date, active['ticker']) in quotes.index else None
            if q is not None:
                field = 'ask' if mark_policy == 'ask_stale' else 'mid'
                candidate = q.get(field, np.nan)
                if np.isfinite(candidate) and candidate >= 0:
                    mark, mark_source = float(candidate), field
                elif mark_policy != 'observed_mid':
                    fallback = 'mid' if field == 'ask' else 'ask'
                    candidate = q.get(fallback, np.nan)
                    if np.isfinite(candidate) and candidate >= 0:
                        mark, mark_source = float(candidate), fallback + ' fallback'
            if mark_policy != 'observed_mid' and not np.isfinite(mark) and np.isfinite(last_mark):
                mark, mark_source = max(last_mark, 0.), 'stale fallback'
            if np.isfinite(mark):
                last_mark = mark
        row = dict(date=date, entered=entered, execution_price=execution_price,
                   active_strike=active['strike'] if active else np.nan,
                   settled_contract_id=settled_id, settled_strike=settled_strike,
                   settlement_soq=soq, settlement_payoff_per_unit=payoff,
                   mark_source=mark_source, contract_id=active['ticker'] if active else None,
                   signal_date=active['signal_date'] if active else pd.NaT,
                   weight=active['weight'] if active else 0.,
                   signal_available=active['signal_available'] if active else False,
                   option_mid=mark, expired=expired)
        for name, h in holdings.items():
            row[f'{name}_pretrade_cash'] = pretrade_cash[name]
            row[f'{name}_entry_premium'] = entry_premium[name]
            row[f'{name}_settlement_payment'] = settlement_payments[name]
            row[f'{name}_cash'] = h['cash']
            row[f'{name}_units'] = h['units']
            row[f'{name}_nav'] = h['cash'] if h['units'] == 0 else h['cash'] - h['units'] * mark
            if np.isfinite(row[f'{name}_nav']) and row[f'{name}_nav'] <= 0:
                raise ValueError('nonpositive NAV; check quote and contract inputs')
        row['official_put_nav'] = put.get(date, np.nan)
        if np.isfinite(row['official_put_nav']) and row['official_put_nav'] <= 0:
            raise ValueError('nonpositive official PUT level')
        records.append(row)
    daily = pd.DataFrame(records)
    for name in ('strategy', 'put', 'cash', 'official_put'):
        daily[f'{name}_return'] = daily[f'{name}_nav'] / daily[f'{name}_nav'].shift(1) - 1
    # Exact close endpoints: missing marks do not silently create multi-period returns.
    endpoints = daily.loc[daily.expired, ['date'] + [f'{n}_nav' for n in ('strategy', 'put', 'cash', 'official_put')]].copy()
    first = daily.iloc[[0]][endpoints.columns]
    endpoints = pd.concat([first, endpoints.loc[endpoints.date != first.date.iloc[0]]], ignore_index=True)
    for name in ('strategy', 'put', 'cash', 'official_put'):
        endpoints[f'{name}_start_nav'] = endpoints[f'{name}_nav'].shift(1)
        endpoints[f'{name}_return'] = endpoints[f'{name}_nav'] / endpoints[f'{name}_start_nav'] - 1
    expiry = endpoints.iloc[1:].copy()
    expiry.insert(0, 'start_date', endpoints.date.shift(1).iloc[1:].to_numpy())
    return {'daily': daily, 'expiry': expiry, 'daily_metrics': _daily_metrics(daily),
            'excluded': pd.DataFrame(exclusions, columns=['entry_date', 'contract_id', 'reason']),
            'metadata': {'clock': 'close-to-close; AM SOQ then replacement EOD assumed',
                         'marks': ('same-day observed mid; missing marks remain missing' if mark_policy == 'observed_mid' else ('current ask, else current mid, else last valid mark floored at zero; approximate sensitivity' if mark_policy == 'ask_stale' else 'current mid, else current ask, else last valid mark floored at zero; approximate sensitivity')),
                         'execution': execution, 'fees': 'none', 'rates': 'strictly prior, ACT/360, source convention unverified',
                         'official_put': 'official close-to-close; not an AM settlement index',
                         'expiry': 'EOD-to-EOD, including replacement execution; final open position excluded'}}


def expiry_metrics(expiry):
    """Close-to-close expiry-period metrics; drawdown is expiry-observed only."""
    rows = []
    cash = expiry['cash_return'].to_numpy(float)
    for name in ('strategy', 'put', 'cash', 'official_put'):
        ret = expiry[f'{name}_return'].to_numpy(float)
        ok = np.isfinite(ret) & np.isfinite(cash)
        excess = ret[ok] - cash[ok]
        sd = np.std(excess, ddof=1) if len(excess) > 1 else np.nan
        downside = np.sqrt(np.mean(np.minimum(excess, 0)**2)) if len(excess) else np.nan
        nav = np.r_[expiry[f'{name}_start_nav'].iloc[0], expiry[f'{name}_nav'].to_numpy(float)] if len(expiry) else np.array([])
        nav = nav[np.isfinite(nav)]
        rows.append(dict(portfolio=name, paired_expiry_returns=int(ok.sum()),
                         missing_expiry_returns=int((~ok).sum()),
                         expiry_sharpe_annualized=np.mean(excess)/sd*np.sqrt(12) if sd > 0 else np.nan,
                         expiry_sortino_annualized=np.mean(excess)/downside*np.sqrt(12) if downside > 0 else np.nan,
                         expiry_observed_max_drawdown=np.min(nav/np.maximum.accumulate(nav)-1) if len(nav) else np.nan,
                         annualization_periods=12))
    return pd.DataFrame(rows)
