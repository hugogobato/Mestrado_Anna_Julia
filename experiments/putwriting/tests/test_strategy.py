import numpy as np
import pandas as pd
import pytest
from experiments.putwriting.src.strategy import build_signals, backtest_puts


def test_prior_window_and_ties_and_no_warmup_trade():
    f = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=6),
                      'iv_percent': [10,20,30,20,1,40], 'h22': [0]*6})
    s = build_signals(f, window=3)
    assert s.weight.iloc[:3].isna().all()
    assert s["quantile"].iloc[3] == 2/3
    assert s["quantile"].iloc[4] == 0
    changed = f.copy(); changed.loc[3,'iv_percent'] = 40
    assert build_signals(changed, 3)["quantile"].iloc[3] == 1
    changed.loc[5, 'iv_percent'] = 1
    assert build_signals(changed, 3)["quantile"].iloc[3] == 1


def test_nonpositive_vrp_zero_and_missing_window():
    f = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=5),
                      'iv_percent': [10, np.nan, 10, 10, 10], 'h22': [.02]*5})
    s = build_signals(f, 2)
    assert np.isnan(s.weight.iloc[2])
    assert np.isnan(s.weight.iloc[3])
    assert s.weight.iloc[4] == 0


def inputs():
    s = pd.DataFrame({'date': pd.to_datetime(['2020-01-01']), 'weight': [.5]})
    c = pd.DataFrame([dict(signal_date='2020-01-01', entry_date='2020-01-02', expiry_date='2020-02-01',
                           contract_id='SPX1', strike=100, premium=2, settlement=90,
                           settlement_type='AM', settlement_source='SOQ', tbill_annual_pct=12, rate_date='2020-01-02')])
    return s, c


def test_hand_accounting_and_cash():
    s, c = inputs(); out = backtest_puts(s, c, initial_cash=100)
    r = out.iloc[0]
    assert r.strategy_units == .5 and r.strategy_collateral == 50
    assert r.put_units == 1 and r.put_collateral == 100
    assert r.strategy_end == pytest.approx(101*1.01 - 5)
    assert r.put_end == pytest.approx(102*1.01 - 10)
    assert r.cash_end == pytest.approx(101)


def test_no_lookahead_and_settlement_rejections():
    s,c = inputs()
    c.loc[0,'signal_date'] = c.loc[0,'entry_date']
    with pytest.raises(ValueError): backtest_puts(s,c)
    s,c = inputs(); c.loc[0,'settlement_source'] = 'OFFICIAL_CLOSE'
    with pytest.raises(ValueError): backtest_puts(s,c)
    s,c = inputs(); c.loc[0,'settlement'] = np.nan
    with pytest.raises(ValueError): backtest_puts(s,c)
    s,c = inputs(); c.loc[0,'rate_date'] = '2020-01-03'
    with pytest.raises(ValueError): backtest_puts(s,c)


def test_overlap_rejected_and_warmup_held_in_cash():
    s,c = inputs(); s.loc[0,'weight'] = np.nan
    out = backtest_puts(s,c, initial_cash=100)
    assert out.strategy_end.iloc[0] == out.cash_end.iloc[0]
    second = c.copy(); second['contract_id'] = 'SPX2'; second['entry_date'] = '2020-01-15'; second['expiry_date'] = '2020-02-15'
    with pytest.raises(ValueError): backtest_puts(s,pd.concat([c,second]))


def test_same_day_am_roll_explicit_and_gap_accrual():
    s,c = inputs()
    s2 = pd.DataFrame({'date': pd.to_datetime(['2020-01-31']), 'weight': [0.]})
    second = c.copy(); second['signal_date'] = '2020-01-31'; second['entry_date'] = '2020-02-01'; second['expiry_date'] = '2020-03-01'; second['contract_id'] = 'SPX2'
    contracts = pd.concat([c,second], ignore_index=True)
    contracts['gap_growth_factor'] = [1, 1.02]
    with pytest.raises(ValueError): backtest_puts(pd.concat([s,s2]), contracts, initial_cash=100)
    out = backtest_puts(pd.concat([s,s2]), contracts, initial_cash=100, allow_same_day_am_roll=True)
    assert out.strategy_end.iloc[1] == pytest.approx(out.strategy_end.iloc[0]*1.02*(1+.12*29/360))
