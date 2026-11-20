import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('followup_strategy', Path(__file__).parents[1] / 'strategy.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


def fixture():
    d = pd.to_datetime(['2025-01-02','2025-01-03','2025-01-06','2025-01-07','2025-01-08'])
    market = dict(spx=pd.DataFrame({'date':d, 'close':[100]*5}),
        tbill=pd.DataFrame({'date':pd.to_datetime(['2025-01-01','2025-01-03']), 'yield_pct':[0., 360.]}),
        put_index=pd.DataFrame({'date':d, 'close':[100,101,102,103,104]}),
        rolls=pd.DataFrame({'roll_date':d[[1,3]], 'expiry':pd.to_datetime(['2025-01-07','2025-02-07']),
          'quote_date':d[[1,3]], 'option_ticker':['a','b'], 'strike':[100.,100.]}),
        settlement=pd.DataFrame({'settle_date':[d[3]],'SOQ':[90.]}),
        quotes=pd.DataFrame({'quote_date':d[[1,2,3,4]],'option_ticker':['a','a','b','b'],
          'mid':[2.,np.nan,3.,4.], 'bid':[1.,np.nan,2.,3.], 'ask':[3.,np.nan,4.,5.]}))
    signals = pd.DataFrame({'date':d,'weight':[1.,0.,.5,1.,1.]})
    return market, signals


def test_lag_settlement_missing_marks_and_put_clock():
    market, signals = fixture()
    result = s.run_daily_strategy(market, signals, start='2025-01-03', execution='bid')
    d = result['daily']
    assert d.loc[0,'weight'] == 1
    assert d.loc[0,'strategy_nav'] == pytest.approx(.99)
    assert np.isnan(d.loc[1,'strategy_nav']) and np.isnan(d.loc[2,'strategy_return'])
    assert d.loc[2,'weight'] == .5 and d.loc[2,'expired']
    # Jan 3 rate is unavailable on Jan 3 itself; applies to Jan 4 onward.
    cash = 1.01 * (1.01 ** 3) - .1
    assert d.loc[2,'strategy_cash'] == pytest.approx(cash * (1 + .5*.02))
    assert d.loc[1,'official_put_return'] == pytest.approx(102/101-1)
    assert len(result['expiry']) == 1
    assert result['expiry'].iloc[0]['start_date'] == d.date.iloc[0]


def test_sensitivity_and_input_validation():
    market, signals = fixture()
    result = s.run_daily_strategy(market, signals, start='2025-01-03', mark_policy='ask_stale')
    assert result['daily'].loc[1,'mark_source'] == 'stale fallback'
    assert result['daily'].strategy_nav.notna().all()
    signals.loc[0,'weight'] = 2
    with pytest.raises(ValueError):
        s.run_daily_strategy(market, signals)
