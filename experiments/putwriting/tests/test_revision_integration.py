"""Independent checks of the calendar and causal interest/benchmark joins."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from run_revision import cash_growth, official_put_returns
from build_strategy_forecasts import validate_common_panel


def test_daily_cash_uses_strictly_prior_quote_and_future_changes_do_not_leak():
    rates = pd.DataFrame({'date': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03']),
                          'yield_decimal': [.036, .072, .99]})
    factor = cash_growth('2020-01-02', '2020-01-04', rates)
    assert factor == pytest.approx((1 + .036/360) * (1 + .072/360))
    rates.loc[2, 'yield_decimal'] = -100
    assert cash_growth('2020-01-02', '2020-01-04', rates) == factor


def test_stale_cash_quote_rejected():
    rates = pd.DataFrame({'date': pd.to_datetime(['2020-01-01']), 'yield_decimal': [.03]})
    with pytest.raises(ValueError, match='missing causal'):
        cash_growth('2020-01-10', '2020-01-11', rates)


def test_official_put_uses_exact_period_endpoints():
    periods = pd.DataFrame({'entry_date': pd.to_datetime(['2020-01-01', '2020-02-01']),
                             'expiry_date': pd.to_datetime(['2020-02-01', '2020-03-01'])})
    market = {'put_index': pd.DataFrame({'date': pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01']),
                                         'close': [100., 90., 99.]})}
    assert official_put_returns(periods, market).tolist() == pytest.approx([-.1, .1])
    market['put_index'] = market['put_index'].iloc[:-1]
    with pytest.raises(ValueError, match='exact-date'):
        official_put_returns(periods, market)


def test_equal_counts_do_not_hide_incompatible_forecast_targets():
    a = pd.DataFrame({'model': ['A', 'A'], 'origin_date': pd.to_datetime(['2020-01-01', '2020-01-02']),
                      'target_date': pd.to_datetime(['2020-02-01', '2020-02-02']),
                      'window_id': [0, 0], 'horizon': [22, 22], 'y': [.1, .2]})
    b = a.assign(model='B')
    validate_common_panel(pd.concat([a, b]), expected_models=2)
    b.loc[0, 'target_date'] += pd.Timedelta(days=1)
    with pytest.raises(ValueError, match='differs'):
        validate_common_panel(pd.concat([a, b]), expected_models=2)
