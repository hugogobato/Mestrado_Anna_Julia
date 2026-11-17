"""Independently reconcile persisted strategy outputs and their input hashes."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from market_data import load_market_data
from run_revision import ROOT, EXPERIMENT, cash_growth


def run():
    output = EXPERIMENT / 'results'
    manifest = json.loads((output / 'run_manifest.json').read_text())
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == value
              for name, value in manifest['source_code_sha256'].items()}
    if not all(hashes.values()):
        raise AssertionError('source code differs from executed manifest')
    ledgers = pd.read_csv(output / 'expiry_ledgers.csv', parse_dates=['entry_date', 'expiry_date', 'signal_date'])
    market = load_market_data(ROOT / 'BASES DE DADOS-20261117T124409Z-1-001.zip')
    sessions = pd.DatetimeIndex(market['spx'].loc[market['spx'].close.notna(), 'date'])
    assert (ledgers.signal_date < ledgers.entry_date).all()
    assert all(row.signal_date == sessions[sessions < row.entry_date][-1]
               for row in ledgers[['signal_date', 'entry_date']].drop_duplicates().itertuples(index=False))
    assert ledgers.groupby(['model', 'iv_source', 'premium_source']).size().eq(109).all()
    assert len(ledgers.groupby(['model', 'iv_source', 'premium_source'])) == 36
    assert ledgers.entry_date.max() < pd.Timestamp('2025-01-01')
    assert ledgers.weight.between(0, 1).all()
    for name in ['strategy', 'put', 'cash']:
        start = ledgers[f'{name}_start'] * ledgers.gap_growth_factor
        units = ledgers[f'{name}_units']
        collateral = ledgers[f'{name}_collateral']
        assert np.allclose(units * ledgers.strike, collateral)
        assert (collateral <= start + 1e-12).all()
        expected = (start + units * ledgers.premium) * ledgers.cash_growth_factor - units * ledgers.intrinsic
        assert np.allclose(expected, ledgers[f'{name}_end'], rtol=1e-12, atol=1e-12)
    contracts = ledgers[['entry_date', 'expiry_date', 'cash_growth_factor']].drop_duplicates()
    for row in contracts.itertuples(index=False):
        assert np.isclose(cash_growth(row.entry_date, row.expiry_date, market['tbill']), row.cash_growth_factor)
    forecasts = pd.read_parquet(output / 'forecasts_h22.parquet')
    assert forecasts.groupby('model').size().eq(4002).all()
    assert forecasts.model.nunique() == 9
    assert not forecasts.duplicated(['model', 'origin_date']).any()
    pivot = forecasts.pivot(index='origin_date', columns='model', values='y')
    assert np.allclose(pivot, pivot.iloc[:, 0].to_numpy()[:, None])
    assert (ledgers.query("model == 'VIX' and iv_source == 'vix'").weight == 0).all()
    report = dict(status='passed', strategy_rows=len(ledgers), contracts=109,
                  financial_scenarios=36, forecast_models=9, h22_origins_per_model=4002,
                  source_hashes_match=hashes,
                  reconciliations=['previous-session signals', 'no 2025 entries', 'cash-secured collateral',
                      'cash/premium/payoff accounting', 'causal daily interest', 'forecast realization agreement',
                      'VIX forecast and VIX IV imply zero VRP'],
                  scope='deterministic output reconciliation; does not validate missing market metadata or prove a profitable strategy')
    (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    run()
