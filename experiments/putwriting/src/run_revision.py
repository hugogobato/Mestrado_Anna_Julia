"""Reproduce the qualification revisions from supplied data and saved forecasts."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from market_data import load_market_data
from audit_market_data import run as audit_sources
from strategy import build_signals, backtest_puts, expiry_metrics
from bootstrap import paired_block_bootstrap, bootstrap_index_paths

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cash_growth(start, end, rates, basis=360.0):
    """Approximate daily cash accrual using only quotes strictly before each day.

    This is an explicitly assumed simple annual yield, ACT/360, reinvested daily;
    the source does not identify its Bloomberg field or yield/discount convention.
    No quote older than seven calendar days may be used.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if end < start:
        raise ValueError('cash interval is reversed')
    days = pd.DataFrame({'accrual_date': pd.date_range(start, end, inclusive='left')})
    if days.empty:
        return 1.0
    matched = pd.merge_asof(days, rates.sort_values('date'), left_on='accrual_date',
                            right_on='date', allow_exact_matches=False,
                            tolerance=pd.Timedelta(days=7))
    if matched.yield_decimal.isna().any():
        raise ValueError('missing causal T-bill quote within seven days')
    factors = 1 + matched.yield_decimal.to_numpy(float) / basis
    if not np.isfinite(factors).all() or (factors <= 0).any():
        raise ValueError('invalid cash accrual factor')
    return float(np.prod(factors))


def contracts_for_sample(market, calendar, output):
    """Use supplied contracts; record excluded periods without stale signal fills."""
    selected, excluded = [], []
    calendar = pd.DatetimeIndex(calendar).sort_values()
    rates = market['tbill']
    for row in market['rolls'].itertuples(index=False):
        previous = calendar[calendar < row.roll_date]
        reason = None
        signal_date = previous[-1] if len(previous) else pd.NaT
        if not len(previous) or row.roll_date > calendar.max():
            reason = 'no current SPX/IV coverage at entry'
        elif (row.roll_date - signal_date).days > 7:
            reason = 'previous signal session more than seven days old'
        elif row.quote_date != row.roll_date:
            reason = 'quote date differs from entry'
        soq = market['settlement'].loc[market['settlement'].settle_date == row.expiry, 'SOQ']
        if len(soq) != 1 or not np.isclose(soq.iloc[0], row.SOQ_settle):
            reason = 'missing or inconsistent SPX SOQ'
        if reason:
            excluded.append({'roll_date': row.roll_date, 'expiry': row.expiry, 'reason': reason})
            continue
        rate = rates.loc[rates.date < row.roll_date].iloc[-1]
        selected.append(dict(signal_date=signal_date, entry_date=row.roll_date,
                             expiry_date=row.expiry, contract_id=row.option_ticker,
                             strike=row.strike, bid=row.prem_bid, mid=row.prem_mid,
                             settlement=row.SOQ_settle, settlement_type='AM',
                             settlement_source='SOQ', tbill_annual_pct=rate.yield_pct,
                             rate_date=rate.date,
                             cash_growth_factor=cash_growth(row.roll_date, row.expiry, rates)))
    contracts = pd.DataFrame(selected)
    if contracts.empty:
        raise ValueError('no contracts have supported signals and settlement')
    contracts['gap_growth_factor'] = [
        1.0 if i == 0 else cash_growth(contracts.expiry_date.iloc[i-1], row.entry_date, rates)
        for i, row in enumerate(contracts.itertuples(index=False))]
    pd.DataFrame(excluded, columns=['roll_date', 'expiry', 'reason']).to_csv(output / 'excluded_contracts.csv', index=False)
    contracts.to_csv(output / 'contracts_used.csv', index=False)
    return contracts


def official_put_returns(periods, market):
    """Close-to-close official index proxy on the same calendar endpoints.

    The strategy settles in the morning under an explicit assumption, so these
    date-matched closes do not constitute synchronized intraday returns.
    """
    levels = market['put_index'].set_index('date').close
    starts = [periods.entry_date.iloc[0], *periods.expiry_date.iloc[:-1]]
    ends = periods.expiry_date.tolist()
    if not pd.DatetimeIndex(starts + ends).isin(levels.index).all():
        raise ValueError('missing exact-date official PUT index endpoint')
    return np.asarray([levels.loc[b] / levels.loc[a] - 1 for a, b in zip(starts, ends)])


def fit_har(panel, results):
    env = os.environ.copy()
    env.update(DAILY_H22_DATA_DIR=str(panel.parent.resolve()),
               DAILY_H22_RESULTS_DIR=str(results.resolve()),
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    scripts = ROOT / 'experiments/daily_h22/src'
    subprocess.run([sys.executable, str(scripts / '11_statistical_benchmarks.py'),
                    '--models', 'HAR', 'HAR-VIX', '--no-resume',
                    '--time-budget-minutes', '0'], check=True, env=env)
    subprocess.run([sys.executable, str(scripts / '15_evaluate_daily.py'),
                    '--nested-only'], check=True, env=env)


def run(args):
    if not args.assume_am_roll:
        raise ValueError('Pass --assume-am-roll to run the explicitly provisional AM settlement / subsequent same-day entry scenario; the CSV has no intraday timestamps.')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    audit_sources(args.source_zip, output / 'data_audit')
    market = load_market_data(args.source_zip)
    revision = output / 'forecast_revision'
    if not args.skip_forecast_fit:
        fit_har(args.panel, revision)
    else:
        previous = output / 'run_manifest.json'
        if not previous.exists():
            raise ValueError('--skip-forecast-fit requires a completed run manifest')
        prior = json.loads(previous.read_text())
        if prior.get('panel_sha256') != digest(args.panel):
            raise ValueError('panel changed: refit HAR/HAR-VIX before reuse')
        if prior.get('added_har_vix_sha256') != digest(revision / 'forecasts/har-vix.parquet'):
            raise ValueError('HAR-VIX forecast changed: refit before reuse')
    saved = pd.read_parquet(args.forecasts)
    saved = saved.loc[saved.horizon == 22].copy()
    added = pd.read_parquet(revision / 'forecasts/har-vix.parquet')
    added = added.loc[added.horizon == 22].copy()
    forecasts = pd.concat([saved, added], ignore_index=True)
    if forecasts.duplicated(['model', 'origin_date']).any():
        raise ValueError('duplicate model/origin forecasts')
    forecasts.origin_date = pd.to_datetime(forecasts.origin_date)
    forecasts.target_date = pd.to_datetime(forecasts.target_date)
    keep = ['model', 'window_id', 'origin_date', 'target_date', 'horizon', 'y', 'y_hat']
    forecasts[keep].to_parquet(output / 'forecasts_h22.parquet', index=False)
    forecast_metrics = []
    for model, group in forecasts.groupby('model'):
        ratio = group.y / group.y_hat
        forecast_metrics.append(dict(model=model, n=len(group),
            MSE=float(np.mean((group.y - group.y_hat)**2)),
            MAE=float(np.mean(np.abs(group.y - group.y_hat))),
            QLIKE=float(np.mean(ratio - np.log(ratio) - 1))))
    pd.DataFrame(forecast_metrics).to_csv(output / 'forecast_metrics_h22.csv', index=False)
    forecasts.loc[forecasts.model == 'HAR-VIX', keep].nlargest(10, 'y_hat').to_csv(
        output / 'har_vix_extreme_forecasts.csv', index=False)
    # The original forecast panel and supplied source agree on all common SPX closes.
    panel = pd.read_parquet(args.panel)
    joined = panel[['date', 'spx_close']].merge(market['spx'][['date', 'close']], on='date')
    if not np.allclose(joined.spx_close, joined.close, rtol=0, atol=1e-6):
        raise ValueError('forecast panel and market SPX source differ')
    valid_spx = market['spx'].loc[np.isfinite(market['spx'].close) & (market['spx'].close > 0)]
    market['spx'].loc[~market['spx'].index.isin(valid_spx.index)].to_csv(
        output / 'excluded_spx_rows.csv', index=False)
    calendar = valid_spx.date
    contracts = contracts_for_sample(market, calendar, output)
    signal_frames, ledgers, summaries, bootstrap_rows = [], [], [], []
    models = sorted(forecasts.model.unique())
    for model in models:
        pred = forecasts.loc[forecasts.model == model, ['origin_date', 'y_hat']].rename(
            columns={'origin_date': 'date', 'y_hat': 'h22'})
        for iv_name in ('iv', 'vix'):
            frame = pd.DataFrame({'date': calendar})
            frame = frame.merge(pred, on='date', how='left', validate='one_to_one')
            frame = frame.merge(market[iv_name][['date', 'value_pct']].rename(
                columns={'value_pct': 'iv_percent'}), on='date', how='left', validate='one_to_one')
            signals = build_signals(frame, window=504)
            signals['model'], signals['iv_source'] = model, iv_name
            signal_frames.append(signals)
            # Never interpret a missing forecast/IV as a deliberate zero-VRP signal.
            dates = pd.Index(contracts.signal_date)
            selected_signals = signals.set_index('date').loc[dates]
            if selected_signals[['h22', 'iv_percent', 'weight']].isna().any().any():
                raise ValueError(f'{model}/{iv_name}: missing input or incomplete 504-session history at entry')
            for premium_name in ('bid', 'mid'):
                c = contracts.copy()
                c['premium'] = c[premium_name]
                ledger = backtest_puts(signals, c, allow_same_day_am_roll=True)
                ledger['official_put_return'] = official_put_returns(ledger, market)
                ledger['official_put_end'] = np.cumprod(1 + ledger.official_put_return)
                ledger['model'], ledger['iv_source'], ledger['premium_source'] = model, iv_name, premium_name
                ledgers.append(ledger)
                metrics = expiry_metrics(ledger)
                official = ledger.copy()
                official['strategy_return'] = official.official_put_return
                om = expiry_metrics(official).query("portfolio == 'strategy'").copy()
                om['portfolio'] = 'official_PUT_close_proxy'
                metrics = pd.concat([metrics, om], ignore_index=True)
                metrics['model'], metrics['iv_source'], metrics['premium_source'] = model, iv_name, premium_name
                summaries.append(metrics)
                block_sizes = sorted({1, 3, 6, max(1, int(np.ceil(len(ledger) / 20)))})
                for comparator in ('put_return', 'official_put_return'):
                    boot = paired_block_bootstrap(ledger, block_sizes=block_sizes,
                                                  n_bootstrap=args.bootstrap_reps,
                                                  seed=args.seed, comparator_col=comparator)
                    boot['model'], boot['iv_source'], boot['premium_source'] = model, iv_name, premium_name
                    boot['comparator'] = comparator
                    bootstrap_rows.append(boot)
    pd.concat(signal_frames, ignore_index=True).to_parquet(output / 'daily_signals.parquet', index=False)
    all_ledgers = pd.concat(ledgers, ignore_index=True)
    all_ledgers.to_csv(output / 'expiry_ledgers.csv', index=False)
    all_ledgers.to_parquet(output / 'expiry_ledgers.parquet', index=False)
    metrics = pd.concat(summaries, ignore_index=True)
    metrics.to_csv(output / 'performance_expiry.csv', index=False)
    pd.concat(bootstrap_rows, ignore_index=True).to_csv(output / 'bootstrap_paired_monthly.csv', index=False)
    index_draws, index_paths, index_meta = bootstrap_index_paths(
        valid_spx[['date', 'close']], n_bootstrap=args.bootstrap_reps,
        num_blocks=20, seed=args.seed)
    index_draws.to_csv(output / 'bootstrap_index_draws.csv', index=False)
    index_paths.to_csv(output / 'bootstrap_index_example_paths.csv', index=True)
    (output / 'bootstrap_index_config.json').write_text(
        json.dumps(index_meta, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    plot_curves(all_ledgers, output)
    manifest = dict(source_zip_sha256=digest(args.source_zip),
                    saved_forecasts_sha256=digest(args.forecasts),
                    added_har_vix_sha256=digest(revision / 'forecasts/har-vix.parquet'),
                    panel_sha256=digest(args.panel), models=models,
                    contracts=len(contracts), first_entry=str(contracts.entry_date.min().date()),
                    last_expiry=str(contracts.expiry_date.max().date()),
                    bootstrap_reps=args.bootstrap_reps, seed=args.seed,
                    quantile_sessions=504, signal_timing='previous SPX session close',
                    same_day_am_roll_assumed=True, option_settlement_type='AM assumed; SOQ values cross-checked',
                    cash_convention='assumed simple annual yield ACT/360, compounded daily using strictly previous quotes, max staleness 7 days',
                    fractional_contracts=True, contract_multiplier='normalized away in index-point cash units',
                    premium_scenarios=['bid', 'mid'], fees_and_taxes=0,
                    metric_frequency='monthly expiry endpoints; sqrt(12) annualization; no daily option marks',
                    official_put_timing='date-matched close proxy, not simultaneous with AM settlement',
                    forecast_interpretation='plug-in variance point forecasts, not established conditional arithmetic means; legacy log forecasts lack smearing correction',
                    neural_forecasts='separate purged-boundary refit; six original statistical forecasts preserved',
                    iv_interpretation='generic Bloomberg LIVE ATM 1M, not verified contract-specific IV; VIX robustness uses its own tenor',
                    horizon_note='forecast 22 sessions from previous close; actual monthly contract tenor varies',
                    bootstrap_scope='paired observed monthly returns conditional on forecasts/signals/contracts, not full pipeline refit',
                    forecast_export_scope='all saved origins 2010-2025; trading uses only entry dates with source IV/SPX coverage, through 2024',
                    forecast_artifact_path_base={'HAR_and_HAR_VIX': 'experiments/daily_h22; supplemental paths use ../putwriting/...',
                                                 'purged_neural': 'repository root; paths begin experiments/putwriting/...'},
                    status='provisional research scenario pending quote timestamps, contract AM identification, rate convention and daily marks')
    manifest['versions'] = {name: importlib.metadata.version(name) for name in
        ['numpy', 'pandas', 'scipy', 'statsmodels', 'scikit-learn', 'pyarrow', 'openpyxl', 'matplotlib', 'arch']}
    manifest['python_version'] = sys.version
    manifest['source_code_sha256'] = {str(path.relative_to(ROOT)): digest(path) for path in
        [*sorted((EXPERIMENT / 'src').glob('*.py')),
         *[ROOT / 'experiments/daily_h22/src' / name for name in
           ['11_statistical_benchmarks.py', '12_tsmixerx_optuna.py', '13_neural_rolling.py',
            '15_evaluate_daily.py', 'config.py', 'utils.py', 'feature_selection.py', 'models.py']]]}
    (output / 'run_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(metrics.query("iv_source == 'iv' and premium_source == 'bid' and portfolio == 'strategy'").to_string(index=False))


def plot_curves(ledgers, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for iv_name in ('iv', 'vix'):
        fig, ax = plt.subplots(figsize=(11, 6))
        subset = ledgers.query('iv_source == @iv_name and premium_source == "bid"')
        for model, frame in subset.groupby('model'):
            ax.plot([frame.entry_date.iloc[0], *frame.expiry_date], [1., *frame.strategy_end], label=model, lw=1)
        last = subset[subset.model == subset.model.iloc[0]]
        ax.plot([last.entry_date.iloc[0], *last.expiry_date], [1., *last.official_put_end], label='PUT oficial (fechamentos)', color='black', lw=2)
        ax.plot([last.entry_date.iloc[0], *last.expiry_date], [1., *last.cash_end], label='Caixa aproximado', color='gray', ls=':')
        ax.set(title=f'Cenário provisório: {iv_name.upper()}, bid, valores nos vencimentos', ylabel='Patrimônio inicial = 1')
        ax.legend(ncol=3, fontsize=8)
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(output / f'equity_expiry_{iv_name}_bid.png', dpi=150)
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-zip', type=Path, default=ROOT / 'BASES DE DADOS-20261117T124409Z-1-001.zip')
    parser.add_argument('--forecasts', type=Path, default=EXPERIMENT / 'data/strategy_forecasts_h22.parquet')
    parser.add_argument('--panel', type=Path, default=EXPERIMENT / 'data/daily_panel.parquet')
    parser.add_argument('--output', type=Path, default=EXPERIMENT / 'results')
    parser.add_argument('--skip-forecast-fit', action='store_true')
    parser.add_argument('--assume-am-roll', action='store_true')
    parser.add_argument('--bootstrap-reps', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    run(parser.parse_args())
