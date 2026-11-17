"""Conditional moving-block bootstrap of paired, realized holding-period returns.

This conditions on the fitted forecasts, historical signals, chosen contracts,
execution prices and observed returns. It does not refit forecasting models or
recompute signals, and block size is measured in rows, not calendar days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def paired_block_bootstrap(periods: pd.DataFrame, block_sizes=(1, 3, 6),
                           n_bootstrap: int = 2000, seed: int = 0,
                           strategy_col: str = 'strategy_return',
                           comparator_col: str = 'put_return') -> pd.DataFrame:
    """Resample the same contiguous blocks for both portfolio return series.

    Percentile intervals describe mean per-period return differences. Each draw
    samples overlapping non-circular historical blocks with replacement and
    truncates the final block to retain the original sample length. Estimates
    with few observed blocks are flagged rather than represented as robust.
    """
    if n_bootstrap < 1:
        raise ValueError('n_bootstrap must be positive')
    x = periods[[strategy_col, comparator_col]].to_numpy(dtype=float)
    if len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('need at least two complete paired observations')
    n = len(x)
    rng = np.random.default_rng(seed)
    rows = []
    for size in block_sizes:
        if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 1:
            raise ValueError('block sizes must be positive integers')
        if size > n:
            rows.append(dict(block_size=size, n_periods=n, nonoverlapping_blocks=n // size,
                             n_bootstrap=n_bootstrap, estimate=np.nan, ci_low=np.nan,
                             ci_high=np.nan, bootstrap_fraction_mean_diff_le_zero=np.nan, status='block exceeds sample'))
            continue
        blocks_per_draw = int(np.ceil(n / size))
        diffs = np.empty(n_bootstrap)
        offsets = np.arange(size)
        for i in range(n_bootstrap):
            starts = rng.integers(0, n - size + 1, size=blocks_per_draw)
            index = (starts[:, None] + offsets).ravel()[:n]
            sample = x[index]
            diffs[i] = np.mean(sample[:, 0] - sample[:, 1])
        count = n // size
        rows.append(dict(block_size=size, n_periods=n, nonoverlapping_blocks=count,
                         n_bootstrap=n_bootstrap, estimate=np.mean(x[:, 0] - x[:, 1]),
                         ci_low=np.quantile(diffs, .025), ci_high=np.quantile(diffs, .975),
                         bootstrap_fraction_mean_diff_le_zero=np.mean(diffs <= 0),
                         status='ok' if count >= 20 else 'fewer than 20 nonoverlapping blocks'))
    return pd.DataFrame(rows)


def bootstrap_index_paths(prices, n_bootstrap: int = 1000, num_blocks: int = 20,
                          seed: int = 42, n_paths: int = 5):
    """Resample contiguous daily index log-return blocks for diagnostics only.

    Returns (draws, paths, metadata). `draws` has one row per simulated path;
    `paths` has the initial price and up to `n_paths` simulated paths as columns
    over the original dates. Blocks are sampled with replacement and the last
    block is truncated to the original number of returns. No option prices,
    strategy returns, trading rules, or forecasting models are recomputed.
    """
    if isinstance(prices, pd.DataFrame):
        p = prices[['date', 'close']].copy()
        dates = pd.to_datetime(p['date'])
        if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise ValueError('dates must be valid, unique, and increasing')
        values = pd.to_numeric(p['close'], errors='raise').to_numpy(dtype=float)
        index = pd.DatetimeIndex(dates, name='date')
    elif isinstance(prices, pd.Series):
        values = pd.to_numeric(prices, errors='raise').to_numpy(dtype=float)
        index = prices.index
        if not index.is_unique or not index.is_monotonic_increasing:
            raise ValueError('price index must be unique and increasing')
    else:
        raise TypeError('prices must be a date/close DataFrame or Series')
    if len(values) < 3 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('at least three finite, positive prices are required')
    if (not isinstance(n_bootstrap, (int, np.integer)) or isinstance(n_bootstrap, bool) or n_bootstrap < 1
            or not isinstance(num_blocks, (int, np.integer)) or isinstance(num_blocks, bool) or num_blocks < 1
            or not isinstance(n_paths, (int, np.integer)) or isinstance(n_paths, bool) or n_paths < 0):
        raise ValueError('invalid simulation counts')
    returns = np.diff(np.log(values))
    n = len(returns)
    block_length = int(np.ceil(n / num_blocks))
    if block_length > n:
        raise ValueError('block length exceeds return sample')
    blocks_per_draw = int(np.ceil(n / block_length))
    rng = np.random.default_rng(seed)
    draws = []
    paths = {}
    for i in range(n_bootstrap):
        starts = rng.integers(0, n - block_length + 1, size=blocks_per_draw)
        ix = (starts[:, None] + np.arange(block_length)).ravel()[:n]
        sampled = returns[ix]
        path = values[0] * np.exp(np.r_[0., np.cumsum(sampled)])
        if not np.isfinite(path).all() or (path <= 0).any():
            raise ValueError('resampled path overflowed or underflowed')
        peak = np.maximum.accumulate(path)
        draws.append({'total_return': path[-1] / path[0] - 1,
                      'daily_max_drawdown': np.min(path / peak - 1),
                      'annualized_log_return_volatility': np.std(sampled, ddof=1) * np.sqrt(252)})
        if i < n_paths:
            paths[f'path_{i+1}'] = path
    metadata = {'scope': 'index paths only; no option repricing, strategy simulation, or model refitting',
                'n_prices': len(values), 'n_returns': n, 'num_blocks_requested': num_blocks,
                'block_length': block_length, 'blocks_per_draw': blocks_per_draw,
                'n_bootstrap': n_bootstrap, 'seed': seed, 'volatility_annualization': 252}
    return pd.DataFrame(draws), pd.DataFrame(paths, index=index), metadata
