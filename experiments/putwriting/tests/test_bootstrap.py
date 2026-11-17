import numpy as np
import pandas as pd
import pytest
from experiments.putwriting.src.bootstrap import paired_block_bootstrap


def test_paired_and_reproducible():
    a = np.sin(np.arange(120)); p = pd.DataFrame({'strategy_return':a+.01, 'put_return':a})
    x = paired_block_bootstrap(p, (3,6,12,200), 100, 3)
    pd.testing.assert_frame_equal(x, paired_block_bootstrap(p, (3,6,12,200), 100, 3))
    assert x.loc[0,'ci_low'] == pytest.approx(.01)
    assert x.loc[0,'ci_high'] == pytest.approx(.01)
    assert x.loc[0,'status'] == 'ok'
    assert x.loc[2,'status'].startswith('fewer')
    assert x.loc[3,'status'] == 'block exceeds sample'


def test_index_paths_reproducible_contiguous_and_length():
    from experiments.putwriting.src.bootstrap import bootstrap_index_paths
    log_returns = np.arange(1, 11) * .001
    prices = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=11),
                           'close': 100 * np.exp(np.r_[0, np.cumsum(log_returns)])})
    draws, paths, metadata = bootstrap_index_paths(prices, n_bootstrap=4, num_blocks=3, seed=4)
    draws2, paths2, meta2 = bootstrap_index_paths(prices, n_bootstrap=4, num_blocks=3, seed=4)
    pd.testing.assert_frame_equal(draws, draws2)
    pd.testing.assert_frame_equal(paths, paths2)
    assert metadata == meta2 and metadata['block_length'] == 4
    assert paths.shape == (11, 4) and (paths.to_numpy() > 0).all()
    # Within each full block, adjacent sampled observations are adjacent in source.
    sampled = np.diff(np.log(paths.iloc[:, 0]))
    assert np.allclose(np.diff(sampled[:4]), .001)
    assert np.allclose(np.diff(sampled[4:8]), .001)


def test_bootstrap_fraction_is_not_called_p_value():
    data = pd.DataFrame({'strategy_return': [0., .1, .2], 'put_return': [0., 0., 0.]})
    out = paired_block_bootstrap(data, (1,), n_bootstrap=5)
    assert 'bootstrap_fraction_mean_diff_le_zero' in out.columns
    assert 'p_mean_diff_le_zero' not in out.columns
