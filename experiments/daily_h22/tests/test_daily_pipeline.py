"""Testes rápidos das invariantes metodológicas do experimento v2."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import config as C  # noqa: E402
from models import DirectVolatilityRegressor  # noqa: E402
from utils import (compute_yang_zhang, rolling_821_windows,  # noqa: E402
                   target_matrix, valid_origin_mask)


class TestYangZhang(unittest.TestCase):
    def test_matches_direct_formula(self):
        rng = np.random.default_rng(123)
        n = 80
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        open_ = np.r_[close[0], close[:-1] * np.exp(rng.normal(0, 0.003, n - 1))]
        high = np.maximum(open_, close) * np.exp(rng.uniform(0.001, 0.02, n))
        low = np.minimum(open_, close) * np.exp(-rng.uniform(0.001, 0.02, n))
        frame = pd.DataFrame({"spx_open": open_, "spx_high": high,
                              "spx_low": low, "spx_close": close})
        result = compute_yang_zhang(frame, window=22)

        end = 50
        sl = slice(end - 21, end + 1)
        overnight = np.log(open_ / np.r_[np.nan, close[:-1]])
        open_close = np.log(close / open_)
        high_open = np.log(high / open_)
        low_open = np.log(low / open_)
        rs = high_open * (high_open - open_close) + low_open * (low_open - open_close)
        k = 0.34 / (1.34 + 23 / 21)
        expected = 252 * (np.var(overnight[sl], ddof=1)
                          + k * np.var(open_close[sl], ddof=1)
                          + (1 - k) * np.mean(rs[sl]))
        self.assertAlmostEqual(float(result.iloc[end]), float(expected), places=12)


class TestAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not C.DAILY_PANEL.exists():
            raise unittest.SkipTest("daily_panel.parquet ainda não foi gerado")
        cls.panel = pd.read_parquet(C.DAILY_PANEL).sort_values("date").reset_index(drop=True)

    def test_h22_is_exactly_origin_plus_22(self):
        mask = valid_origin_mask(self.panel)
        origins = np.flatnonzero(mask)[100:110]
        targets = target_matrix(self.panel, origins)
        for row, origin in enumerate(origins):
            self.assertEqual(targets[row, -1], self.panel.loc[origin + 22, C.TARGET])

    def test_windows_are_821(self):
        windows = rolling_821_windows(self.panel)
        self.assertEqual(len(windows), 16)
        for window in windows:
            self.assertEqual(window["train_year_end"] - window["train_year_start"] + 1, 8)
            self.assertEqual(window["val_year_end"] - window["val_year_start"] + 1, 2)
            self.assertEqual(self.panel.loc[window["test_idx"], "date"].dt.year.nunique(), 1)

    def test_vix3m_not_in_main_features(self):
        from utils import feature_columns

        self.assertNotIn("vix3m", feature_columns(self.panel))


class TestNeuralArtifact(unittest.TestCase):
    def test_fit_predict_save_reload(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch não instalado")
        rng = np.random.default_rng(7)
        train_x = rng.normal(size=(32, 5, 4)).astype(np.float32)
        train_y = np.exp(rng.normal(-3.5, 0.2, size=(32, 22))).astype(np.float32)
        val_x = rng.normal(size=(12, 5, 4)).astype(np.float32)
        val_y = np.exp(rng.normal(-3.5, 0.2, size=(12, 22))).astype(np.float32)
        model = DirectVolatilityRegressor(input_size=5, n_block=1, ff_dim=8,
                                          max_steps=2, val_check_steps=1,
                                          early_stop_patience=2, batch_size=8,
                                          device="cpu")
        model.fit(train_x, train_y, val_x, val_y)
        expected = model.predict(val_x[:2])
        self.assertEqual(expected.shape, (2, 22))
        self.assertTrue((expected > 0).all())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            model.save(path, ["y", "a", "b", "c"])
            loaded, _ = DirectVolatilityRegressor.load(path, device="cpu")
            actual = loaded.predict(val_x[:2])
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    unittest.main()

