"""Numerical checks for nested HAR-VIX forecasts and Clark-West."""
import importlib.util
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, SRC / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = load("bench", "11_statistical_benchmarks.py")
evalmod = load("evaluate", "15_evaluate_daily.py")


class RevisionModels(unittest.TestCase):
    def test_vix_transform_and_no_future_information(self):
        rng = np.random.default_rng(12)
        panel = pd.DataFrame({"rv_yz_22": np.exp(rng.normal(-3, .3, 250)),
                              "vix": rng.uniform(10, 40, 250)})
        train, test = np.arange(25, 120), np.arange(160, 170)
        pred, beta = bench._har_vix_predict(panel, train, test)
        x = np.column_stack([bench._har_design(panel, test), (panel.vix.to_numpy()[test] / 100) ** 2])
        np.testing.assert_allclose(pred, np.exp(np.clip(x @ beta, -30, 10)))
        changed = panel.copy()
        changed.loc[171:, ["rv_yz_22", "vix"]] *= 100
        pred2, _ = bench._har_vix_predict(changed, train, test)
        np.testing.assert_allclose(pred, pred2)
        changed.loc[test[0], "vix"] = np.nan
        with self.assertRaises(ValueError):
            bench._har_vix_predict(changed, train, test)

    def test_xgboost_dispatch_keeps_selection_across_windows(self):
        panel = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=3)})
        windows = [{"window_id": i, "test_year": 2020 + i,
                    "train_idx": np.array([0]), "test_idx": np.array([1])} for i in range(2)]
        predictions = np.ones((1, 22))
        metadata = {"selected_features": []}
        with tempfile.TemporaryDirectory() as root, \
             patch.object(bench.C, "MODELS_DIR", Path(root) / "models"), \
             patch.object(bench.C, "INPUTS_DIR", Path(root) / "inputs"), \
             patch.object(bench, "ensure_directories"), \
             patch.object(bench.pd, "read_parquet", return_value=panel), \
             patch.object(bench, "rolling_821_windows", return_value=windows), \
             patch.object(bench, "_xgb_predict", return_value=(predictions, [object()], metadata)) as fit, \
             patch.object(bench, "_xgb_matrix", return_value=(np.empty((1, 0)), {"columns": []})), \
             patch.object(pd.DataFrame, "to_parquet"), \
             patch.object(bench, "forecasts_long", return_value=pd.DataFrame()), \
             patch.object(bench, "_save_forecast") as save, \
             patch.object(bench, "combine_outputs") as combine, \
             patch.dict(sys.modules, {"joblib": None}):
            bench.run(models=["XGBoost"], resume=False, time_budget_minutes=0)
        self.assertEqual(fit.call_count, 2)
        self.assertEqual(save.call_count, 2)
        combine.assert_called_once_with(["XGBoost"])

    def test_purged_splits_exclude_boundary_crossing_targets(self):
        import utils
        rng = np.random.default_rng(5)
        dates = pd.bdate_range("2000-01-03", "2012-12-31")
        panel = pd.DataFrame({"date": dates, "rv_yz_22": rng.uniform(.01, .1, len(dates))})
        legacy = utils.rolling_821_windows(panel)
        purged = utils.rolling_821_windows(panel, purge_boundaries=True)
        self.assertEqual(len(legacy), len(purged))
        for old, new in zip(legacy, purged):
            self.assertEqual(len(old["train_idx"]) - len(new["train_idx"]), 22)
            self.assertEqual(len(old["val_idx"]) - len(new["val_idx"]), 22)
            np.testing.assert_array_equal(old["test_idx"], new["test_idx"])
            self.assertLess(new["train_idx"][-1] + 22, old["val_idx"][0])
            self.assertLess(new["val_idx"][-1] + 22, old["test_idx"][0])

    def test_cw_formula_one_sided_and_matching(self):
        rng = np.random.default_rng(17)
        n = 100
        y = rng.uniform(.01, .08, n)
        f0 = rng.uniform(.02, .06, n)
        f1 = rng.uniform(.02, .06, n)
        dates = pd.date_range("2020-01-01", periods=n)
        base = pd.DataFrame({"origin_date": dates, "target_date": dates + pd.Timedelta(days=1),
                             "horizon": 1, "window_id": 0, "y": y})
        a = base.assign(model="HAR", y_hat=f0)
        b = base.assign(model="HAR-VIX", y_hat=f1)
        frame = pd.concat([a, b])
        result = evalmod.clark_west_nested(frame).iloc[0]
        adjusted = (y-f0)**2 - (y-f1)**2 + (f0-f1)**2
        stat, _ = evalmod._hac_mean_test(adjusted, 21)
        self.assertAlmostEqual(result.mean_adjusted_loss_difference, adjusted.mean())
        self.assertAlmostEqual(result.CW_stat, stat)
        self.assertAlmostEqual(result.CW_pvalue_one_sided, norm.sf(stat))
        self.assertEqual(result.hac_lags, 21)
        b.loc[b.index[0], "y"] *= 2
        with self.assertRaises(ValueError):
            evalmod.clark_west_nested(pd.concat([a, b]))
        with self.assertRaises(ValueError):
            evalmod.clark_west_nested(pd.concat([a, b.iloc[1:]]))


if __name__ == "__main__":
    unittest.main()
