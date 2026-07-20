"""Análises suplementares do TSMixerX com configuração HP fixa.

As análises não refazem a busca Optuna. Elas reutilizam a configuração e o
conjunto de features selecionados na busca principal, evitando contaminar as
comparações de robustez. São produzidas três saídas: painel com VIX3M desde
2007, janela expanding e repetição por seeds.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from utils import mae, mse, qlike, rolling_indices


def _load_tsmixer_module():
    path = Path(__file__).with_name("03_tsmixerx.py")
    spec = importlib.util.spec_from_file_location("tsmixerx_pipeline", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Não foi possível importar {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_hp_config() -> tuple[dict, list[str]]:
    path = C.HP_SEARCH_DIR / "best_config.json"
    if not path.exists():
        raise FileNotFoundError("Execute 03_tsmixerx.py antes de executar as robustezes.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = dict(payload["best_params"])
    params.pop("n_features", None)
    return params, list(payload["selected_features"])


def _prepare_panel(path: Path) -> pd.DataFrame:
    panel = pd.read_parquet(path).sort_values("date").reset_index(drop=True)
    return panel


def _run_variant(panel: pd.DataFrame, variant: str, model_params: dict,
                 selected: list[str], seed: int, max_steps: int,
                 expanding: bool = False) -> pd.DataFrame:
    tm = _load_tsmixer_module()
    panel = tm._add_target_lags(panel)
    missing = [c for c in selected if c not in panel.columns]
    if missing:
        raise KeyError(f"Features da busca ausentes na robustez {variant}: {missing}")
    y = panel[C.TARGET].to_numpy(dtype=float)
    rows = []
    for window_id, info in enumerate(rolling_indices(panel, expanding=expanding)):
        tr0, tr1 = info["train_idx"]
        va0, va1 = info["val_idx"]
        te0, te1 = info["test_idx"]
        fit_idx = np.arange(tr0, tr1, dtype=int)
        validation_idx = np.arange(va0, va1, dtype=int)
        test_idx = np.arange(te0, te1, dtype=int)
        x, _, _ = tm._matrix(panel, selected, fit_idx)
        params = dict(model_params)
        params.update({"max_steps": max_steps, "seed": seed})
        model = tm.TSMixerXRegressor(**params)
        model.fit(x, y, fit_idx, validation_positions=validation_idx)
        pred = model.predict(x, test_idx)
        rows.append(pd.DataFrame({
            "model": "TSMixerX",
            "variant": variant,
            "seed": seed,
            "window_id": window_id,
            "date": panel.loc[test_idx, "date"].to_numpy(),
            "y": y[test_idx],
            "y_hat": pred,
        }))
    return pd.concat(rows, ignore_index=True)


def _summary(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in forecasts.groupby(["variant", "seed"], dropna=False):
        y = group.y.to_numpy(float)
        pred = group.y_hat.to_numpy(float)
        rows.append({"variant": keys[0], "seed": keys[1], "n": len(group),
                     "QLIKE": qlike(y, pred), "MSE": mse(y, pred), "MAE": mae(y, pred),
                     "mean_prediction": float(np.mean(pred))})
    return pd.DataFrame(rows)


def run(panel_path: Path = C.DATA_PARQUET, output_dir: Path = C.RESULTS_DIR / "robustness",
        seeds: list[int] | None = None, max_steps: int = C.MAX_STEPS_DEFAULT,
        smoke: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    seeds = seeds or [C.SEED, 123, 2024]
    if smoke:
        seeds = seeds[:1]
        max_steps = min(max_steps, 30)
    params, selected = _load_hp_config()
    base = _prepare_panel(panel_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    vix3m_panel = base.loc[
        (base.date >= pd.Timestamp("2007-01-01")) & base.vix3m_mean.notna()
    ].reset_index(drop=True)
    vix3m = _run_variant(vix3m_panel, "vix3m_2007_plus", params, selected, seeds[0], max_steps)
    expanding = _run_variant(base, "expanding_window", params, selected, seeds[0], max_steps, expanding=True)
    seed_frames = [_run_variant(base, "seed_sensitivity", params, selected, seed, max_steps)
                   for seed in seeds]
    seed_forecasts = pd.concat(seed_frames, ignore_index=True)
    all_forecasts = pd.concat([vix3m, expanding, seed_forecasts], ignore_index=True)

    for name, frame in [("forecasts_tsmixerx_vix3m", vix3m),
                        ("forecasts_tsmixerx_expanding", expanding),
                        ("forecasts_tsmixerx_seeds", seed_forecasts)]:
        frame.to_parquet(output_dir / f"{name}.parquet", index=False)
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    summary = _summary(all_forecasts)
    summary.to_csv(output_dir / "metrics_robustness.csv", index=False)
    (output_dir / "robustness_manifest.json").write_text(json.dumps({
        "hp_config": str(C.HP_SEARCH_DIR / "best_config.json"),
        "fixed_features": selected,
        "seeds": seeds,
        "max_steps": max_steps,
        "vix3m_sample_start": "2007-01-01",
        "expanding_window": True,
    }, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    return all_forecasts, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=C.DATA_PARQUET)
    parser.add_argument("--output-dir", type=Path, default=C.RESULTS_DIR / "robustness")
    parser.add_argument("--seeds", type=int, nargs="+", default=[C.SEED, 123, 2024])
    parser.add_argument("--max-steps", type=int, default=C.MAX_STEPS_DEFAULT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.panel, args.output_dir, args.seeds, args.max_steps, args.smoke)
    return 0


if __name__ == "__main__":
    sys.exit(main())
