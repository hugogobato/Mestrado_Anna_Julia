"""Ajusta TSMixerX/LSTM nas janelas 8/2/1 com retomada e orçamento de sessão."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from feature_selection import select_features
from models import DirectVolatilityRegressor
from utils import (build_sequences, ensure_directories, forecasts_long,
                   rolling_821_windows, save_json)


def _artifact_name(path: Path) -> str:
    return str(path.relative_to(C.EXPERIMENT_ROOT)) if path.is_relative_to(C.EXPERIMENT_ROOT) else str(path)


def _partial_path(model: str, window_id: int) -> Path:
    path = C.FORECAST_DIR / "partial" / f"{model}_window_{window_id:02d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _model_params(model_name: str, best: dict, max_steps: int, seed: int) -> dict:
    params = dict(best["best_params"])
    params.pop("n_features", None)
    if model_name == "lstm":
        return {
            "model_type": "lstm",
            "input_size": int(params.get("input_size", 66)),
            "hidden_size": 64,
            "dropout": float(params.get("dropout", 0.1)),
            "learning_rate": 1e-3,
            "batch_size": int(params.get("batch_size", 64)),
            "max_steps": max_steps,
            "loss": "huber",
            "val_check_steps": 20,
            "early_stop_patience": 10,
            "seed": seed,
        }
    return {
        "model_type": "tsmixerx",
        "input_size": int(params["input_size"]),
        "n_block": int(params["n_block"]),
        "ff_dim": int(params["ff_dim"]),
        "dropout": float(params["dropout"]),
        "learning_rate": float(params["learning_rate"]),
        "batch_size": int(params["batch_size"]),
        "max_steps": max_steps,
        "loss": params["loss"],
        "val_check_steps": 20,
        "early_stop_patience": 10,
        "seed": seed,
    }


def _save_inputs(panel: pd.DataFrame, selected: list[str], origins: np.ndarray,
                 input_size: int, model_name: str, window_id: int) -> Path:
    start = max(0, int(origins[0]) - input_size + 1)
    end = int(origins[-1]) + 1
    columns = ["date", *selected]
    inputs = panel.loc[start:end - 1, columns].copy()
    inputs["window_id"] = window_id
    inputs["model"] = model_name
    path = C.INPUTS_DIR / f"{model_name}_window_{window_id:02d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    inputs.to_parquet(path, index=False)
    return path


def _combine(model_name: str) -> None:
    parts = sorted((_partial_path(model_name, 0).parent).glob(f"{model_name}_window_*.parquet"))
    if not parts:
        return
    out = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    out.to_parquet(C.FORECAST_DIR / f"{model_name}.parquet", index=False)
    out.to_csv(C.FORECAST_DIR / f"{model_name}.csv", index=False)


def run(models: list[str], max_steps: int | None = None,
        max_windows: int | None = None, window_ids: list[int] | None = None,
        time_budget_minutes: float = 60.0, resume: bool = True,
        seed: int = C.SEED) -> None:
    ensure_directories()
    best_path = C.HP_DIR / "best_config.json"
    if not best_path.exists():
        raise FileNotFoundError("Execute primeiro 12_tsmixerx_optuna.py; best_config.json ausente.")
    best = json.loads(best_path.read_text(encoding="utf-8"))
    max_steps = int(max_steps or best.get("max_steps", 400))
    panel = pd.read_parquet(C.DAILY_PANEL).sort_values("date").reset_index(drop=True)
    windows = rolling_821_windows(panel)
    if window_ids is not None:
        windows = [w for w in windows if w["window_id"] in window_ids]
    if max_windows is not None:
        windows = windows[:max_windows]
    top_k = best["best_params"].get("n_features", "all")
    started = time.monotonic()
    # Não inicia um novo ajuste nos dez minutos finais do orçamento. A margem
    # é destinada à finalização do ajuste corrente, manifestos e ZIP.
    start_buffer = min(10.0, max(1.0, time_budget_minutes * 0.20))
    safe_start_limit = time_budget_minutes - start_buffer
    manifest_path = C.MODELS_DIR / "neural_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"artifacts": []}

    for w in windows:
        for model_name in models:
            model_name = model_name.lower()
            partial = _partial_path(model_name, w["window_id"])
            if resume and partial.exists():
                print(f"Skip {model_name} janela {w['window_id']:02d}: já concluída")
                continue
            elapsed = (time.monotonic() - started) / 60.0
            if elapsed >= safe_start_limit:
                print("Limite seguro atingido. Nenhum novo ajuste será iniciado; "
                      "os artefatos concluídos já estão no Drive.")
                for name in models:
                    _combine(name.lower())
                return

            params = _model_params(model_name, best, max_steps, seed)
            input_size = params["input_size"]
            selected, ranking = select_features(panel, w["train_idx"], top_k)
            ranking_dir = C.RESULTS_DIR / "feature_selection"
            ranking_dir.mkdir(parents=True, exist_ok=True)
            ranking.to_csv(ranking_dir / f"ranking_{model_name}_window_{w['window_id']:02d}.csv", index=False)

            # O contexto de uma sequência de treino não pode começar antes da
            # janela de oito anos. Validação e teste podem usar o histórico já
            # observado das partições anteriores.
            train_origins = w["train_idx"][w["train_idx"] >= w["train_idx"][0] + input_size - 1]
            train_x, train_y, _ = build_sequences(panel, train_origins, selected, input_size)
            val_x, val_y, _ = build_sequences(panel, w["val_idx"], selected, input_size)
            test_x, _, test_origins = build_sequences(panel, w["test_idx"], selected, input_size)
            print(f"{model_name} janela {w['window_id']:02d} ({w['test_year']}): "
                  f"train={len(train_x)}, val={len(val_x)}, test={len(test_x)}, features={len(selected)}")

            estimator = DirectVolatilityRegressor(**params)
            estimator.fit(train_x, train_y, val_x, val_y)
            pred = estimator.predict(test_x)
            artifact = C.MODELS_DIR / model_name / f"{model_name}_window_{w['window_id']:02d}.pt"
            metadata = {
                **{k: v for k, v in w.items() if not k.endswith("_idx")},
                "selected_features": selected,
                "target": C.TARGET,
                "horizon": C.HORIZON,
                "target_transform": "log variance",
            }
            estimator.save(artifact, selected, metadata)
            history = estimator.history_frame(model=model_name, window_id=w["window_id"],
                                                test_year=w["test_year"], seed=seed)
            history.to_csv(C.HISTORY_DIR / f"{model_name}_window_{w['window_id']:02d}.csv", index=False)
            inputs_path = _save_inputs(panel, selected, test_origins, input_size,
                                       model_name, w["window_id"])

            # Valida imediatamente que o artefato sobrevive à serialização.
            loaded, _ = DirectVolatilityRegressor.load(artifact)
            reload_pred = loaded.predict(test_x[:min(2, len(test_x))])
            if not np.allclose(reload_pred, pred[:len(reload_pred)], rtol=1e-5, atol=1e-8):
                raise RuntimeError(f"Reload divergente em {artifact}")

            extra_columns = [c for c in selected if c != C.TARGET]
            for common in ["vix", "spx_close", "close_variance_forward_22",
                           "close_volatility_forward_22"]:
                if common not in extra_columns:
                    extra_columns.append(common)
            frame = forecasts_long(panel, test_origins, pred,
                                   "TSMixerX" if model_name == "tsmixerx" else "LSTM",
                                   w["window_id"], _artifact_name(artifact),
                                   extra_origin_columns=extra_columns)
            frame.to_parquet(partial, index=False)
            frame.to_csv(partial.with_suffix(".csv"), index=False)
            manifest["artifacts"].append({
                "model": model_name,
                "window_id": w["window_id"],
                "test_year": w["test_year"],
                "artifact": _artifact_name(artifact),
                "inputs": _artifact_name(inputs_path),
                "history": _artifact_name(C.HISTORY_DIR / f"{model_name}_window_{w['window_id']:02d}.csv"),
                "best_step": int(history.loc[history.validation_qlike.idxmin(), "step"]),
                "stopped_early": bool(history.stopped_early.any()),
                "reload_verified": True,
            })
            save_json(manifest, manifest_path)
            _combine(model_name)
    for model_name in models:
        _combine(model_name.lower())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=["tsmixerx", "lstm"],
                        default=["tsmixerx", "lstm"])
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--window-ids", type=int, nargs="*")
    parser.add_argument("--time-budget-minutes", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    run(args.models, args.max_steps, args.max_windows, args.window_ids,
        args.time_budget_minutes, not args.no_resume, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
