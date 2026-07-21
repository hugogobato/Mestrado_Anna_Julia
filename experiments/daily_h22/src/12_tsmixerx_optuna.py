"""Busca Optuna retomável do TSMixerX diário H=22 na primeira janela 8/2."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

import config as C
from feature_selection import select_features
from models import DirectVolatilityRegressor
from utils import (build_sequences, ensure_directories, rolling_821_windows,
                   save_json)


def _trial_params(trial, max_steps: int) -> dict:
    return {
        "model_type": "tsmixerx",
        "input_size": trial.suggest_categorical("input_size", C.INPUT_SIZES),
        "n_block": trial.suggest_categorical("n_block", [2, 3, 4, 6]),
        "ff_dim": trial.suggest_categorical("ff_dim", [32, 64, 128, 256]),
        "dropout": trial.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.3]),
        "learning_rate": trial.suggest_categorical("learning_rate", [1e-4, 3e-4, 1e-3, 3e-3]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "loss": trial.suggest_categorical("loss", ["mae", "mse", "huber"]),
        "max_steps": max_steps,
        "val_check_steps": 20,
        "early_stop_patience": 10,
        "seed": C.SEED,
    }


def run(target_trials: int = 40, max_steps: int = 400,
        timeout_minutes: float = 60.0, storage_path: Path | None = None,
        smoke: bool = False, backup_dir: Path | None = None) -> dict | None:
    import optuna

    ensure_directories()
    panel = pd.read_parquet(C.DAILY_PANEL).sort_values("date").reset_index(drop=True)
    first = rolling_821_windows(panel)[0]
    train, val = first["train_idx"], first["val_idx"]
    if smoke:
        target_trials, max_steps = min(target_trials, 3), min(max_steps, 30)
    storage_path = Path(storage_path or (C.HP_DIR / "tsmixerx_daily.db")).resolve()
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path = None
    if backup_dir is not None:
        backup_dir = Path(backup_dir).resolve()
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / storage_path.name
        if backup_path.exists() and not storage_path.exists():
            with sqlite3.connect(backup_path) as source, sqlite3.connect(storage_path) as destination:
                source.backup(destination)
    study_name = f"daily_h22_steps_{max_steps}"
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=C.SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=100,
                                           interval_steps=20),
    )
    history_dir = C.HP_DIR / "trial_histories" / study_name
    history_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        top_k = trial.suggest_categorical("n_features", [20, 30, 40, 50, "all"])
        features, ranking = select_features(panel, train, top_k)
        params = _trial_params(trial, max_steps)
        train_x, train_y, _ = build_sequences(panel, train, features, params["input_size"])
        val_x, val_y, _ = build_sequences(panel, val, features, params["input_size"])
        if not len(train_x) or not len(val_x):
            raise optuna.TrialPruned("Sem sequências válidas")
        model = DirectVolatilityRegressor(**params)
        try:
            model.fit(train_x, train_y, val_x, val_y, trial=trial)
        finally:
            if model.history:
                model.history_frame(trial_number=trial.number).to_csv(
                    history_dir / f"trial_{trial.number:04d}.csv", index=False)
        trial.set_user_attr("selected_features", features)
        trial.set_user_attr("n_train_sequences", len(train_x))
        trial.set_user_attr("n_val_sequences", len(val_x))
        trial.set_user_attr("best_validation_qlike", model.best_validation_qlike)
        if trial.number == 0:
            ranking.to_csv(C.HP_DIR / "feature_ranking_initial.csv", index=False)
            save_json(ranking.attrs.get("counts", {}), C.HP_DIR / "feature_selection_counts.json")
        return model.best_validation_qlike

    remaining = max(0, target_trials - len(study.trials))
    def backup_callback(_study, _trial):
        if backup_path is None:
            return
        with sqlite3.connect(storage_path) as source, sqlite3.connect(backup_path) as destination:
            source.backup(destination)

    if remaining:
        # Reserva cinco minutos para exportar tabelas, fechar o SQLite e deixar
        # o notebook executar as células de manifesto/compactação.
        optimization_seconds = max(1.0, (timeout_minutes - 5.0) * 60.0)
        study.optimize(objective, n_trials=remaining,
                       timeout=optimization_seconds,
                       callbacks=[backup_callback], gc_after_trial=True,
                       show_progress_bar=True)
    else:
        print(f"Meta de {target_trials} trials já alcançada.")

    study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "duration")).to_csv(
        C.HP_DIR / f"trials_{study_name}.csv", index=False)
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        print("Ainda não há trial completo; execute o notebook novamente.")
        return None
    best = study.best_trial
    top_k = best.params["n_features"]
    features, ranking = select_features(panel, train, top_k)
    ranking.to_csv(C.HP_DIR / "feature_ranking_best.csv", index=False)
    payload = {
        "study_name": study_name,
        "storage": str(storage_path),
        "target_trials": target_trials,
        "trials_recorded": len(study.trials),
        "complete_trials": len(complete),
        "best_trial": best.number,
        "best_validation_qlike": best.value,
        "best_params": best.params,
        "selected_features": features,
        "max_steps": max_steps,
        "hp_window": {k: v for k, v in first.items() if not k.endswith("_idx")},
        "validation_scheme": "fixed first 8-year train / 2-year validation; parameters frozen before all test years",
        "pruning": "MedianPruner with trial.report and trial.should_prune at validation checks",
        "target_transform": "log variance",
    }
    save_json(payload, C.HP_DIR / "best_config.json")
    print(f"Trials: {len(study.trials)}/{target_trials}; completos={len(complete)}")
    print(f"Melhor trial={best.number}; QLIKE validação={best.value:.6f}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-trials", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--timeout-minutes", type=float, default=60.0)
    parser.add_argument("--storage", type=Path)
    parser.add_argument("--backup-dir", type=Path,
                        help="diretório persistente; recebe backup SQLite após cada trial")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.target_trials, args.max_steps, args.timeout_minutes, args.storage,
        args.smoke, args.backup_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
