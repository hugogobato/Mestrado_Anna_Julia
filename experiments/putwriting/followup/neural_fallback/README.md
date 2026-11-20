# Prior-configuration fallback

The hyperparameters were frozen from the earlier purged study and were selected on its older feature set, which included variables subsequently excluded in this follow-up. The models here are freshly fitted on the corrected panel, not copied from the prior run. Treat this as a prior-configuration fallback, not a fresh search on the corrected data. See ../neural_clean/README.md for the train-only Duan correction and artifact details. Check rolling.log and model manifests for completion; partial artifacts are not a complete backtest.

From the repository root, after preparing `followup/results/panel.parquet`, reproduce into a fresh results directory (Python environment must contain NumPy, pandas, pyarrow, scikit-learn and PyTorch):

```sh
mkdir -p experiments/putwriting/followup/neural/data experiments/putwriting/followup/neural_fallback_repro/hp_search
ln -sfn ../../results/panel.parquet experiments/putwriting/followup/neural/data/daily_panel.parquet
cp experiments/putwriting/followup/neural_fallback/results/hp_search/best_config_purged.json experiments/putwriting/followup/neural_fallback_repro/hp_search/
DAILY_H22_DATA_DIR="$PWD/experiments/putwriting/followup/neural/data" DAILY_H22_RESULTS_DIR="$PWD/experiments/putwriting/followup/neural_fallback_repro" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python experiments/putwriting/followup/neural_src/13_neural_rolling.py --time-budget-minutes 120
```

The copied JSON is the frozen earlier study artifact and contains its original provenance, trial count and search results. `neural_resume_policy.json` fingerprints that artifact and the current input panel and source. For the corrected study see `../neural_clean/`; its search and resulting models are separate. `y_hat_log` is the clipped fitted log prediction; validation-based early stopping and tuning use the raw forecasts, whereas final forecast output applies the in-sample training factors saved in each checkpoint. When targets fall beyond the final session, target date and realized target are missing.

The `artifact`, `inputs`, and `history` paths in the JSON model manifest, and `artifact_path` in the forecasts, are relative to the repository's `experiments/` directory.
