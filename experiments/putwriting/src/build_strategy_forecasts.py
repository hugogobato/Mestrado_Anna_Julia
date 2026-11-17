"""Combine preserved statistical forecasts with a clean, separately refit neural run."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[1]
COLS = ['model', 'window_id', 'origin_date', 'target_date', 'horizon', 'y', 'y_hat']


def validate_common_panel(out, expected_models=8):
    if out.duplicated(['model', 'origin_date']).any():
        raise ValueError('duplicate forecast origin')
    counts = out.groupby('model').size()
    if counts.nunique() != 1 or len(counts) != expected_models:
        raise ValueError(f'forecast coverage differs: {counts.to_dict()}')
    keys = ['origin_date', 'target_date', 'horizon', 'window_id']
    reference = out.loc[out.model == counts.index[0], keys + ['y']].sort_values(keys).reset_index(drop=True)
    for model, group in out.groupby('model'):
        group = group.sort_values(keys).reset_index(drop=True)
        if not group[keys].equals(reference[keys]) or not np.allclose(group.y, reference.y, rtol=1e-12, atol=1e-14):
            raise ValueError(f'{model}: origin/target/horizon/window or realization differs')
    return counts


def run(base=BASE):
    base = Path(base)
    archived = base / 'data/archived_forecasts_h22.parquet'
    original = pd.read_parquet(archived)
    frames = [original.loc[~original.model.isin(['LSTM', 'TSMixerX']), COLS]]
    paths = [archived]
    for name in ('lstm', 'tsmixerx'):
        path = base / f'results/neural_purged/forecasts/{name}.parquet'
        frame = pd.read_parquet(path)
        frames.append(frame.loc[frame.horizon == 22, COLS])
        paths.append(path)
    out = pd.concat(frames, ignore_index=True).sort_values(['model', 'origin_date'])
    counts = validate_common_panel(out)
    target = base / 'data/strategy_forecasts_h22.parquet'
    out.to_parquet(target, index=False)
    meta = dict(description='Statistical forecasts preserved; LSTM and TSMixerX replaced by a separate run with purged train/validation boundaries and fresh Optuna study.',
                inputs_sha256={str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                output_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                coverage=counts.to_dict())
    (base / 'data/strategy_forecasts_provenance.json').write_text(
        json.dumps(meta, indent=2) + '\n', encoding='utf8')
    print(counts.to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=BASE)
    run(parser.parse_args().base)
