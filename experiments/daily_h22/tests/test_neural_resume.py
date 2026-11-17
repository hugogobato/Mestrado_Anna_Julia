import importlib.util
import json
from pathlib import Path
import sys

import pytest

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
spec = importlib.util.spec_from_file_location('neural_rolling_resume_test', SRC / '13_neural_rolling.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_guard_requires_matching_policy_for_partial(tmp_path):
    panel = tmp_path / 'panel'; panel.write_bytes(b'panel')
    config = tmp_path / 'best.json'; config.write_text('{}')
    results = tmp_path / 'results'
    guard = lambda seed=42, steps=400: module._guard_resume_policy(results, panel, config, 'purged-v1', seed, steps)
    policy = guard()
    assert json.loads((results / 'neural_resume_policy.json').read_text()) == policy
    partial = results / 'forecasts' / 'partial' / 'tsmixerx_window_00.parquet'
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b'partial')
    assert guard() == policy
    with pytest.raises(ValueError, match='fresh results directory'): guard(seed=43)
    with pytest.raises(ValueError, match='fresh results directory'): guard(steps=399)
    config.write_text('{"changed": true}')
    with pytest.raises(ValueError, match='fresh results directory'): guard()


def test_missing_and_corrupt_policy_rejected(tmp_path):
    panel = tmp_path / 'panel'; panel.write_bytes(b'panel')
    config = tmp_path / 'best'; config.write_bytes(b'config')
    results = tmp_path / 'results'
    partial = results / 'forecasts' / 'partial' / 'lstm_window_01.parquet'
    partial.parent.mkdir(parents=True)
    partial.touch()
    args = (results, panel, config, 'purged-v1', 42, 400)
    with pytest.raises(ValueError, match='fresh results directory'): module._guard_resume_policy(*args)
    (results / 'neural_resume_policy.json').write_text('{')
    with pytest.raises(ValueError, match='fresh results directory'): module._guard_resume_policy(*args)
