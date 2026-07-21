"""Gera os três notebooks Colab do experimento diário H=22."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SETUP = r'''# Configuração compartilhada: repositório + armazenamento persistente
from pathlib import Path
import os, shutil, subprocess, sys

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception as exc:
    print('(Not on Colab / Drive mount skipped):', exc)

REPO_URL = 'https://github.com/hugogobato/Mestrado_Anna_Julia.git'
REPO = Path('/content/Mestrado_Anna_Julia')
if not REPO.exists():
    subprocess.run(['git', 'clone', REPO_URL, str(REPO)], check=True)
else:
    subprocess.run(['git', '-C', str(REPO), 'pull', '--ff-only'], check=True)

EXP = REPO / 'experiments/daily_h22'
PERSIST = Path('/content/drive/MyDrive/Mestrado_Anna_Julia/daily_h22')
DATA_DIR = PERSIST / 'data'
RESULTS_DIR = PERSIST / 'results'
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
os.environ['DAILY_H22_DATA_DIR'] = str(DATA_DIR)
os.environ['DAILY_H22_RESULTS_DIR'] = str(RESULTS_DIR)
os.environ['MPLCONFIGDIR'] = '/content/matplotlib_cache'
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r',
                str(EXP / 'requirements-colab.txt')], check=True)

def run_script(name, *args):
    command = [sys.executable, str(EXP / 'src' / name), *map(str, args)]
    print('RUN:', ' '.join(command))
    subprocess.run(command, check=True, env=os.environ.copy())

print('Experiment:', EXP)
print('Persistent data:', DATA_DIR)
print('Persistent results:', RESULTS_DIR)
try:
    import torch
    print('Torch:', torch.__version__, '| CUDA:', torch.cuda.is_available(),
          '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
except Exception as exc:
    print('Torch check failed:', exc)
'''


SAFE_DOWNLOAD = r'''# Empacotamento e download seguro do estado atual
import shutil
from pathlib import Path

archive_base = Path('/content') / ARCHIVE_NAME
output_file = shutil.make_archive(str(archive_base), 'zip', root_dir=PERSIST)
print('Archive created:', output_file)
try:
    from google.colab import files
    files.download(output_file)
    print("Downloaded:", output_file)
except Exception as e:
    print("(Not on Colab / download skipped):", e)
'''


nb1 = notebook([
    markdown("""# Daily H=22, notebook 1/3: dados e benchmarks estatísticos

Este notebook cria o target Yang–Zhang diário corrigido e executa RW, HAR,
GARCH(1,1), VIX raw, VIX calibrado e XGBoost em janelas 8/2/1. Ele não altera
os resultados da execução v1.

**Tempo estimado no Colab:** 10–20 minutos. O teto operacional é 60 minutos,
deixando ao menos 10 minutos para empacotar e baixar os resultados antes do
limite de 70 minutos."""),
    code(SETUP),
    markdown("""## Configuração da sessão

Use `smoke` para testar uma janela rapidamente. Use `final` para produzir os
resultados completos nas 16 janelas em uma única execução."""),
    code(r'''RUN_MODE = 'final'  # 'smoke' ou 'final'
SESSION_BUDGET_MINUTES = 60
if RUN_MODE not in {'smoke', 'final'}:
    raise ValueError('RUN_MODE deve ser smoke ou final')
print('Mode:', RUN_MODE, '| session budget:', SESSION_BUDGET_MINUTES, 'minutes')
'''),
    markdown("""## Preparação diária

O script gera uma linha por pregão, aplica um atraso conservador às variáveis
macro sem data de publicação e salva o manifesto das 16 janelas 8/2/1."""),
    code(r'''prep_args = ['--smoke'] if RUN_MODE == 'smoke' else []
run_script('10_data_prep_daily.py', *prep_args)

import pandas as pd
display(pd.read_csv(DATA_DIR / 'split_manifest.csv'))
panel = pd.read_parquet(DATA_DIR / 'daily_panel.parquet')
display(panel[['date', 'rv_yz_22', 'close_variance_forward_22', 'vix']].tail())
'''),
    markdown("""## Benchmarks estatísticos

Cada arquivo parcial é salvo antes de avançar. XGBoost usa 22 regressores
diretos, um por horizonte. O VIX raw é headline apenas em `h=22`."""),
    code(r'''args = ['--time-budget-minutes', SESSION_BUDGET_MINUTES]
if RUN_MODE == 'smoke':
    args += ['--smoke']
run_script('11_statistical_benchmarks.py', *args)

partial = sorted((RESULTS_DIR / 'forecasts/partial').glob('*.parquet'))
print('Partial forecast files:', len(partial))
coverage = []
for path in sorted((RESULTS_DIR / 'forecasts').glob('*.parquet')):
    if path.name == 'all_models.parquet':
        continue
    frame = pd.read_parquet(path)
    coverage.append({'file': path.name, 'windows': frame.window_id.nunique(),
                     'rows': len(frame)})
display(pd.DataFrame(coverage))
if RUN_MODE == 'final' and coverage and not all(row['windows'] == 16 for row in coverage):
    raise RuntimeError('Cobertura incompleta: o teto de 60 minutos foi atingido inesperadamente.')
'''),
    markdown("""## Diagnóstico rápido

Esta avaliação parcial é útil para detectar problemas de escala. A avaliação
oficial completa será executada no notebook 3 depois dos modelos neurais."""),
    code(r'''run_script('15_evaluate_daily.py', '--smoke')
metrics_h22 = RESULTS_DIR / 'metrics/metrics_h22.csv'
if metrics_h22.exists():
    display(pd.read_csv(metrics_h22).sort_values('MSE'))
'''),
    code("ARCHIVE_NAME = 'daily_h22_notebook1_state'\n" + SAFE_DOWNLOAD),
])


nb2 = notebook([
    markdown("""# Daily H=22, notebook 2/3: busca Optuna do TSMixerX

Este notebook executa a busca na primeira janela de oito anos de treino e dois
de validação. O preset final usa 40 trials, até 400 passos, pruning e early
stopping.

**Tempo estimado no Colab com GPU:** 35–55 minutos. O teto operacional é 60
minutos, deixando 10 minutos para exportar o banco, empacotar e baixar os
resultados antes do limite de 70 minutos."""),
    code(SETUP),
    markdown("""## Configuração

O modo final foi dimensionado para concluir em uma sessão de até 70 minutos.
Ele continua sendo uma busca de hiperparâmetros com seleção conjunta do número
de features, mas usa um orçamento computacional compatível com o Colab."""),
    code(r'''RUN_MODE = 'final'  # 'smoke', 'validation' ou 'final'
SESSION_BUDGET_MINUTES = 60
TARGET_TRIALS = {'smoke': 3, 'validation': 15, 'final': 40}[RUN_MODE]
MAX_STEPS = {'smoke': 30, 'validation': 200, 'final': 400}[RUN_MODE]
LOCAL_DB = Path('/content/tsmixerx_daily.db')
BACKUP_DIR = RESULTS_DIR / 'hp_search/sqlite_backup'
print(RUN_MODE, TARGET_TRIALS, MAX_STEPS, BACKUP_DIR)
'''),
    markdown("""## Executar ou retomar a busca

O objetivo é o QLIKE médio dos 22 horizontes na validação. O pruning chama
`trial.report` e `trial.should_prune` em cada checagem. O target é treinado em
log-variância para impedir previsões negativas ou praticamente nulas."""),
    code(r'''args = [
    '--target-trials', TARGET_TRIALS,
    '--max-steps', MAX_STEPS,
    '--timeout-minutes', SESSION_BUDGET_MINUTES,
    '--storage', LOCAL_DB,
    '--backup-dir', BACKUP_DIR,
]
if RUN_MODE == 'smoke':
    args += ['--smoke']
run_script('12_tsmixerx_optuna.py', *args)
'''),
    markdown("""## Estado da busca e melhor configuração"""),
    code(r'''import json, pandas as pd
best_path = RESULTS_DIR / 'hp_search/best_config.json'
if best_path.exists():
    best = json.loads(best_path.read_text())
    print(json.dumps(best, indent=2))
trial_files = sorted((RESULTS_DIR / 'hp_search').glob('trials_*.csv'))
if trial_files:
    trials = pd.read_csv(trial_files[-1])
    display(trials.tail(20))
    print(trials['state'].value_counts(dropna=False))
    if RUN_MODE == 'final' and len(trials) < TARGET_TRIALS:
        raise RuntimeError('Busca incompleta: o teto de 60 minutos foi atingido inesperadamente.')
'''),
    code("ARCHIVE_NAME = 'daily_h22_notebook2_hp_state'\n" + SAFE_DOWNLOAD),
])


nb3 = notebook([
    markdown("""# Daily H=22, notebook 3/3: rolling neural, explicabilidade e avaliação

Este notebook ajusta TSMixerX e LSTM nas 16 janelas 8/2/1 e depois executa
explicabilidade e avaliação final. Cada combinação modelo/janela salva pesos,
forecast, inputs, ranking e histórico.

**Tempo estimado no Colab com GPU:** 20–40 minutos. O teto operacional é 60
minutos, deixando 10 minutos para manifesto, ZIP e download antes do limite de
70 minutos."""),
    code(SETUP),
    markdown("""## Configuração da sessão"""),
    code(r'''RUN_MODE = 'final'  # 'smoke', 'validation' ou 'final'
SESSION_BUDGET_MINUTES = 60
MAX_STEPS = {'smoke': 30, 'validation': 200, 'final': 400}[RUN_MODE]
MAX_WINDOWS = {'smoke': 1, 'validation': 2, 'final': None}[RUN_MODE]
print(RUN_MODE, MAX_STEPS, MAX_WINDOWS)
'''),
    markdown("""## Ajustes rolling

O preset final executa as 16 janelas de ambos os modelos em uma única sessão.
O tempo é verificado entre ajustes e o script preserva margem para avaliação e
empacotamento."""),
    code(r'''args = [
    '--models', 'tsmixerx', 'lstm',
    '--max-steps', MAX_STEPS,
    '--time-budget-minutes', SESSION_BUDGET_MINUTES,
]
if MAX_WINDOWS is not None:
    args += ['--max-windows', MAX_WINDOWS]
run_script('13_neural_rolling.py', *args)
'''),
    markdown("""## Cobertura

A avaliação final exige 16 janelas para cada modelo neural e para os
benchmarks do notebook 1."""),
    code(r'''import pandas as pd
coverage = []
for name in ['tsmixerx', 'lstm']:
    path = RESULTS_DIR / f'forecasts/{name}.parquet'
    if path.exists():
        frame = pd.read_parquet(path)
        coverage.append({'model': name, 'windows': frame.window_id.nunique(), 'rows': len(frame)})
    else:
        coverage.append({'model': name, 'windows': 0, 'rows': 0})
coverage_df = pd.DataFrame(coverage)
display(coverage_df)
NEURAL_COMPLETE = bool((coverage_df.windows == 16).all())
print('Neural complete:', NEURAL_COMPLETE)
if RUN_MODE == 'final' and not NEURAL_COMPLETE:
    raise RuntimeError('Cobertura neural incompleta: o teto de 60 minutos foi atingido inesperadamente.')
'''),
    markdown("""## Explicabilidade temporal

Esta etapa roda automaticamente apenas quando os modelos neurais estão
completos. Integrated Gradients cobre quatro janelas e quatro horizontes.
Shapley Value Sampling gera valores de Shapley agrupados por feature e por lag
na janela final, reduzindo o custo."""),
    code(r'''if NEURAL_COMPLETE:
    run_script('14_explainability.py', '--window-ids', 0, 5, 10, 15,
               '--origins-per-window', 3, '--horizons', 1, 5, 10, 22,
               '--n-steps', 32)
    run_script('14_explainability.py', '--window-ids', 15,
               '--origins-per-window', 1, '--horizons', 1, 22,
               '--n-steps', 16, '--run-shapley', '--shapley-samples', 10)
else:
    print('Explainability postponed until all neural windows are available.')
'''),
    markdown("""## Avaliação final

O script calcula métricas por `h=1,...,22`, DM e GW com HAC de 21 lags, MCS
em `h=22` com blocos de 22 pregões, gráficos VIX/RV e curvas de convergência."""),
    code(r'''if NEURAL_COMPLETE:
    run_script('15_evaluate_daily.py', '--require-complete', '--mcs-reps', 1000)
    display(pd.read_csv(RESULTS_DIR / 'metrics/metrics_h22.csv').sort_values('MSE'))
    display(pd.read_csv(RESULTS_DIR / 'metrics/model_coverage.csv'))
else:
    print('Final evaluation postponed. Partial files remain safely stored in Drive.')
'''),
    markdown("""## Manifesto reproduzível

O manifesto registra versões, tamanhos e SHA-256 de todos os outputs existentes."""),
    code(r'''run_script('16_package_results.py')
manifest = RESULTS_DIR / 'artifact_manifest.json'
if manifest.exists():
    print(manifest.read_text()[:4000])
'''),
    markdown("""## Verificação de recarga dos modelos"""),
    code(r'''artifacts = sorted((RESULTS_DIR / 'models/tsmixerx').glob('*.pt'))
print('TSMixerX artifacts:', len(artifacts))
if artifacts:
    sys.path.insert(0, str(EXP / 'src'))
    from models import DirectVolatilityRegressor
    loaded, payload = DirectVolatilityRegressor.load(artifacts[-1])
    print('Reloaded:', artifacts[-1].name, '| input_size:', loaded.input_size,
          '| features:', len(payload['features']))
'''),
    code("ARCHIVE_NAME = 'daily_h22_notebook3_results'\n" + SAFE_DOWNLOAD),
])


for name, payload in [
    ("01_data_and_statistical_benchmarks.ipynb", nb1),
    ("02_tsmixerx_optuna.ipynb", nb2),
    ("03_neural_rolling_evaluation.ipynb", nb3),
]:
    (HERE / name).write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print("Wrote", HERE / name)
