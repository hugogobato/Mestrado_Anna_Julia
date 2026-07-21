"""Gera os dois notebooks Colab do experimento diário H=22."""

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


SETUP = r'''# Configuração compartilhada: repositório + armazenamento local da sessão
from pathlib import Path
import os, shutil, subprocess, sys

REPO_URL = 'https://github.com/hugogobato/Mestrado_Anna_Julia.git'
REPO = Path('/content/Mestrado_Anna_Julia')
if not REPO.exists():
    subprocess.run(['git', 'clone', REPO_URL, str(REPO)], check=True)
else:
    subprocess.run(['git', '-C', str(REPO), 'pull', '--ff-only'], check=True)

EXP = REPO / 'experiments/daily_h22'
PERSIST = Path('/content/daily_h22_run')
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
print('Session data:', DATA_DIR)
print('Session results:', RESULTS_DIR)
try:
    import torch
    print('Torch:', torch.__version__, '| CUDA:', torch.cuda.is_available(),
          '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
except Exception as exc:
    print('Torch check failed:', exc)
'''


UPLOAD_TRANSFER = r'''# Importar o ZIP gerado pelo notebook 1
import shutil
from pathlib import Path

transfer_name = 'daily_h22_transfer_to_neural.zip'
transfer_path = Path('/content') / transfer_name
if not transfer_path.exists():
    try:
        from google.colab import files
        uploaded = files.upload()
        if transfer_name not in uploaded:
            raise FileNotFoundError(
                f'Selecione exatamente {transfer_name}; recebido: {list(uploaded)}')
        transfer_path.write_bytes(uploaded[transfer_name])
    except ImportError as exc:
        raise FileNotFoundError(
            f'Fora do Colab, copie {transfer_name} para /content antes de continuar.') from exc

shutil.unpack_archive(str(transfer_path), str(PERSIST))
required = [
    DATA_DIR / 'daily_panel.parquet',
    DATA_DIR / 'split_manifest.csv',
    RESULTS_DIR / 'forecasts/rw.parquet',
    RESULTS_DIR / 'forecasts/har.parquet',
    RESULTS_DIR / 'forecasts/garch_11.parquet',
    RESULTS_DIR / 'forecasts/vix.parquet',
    RESULTS_DIR / 'forecasts/vix_calibrated.parquet',
    RESULTS_DIR / 'forecasts/xgboost.parquet',
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError('ZIP de transferência incompleto:\n' + '\n'.join(missing))
print('Transfer imported successfully:', transfer_path)
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
os resultados da execução v1. Ao final, ele baixa os resultados completos e um
ZIP menor para transferir os dados e forecasts ao notebook 2, mesmo quando os
notebooks são executados em contas diferentes.

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
oficial completa será executada no notebook 2 depois dos modelos neurais."""),
    code(r'''run_script('15_evaluate_daily.py', '--smoke')
metrics_h22 = RESULTS_DIR / 'metrics/metrics_h22.csv'
if metrics_h22.exists():
    display(pd.read_csv(metrics_h22).sort_values('MSE'))
'''),
    markdown("""## ZIP de transferência para o notebook 2

Guarde este arquivo. Na outra conta do Colab, o notebook 2 solicitará seu
upload. Ele contém apenas o painel e os forecasts necessários para continuar,
sem os modelos estatísticos mais pesados."""),
    code(r'''transfer_root = Path('/content/daily_h22_transfer')
if transfer_root.exists():
    shutil.rmtree(transfer_root)
(transfer_root / 'data').mkdir(parents=True)
(transfer_root / 'results').mkdir(parents=True)
shutil.copytree(DATA_DIR, transfer_root / 'data', dirs_exist_ok=True)
shutil.copytree(RESULTS_DIR / 'forecasts',
                transfer_root / 'results/forecasts', dirs_exist_ok=True)
transfer_file = shutil.make_archive('/content/daily_h22_transfer_to_neural',
                                    'zip', root_dir=transfer_root)
print('Transfer archive:', transfer_file)
try:
    from google.colab import files
    files.download(transfer_file)
    print('Downloaded:', transfer_file)
except Exception as e:
    print('(Not on Colab / download skipped):', e)
'''),
    markdown("""## Resultados completos do notebook 1

Este segundo ZIP inclui também os modelos estatísticos e diagnósticos. Guarde-o
como artefato independente; ele não precisa ser enviado ao notebook 2."""),
    code("ARCHIVE_NAME = 'daily_h22_notebook1_results'\n" + SAFE_DOWNLOAD),
])


nb2 = notebook([
    markdown("""# Daily H=22, notebook 2/2: Optuna, rolling neural e avaliação

Este notebook recebe o ZIP de transferência do notebook 1, executa a busca de
hiperparâmetros e usa imediatamente a configuração vencedora para ajustar
TSMixerX e LSTM nas 16 janelas 8/2/1. Depois gera explicabilidade, métricas,
testes e o pacote final.

O preset final usa 25 trials, até 400 passos, pruning e early stopping.
**Tempo estimado no Colab com GPU:** 50–65 minutos. Nenhum arquivo é gravado no
Google Drive; o ZIP final é baixado pelo navegador."""),
    code(SETUP),
    markdown("""## Importar os resultados do notebook 1

Selecione `daily_h22_transfer_to_neural.zip`, baixado pelo notebook 1. Isso
permite usar outra conta do Colab sem depender de uma pasta compartilhada."""),
    code(UPLOAD_TRANSFER),
    markdown("""## Configuração computacional

O orçamento foi dividido entre Optuna e os ajustes rolling para manter a
execução total dentro de 70 minutos. O modo final usa 25 trials, e não os 40 do
desenho anterior, porque busca e treinamento agora rodam na mesma sessão."""),
    code(r'''RUN_MODE = 'final'  # 'smoke', 'validation' ou 'final'
TARGET_TRIALS = {'smoke': 3, 'validation': 10, 'final': 25}[RUN_MODE]
MAX_STEPS = {'smoke': 30, 'validation': 200, 'final': 400}[RUN_MODE]
OPTUNA_BUDGET_MINUTES = {'smoke': 5, 'validation': 15, 'final': 35}[RUN_MODE]
NEURAL_BUDGET_MINUTES = {'smoke': 5, 'validation': 10, 'final': 20}[RUN_MODE]
MAX_WINDOWS = {'smoke': 1, 'validation': 2, 'final': None}[RUN_MODE]
LOCAL_DB = RESULTS_DIR / 'hp_search/tsmixerx_daily.db'
print(RUN_MODE, TARGET_TRIALS, MAX_STEPS,
      OPTUNA_BUDGET_MINUTES, NEURAL_BUDGET_MINUTES)
'''),
    markdown("""## Busca Optuna

O objetivo é o QLIKE médio dos 22 horizontes na validação. A busca seleciona
arquitetura, input size, loss e número de features usando apenas a primeira
janela 8/2. O pruning chama `trial.report` e `trial.should_prune`."""),
    code(r'''optuna_args = [
    '--target-trials', TARGET_TRIALS,
    '--max-steps', MAX_STEPS,
    '--timeout-minutes', OPTUNA_BUDGET_MINUTES,
    '--storage', LOCAL_DB,
]
if RUN_MODE == 'smoke':
    optuna_args += ['--smoke']
run_script('12_tsmixerx_optuna.py', *optuna_args)
'''),
    markdown("""## Melhor configuração"""),
    code(r'''import json, pandas as pd
best_path = RESULTS_DIR / 'hp_search/best_config.json'
if not best_path.exists():
    raise FileNotFoundError('A busca não produziu best_config.json')
best = json.loads(best_path.read_text())
print(json.dumps(best, indent=2))
trial_files = sorted((RESULTS_DIR / 'hp_search').glob('trials_*.csv'))
trials = pd.read_csv(trial_files[-1])
display(trials.tail(25))
print(trials['state'].value_counts(dropna=False))
if RUN_MODE == 'final' and len(trials) < TARGET_TRIALS:
    raise RuntimeError('Busca incompleta dentro do orçamento estimado.')
'''),
    markdown("""## Ajustes rolling TSMixerX e LSTM

A configuração vencedora é congelada e os pesos são reajustados em cada uma das
16 janelas. Os modelos, forecasts, inputs, rankings e históricos são salvos no
armazenamento local da sessão."""),
    code(r'''neural_args = [
    '--models', 'tsmixerx', 'lstm',
    '--max-steps', MAX_STEPS,
    '--time-budget-minutes', NEURAL_BUDGET_MINUTES,
]
if MAX_WINDOWS is not None:
    neural_args += ['--max-windows', MAX_WINDOWS]
run_script('13_neural_rolling.py', *neural_args)
'''),
    markdown("""## Verificação de cobertura"""),
    code(r'''coverage = []
for name in ['tsmixerx', 'lstm']:
    path = RESULTS_DIR / f'forecasts/{name}.parquet'
    if path.exists():
        frame = pd.read_parquet(path)
        coverage.append({'model': name, 'windows': frame.window_id.nunique(),
                         'rows': len(frame)})
    else:
        coverage.append({'model': name, 'windows': 0, 'rows': 0})
coverage_df = pd.DataFrame(coverage)
display(coverage_df)
NEURAL_COMPLETE = bool((coverage_df.windows == 16).all())
if RUN_MODE == 'final' and not NEURAL_COMPLETE:
    raise RuntimeError('Cobertura neural incompleta dentro do orçamento estimado.')
'''),
    markdown("""## Explicabilidade temporal

Integrated Gradients cobre quatro janelas e quatro horizontes. Shapley Value
Sampling é aplicado a uma amostra menor na janela final."""),
    code(r'''if NEURAL_COMPLETE:
    run_script('14_explainability.py', '--window-ids', 0, 5, 10, 15,
               '--origins-per-window', 3, '--horizons', 1, 5, 10, 22,
               '--n-steps', 32)
    run_script('14_explainability.py', '--window-ids', 15,
               '--origins-per-window', 1, '--horizons', 1, 22,
               '--n-steps', 16, '--run-shapley', '--shapley-samples', 10)
'''),
    markdown("""## Avaliação final

São calculados QLIKE, MSE, MAE e R² OOS por horizonte, DM, GW com HAC de 21
lags e MCS via `arch.bootstrap.MCS` em `h=22` com block size 22."""),
    code(r'''if NEURAL_COMPLETE:
    run_script('15_evaluate_daily.py', '--require-complete', '--mcs-reps', 1000)
    display(pd.read_csv(RESULTS_DIR / 'metrics/metrics_h22.csv').sort_values('MSE'))
    display(pd.read_csv(RESULTS_DIR / 'metrics/model_coverage.csv'))
'''),
    markdown("""## Manifesto e verificação de recarga"""),
    code(r'''run_script('16_package_results.py')
manifest = RESULTS_DIR / 'artifact_manifest.json'
if manifest.exists():
    print(manifest.read_text()[:4000])

artifacts = sorted((RESULTS_DIR / 'models/tsmixerx').glob('*.pt'))
print('TSMixerX artifacts:', len(artifacts))
if artifacts:
    sys.path.insert(0, str(EXP / 'src'))
    from models import DirectVolatilityRegressor
    loaded, payload = DirectVolatilityRegressor.load(artifacts[-1])
    print('Reloaded:', artifacts[-1].name, '| input_size:', loaded.input_size,
          '| features:', len(payload['features']))
'''),
    markdown("""## Download dos resultados finais

O ZIP contém o painel, forecasts de todos os modelos, modelos neurais,
históricos, explicabilidade, métricas, figuras e manifesto. Os modelos
estatísticos completos permanecem no ZIP separado baixado pelo notebook 1."""),
    code("ARCHIVE_NAME = 'daily_h22_final_results'\n" + SAFE_DOWNLOAD),
])


for name, payload in [
    ("01_data_and_statistical_benchmarks.ipynb", nb1),
    ("02_tsmixerx_full_pipeline.ipynb", nb2),
]:
    (HERE / name).write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print("Wrote", HERE / name)
