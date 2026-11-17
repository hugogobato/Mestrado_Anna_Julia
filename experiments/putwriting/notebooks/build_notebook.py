"""Generate the portable Colab notebook; downloads one ZIP with all outputs."""
import json
from pathlib import Path


def cell(kind, source):
    import hashlib
    item = {'cell_type': kind, 'id': hashlib.sha256(source.encode()).hexdigest()[:12],
            'metadata': {}, 'source': source.splitlines(keepends=True)}
    if kind == 'code':
        item.update(execution_count=None, outputs=[])
    return item


cells = [
    cell('markdown', '''# Revisões da qualificação e estratégia de puts

Reproduz HAR/HAR-VIX, Clark-West, auditoria, sinais VRP, cenários bid/mid,
métricas nos vencimentos e dois bootstraps. Reutiliza as previsões neurais
da reexecução com fronteiras temporais corrigidas. Não precisa de GPU,
upload ou Google Drive no modo padrão. O README em
`experiments/putwriting/` explica as hipóteses e pendências da base.

A execução assume contratos SPX AM, liquidação SOQ antes da nova venda na
mesma data e juros simples ACT/360 reinvestidos diariamente. A base não
comprova esses horários nem a convenção da taxa. Os resultados são provisórios.
'''),
    cell('code', '''from pathlib import Path
import subprocess
import sys

run_root = Path('/content') if Path('/content').exists() else Path.cwd()
repo = run_root / 'anna_julia_revision'
if not (repo / '.git').exists():
    subprocess.run(['git', 'clone', '--depth', '1',
                    'https://github.com/hugogobato/Mestrado_Anna_Julia.git', str(repo)], check=True)
subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r',
                str(repo / 'experiments/putwriting/requirements.txt')], check=True)
'''),
    cell('code', '''# Opcional: refazer a busca e os 32 ajustes neurais, em uma árvore separada.
# O modo padrão usa a execução corrigida que já acompanha o repositório.
REFIT_NEURAL = False
if REFIT_NEURAL:
    import os
    import shutil
    neural_out = run_root / 'anna_julia_neural_refit'
    if neural_out.exists():
        raise FileExistsError('Use um diretório vazio para uma nova busca neural.')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torch', 'optuna'], check=True)
    env = os.environ.copy()
    env.update(DAILY_H22_DATA_DIR=str(repo / 'experiments/putwriting/data'),
               DAILY_H22_RESULTS_DIR=str(neural_out), OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    scripts = repo / 'experiments/daily_h22/src'
    subprocess.run([sys.executable, str(scripts / '12_tsmixerx_optuna.py'),
                    '--target-trials', '25', '--max-steps', '400',
                    '--timeout-minutes', '120'], env=env, check=True)
    subprocess.run([sys.executable, str(scripts / '13_neural_rolling.py'),
                    '--models', 'tsmixerx', 'lstm', '--max-steps', '400',
                    '--time-budget-minutes', '120'], env=env, check=True)
    base = repo / 'experiments/putwriting'
    shutil.copytree(neural_out, base / 'results/neural_purged', dirs_exist_ok=True)
    subprocess.run([sys.executable, str(base / 'src/build_strategy_forecasts.py')], check=True)
'''),
    cell('code', '''BOOTSTRAP_REPS = 2000
SEED = 42
# Hipótese explícita: a nova venda ocorre depois da liquidação AM na rolagem.
ASSUME_AM_ROLL = True
if not ASSUME_AM_ROLL:
    raise ValueError('Confirme os horários dos contratos antes de executar outro cenário.')
subprocess.run([sys.executable, str(repo / 'experiments/putwriting/src/run_revision.py'),
                '--assume-am-roll', '--bootstrap-reps', str(BOOTSTRAP_REPS),
                '--seed', str(SEED)], cwd=repo, check=True)
'''),
    cell('code', '''import pandas as pd
result_dir = repo / 'experiments/putwriting/results'
metrics = pd.read_csv(result_dir / 'performance_expiry.csv')
display(metrics.query("iv_source == 'iv' and premium_source == 'bid'"))
display(pd.read_csv(result_dir / 'forecast_revision/metrics/clark_west_har_vix.csv').tail(1))
'''),
    cell('code', '''import shutil
output_file = shutil.make_archive(str(run_root / 'anna_julia_revisoes_resultados'),
                                  'zip', root_dir=result_dir)
print('Resultados:', output_file)
try:
    from google.colab import files
    files.download(output_file)
    print("Downloaded:", output_file)
except Exception as e:
    print("(Not on Colab / download skipped):", e)
'''),
]

notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                      'language_info': {'name': 'python', 'version': '3.12'}},
            'nbformat': 4, 'nbformat_minor': 5}
path = Path(__file__).with_name('03_qualification_and_putwriting.ipynb')
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf8')
