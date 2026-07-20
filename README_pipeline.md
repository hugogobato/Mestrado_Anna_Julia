# Pipeline de previsão de volatilidade

O projeto transforma o painel diário do S&P 500 em observações operacionais de 22 pregões. Cada bloco usa somente a informação disponível até sua última data, e o alvo `rv_yz_forward` é a média da RV_YZ dos 22 pregões estritamente seguintes. O painel resultante tem 296 blocos válidos entre 2000 e 2025, porque os 22 pregões são não sobrepostos e o último bloco não possui horizonte futuro completo.

## Execução

No ambiente local, use o Python disponível e um `PYTHONPATH` que inclua o `pyarrow` do `.venv` quando necessário. No Colab, execute primeiro a célula de instalação do notebook.

```bash
python src/00_data_prep.py --smoke
python src/01_feature_selection.py
python src/02_benchmarks.py --smoke
python src/03_tsmixerx.py --trials 20 --max-steps 200
python src/04_evaluate.py
```

O experimento final troca o smoke pelo orçamento escolhido, por exemplo `--trials 100 --max-steps 1000`. A busca salva os trials em `results/hp_search/tsmixerx.db` e a configuração vencedora em `results/hp_search/best_config.json`, permitindo retomada.

## Organização temporal

O pré-filtro de colinearidade e o ranking de features usam apenas o período inicial. A busca TSMixerX usa os primeiros oito anos para ajuste e os dois seguintes para validação. Depois disso, `rolling_indices` cria janelas anuais com quatro anos de treino, um de validação e um de teste, avançando um ano por vez. Isso produz quinze janelas de teste quando a base vai até 2025.

## Modelos

`02_benchmarks.py` gera RW, HAR, GARCH(1,1), XGBoost, LSTM, VIX e VIX3M. `03_tsmixerx.py` implementa a mistura temporal e de features em PyTorch, aceita CUDA e integra seleção `SelectKBest` à busca Optuna. A implementação nativa é usada para manter o índice de 22 pregões exato, sem inventar dias ausentes para satisfazer uma frequência de calendário do NeuralForecast.

`04_evaluate.py` calcula QLIKE, MSE, MAE, R² fora da amostra, DM, GW, Clark-West e uma eliminação sequencial MCS exploratória. O MCS final da dissertação deve usar bootstrap estacionário e ser documentado como análise de sensibilidade, caso essa versão exploratória seja mantida.

## Colab

O notebook `notebooks/volatility_pipeline_colab.ipynb` instala as dependências, executa o smoke test, oferece um modo curto e um modo full, e compacta os resultados para download automático. A implementação PyTorch usa CUDA quando `torch.cuda.is_available()` for verdadeiro.

## Artefatos persistidos

As previsões são gravadas em `results/forecasts_benchmarks.parquet` e `results/forecasts_tsmixerx.parquet`, com cópias CSV para uso fora do Python. Cada janela TSMixerX gera um arquivo `results/models/tsmixerx/tsmixerx_window_XX.pt`, acompanhado de pesos, escalas, configuração, features selecionadas e datas da janela. Os benchmarks treináveis geram pesos LSTM em `.pt`, modelos XGBoost em `.joblib` e parâmetros HAR/GARCH em `.json`; os manifestos indicam a correspondência entre janela e arquivo. O notebook compacta todo o diretório `results/` em `volatility_pipeline_artifacts.zip` e tenta baixá-lo automaticamente no Colab.
