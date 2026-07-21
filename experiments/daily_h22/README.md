# Experimento v2: previsões diárias H=22

Este diretório é independente do pipeline v1 na raiz do repositório. Ele mantém
uma observação por pregão, prevê os 22 próximos valores da variância
Yang–Zhang móvel de 22 pregões e avalia os modelos em janelas rolling 8/2/1.

## Notebooks Colab

1. `notebooks/01_data_and_statistical_benchmarks.ipynb`: prepara o painel,
   valida o alvo e executa RW, HAR, GARCH, VIX, VIX calibrado e XGBoost. Tempo
   estimado no Colab: 10–20 minutos. Ao final, baixa os resultados completos e
   `daily_h22_transfer_to_neural.zip`.
2. `notebooks/02_tsmixerx_full_pipeline.ipynb`: recebe o ZIP de transferência,
   executa Optuna, ajusta TSMixerX e LSTM por janela, gera explicabilidade e faz
   a avaliação final. O preset usa 25 trials e até 400 passos. Tempo estimado no
   Colab com GPU: 50–65 minutos.

Depois que esta pasta estiver no branch `main`, os notebooks poderão ser
abertos diretamente no Colab:

1. [Notebook 1: dados e benchmarks](https://colab.research.google.com/github/hugogobato/Mestrado_Anna_Julia/blob/main/experiments/daily_h22/notebooks/01_data_and_statistical_benchmarks.ipynb)
2. [Notebook 2: Optuna, rolling neural e avaliação](https://colab.research.google.com/github/hugogobato/Mestrado_Anna_Julia/blob/main/experiments/daily_h22/notebooks/02_tsmixerx_full_pipeline.ipynb)

Os notebooks não montam nem criam pastas no Google Drive. Todos os arquivos
ficam em `/content/daily_h22_run` durante a sessão e são baixados como ZIP pelo
navegador. Isso permite executar os notebooks em contas diferentes: execute o
notebook 1, guarde `daily_h22_transfer_to_neural.zip` e envie esse arquivo quando
o notebook 2 solicitar.

Cada notebook foi dimensionado para concluir dentro do limite de 70 minutos e
oferece fallback seguro de download para o Colab.

As estimativas são conservadoras e foram calibradas com benchmarks locais em
CPU: os 16 anos dos benchmarks levaram cerca de 4 minutos, três trials de 400
passos levaram 1 minuto e 47 segundos, e um par TSMixerX+LSTM de uma janela
levou cerca de 19 segundos. A faixa para Colab inclui instalação, upload do ZIP,
I/O, avaliação e variação de hardware.

## Metodologia resumida

O target diário é `rv_yz_22`, a variância Yang–Zhang anualizada calculada em
uma janela móvel de 22 pregões. Uma origem `t` gera previsões para
`rv_yz_22[t+h]`, com `h=1,...,22`. O resultado principal é `h=22`.

O backend neural desta versão é uma implementação PyTorch nativa e direta do
TSMixerX, com blocos de mistura temporal e de features exógenas e uma saída de
22 dimensões. Essa opção preserva o índice exato de pregões, permite atualizar
o contexto em cada origem sem reajustar os pesos e torna a retomada por janela
independente da API de calendário do NeuralForecast. O nome da implementação e
seus hiperparâmetros ficam registrados nos artefatos; ela não é apresentada
como a classe oficial `neuralforecast.models.TSMixerx`.

Cada janela usa oito anos de treino, dois de validação e um de teste. Os
hiperparâmetros são escolhidos na primeira janela 8/2 e congelados. Os pesos
são reajustados anualmente. VIX3M não entra na análise principal.

## Execução por scripts

```bash
python experiments/daily_h22/src/10_data_prep_daily.py --smoke
python experiments/daily_h22/src/11_statistical_benchmarks.py --smoke
python experiments/daily_h22/src/12_tsmixerx_optuna.py --target-trials 3 --max-steps 30
python experiments/daily_h22/src/13_neural_rolling.py --models tsmixerx lstm --max-windows 1 --max-steps 30
python experiments/daily_h22/src/14_explainability.py --smoke
python experiments/daily_h22/src/15_evaluate_daily.py --smoke
python experiments/daily_h22/src/16_package_results.py
```

Os resultados ficam exclusivamente em `experiments/daily_h22/results/`.
