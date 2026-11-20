# Revisões da qualificação e estratégia Cash-Secured Put Writing

Para os dados complementares e a marcação diária até 2025, veja o
[guia da revisão](followup/README.md). Este documento registra a primeira entrega.

Este diretório operacionaliza o pedido de [Modificações Dissertação.md](../../Modifica%C3%A7%C3%B5es%20Disserta%C3%A7%C3%A3o.md). Contém código, dados intermediários com proveniência, resultados efetivamente executados e um notebook Colab. Os resultados anteriores continuam em `experiments/daily_h22/results/`.

## Começar pelos resultados

Leia [RESULTADOS_E_PENDENCIAS.md](RESULTADOS_E_PENDENCIAS.md) para a interpretação e as limitações. A [auditoria dos dados](results/data_audit/AUDITORIA.md) e a [revisão independente](results/INDEPENDENT_REVIEW.md) registram as verificações realizadas. A mensagem para o grupo está em [mensagem_grupo_anna_julia.txt](../../mensagem_grupo_anna_julia.txt), com um parágrafo em cada linha.

| Pedido | Implementação e saída |
|---|---|
| HAR com VIX | `../daily_h22/src/11_statistical_benchmarks.py`; coeficientes de 16 janelas em `results/forecast_revision/models/har_vix/` |
| Teste para modelos aninhados | `../daily_h22/src/15_evaluate_daily.py`; [Clark-West por horizonte](results/forecast_revision/metrics/clark_west_har_vix.csv) |
| Bootstrap da série do índice | `src/bootstrap.py`; [2.000 trajetórias resumidas](results/bootstrap_index_draws.csv), [cinco trajetórias completas](results/bootstrap_index_example_paths.csv), [configuração](results/bootstrap_index_config.json) |
| Sinal VRP e alocação | `src/strategy.py`; `results/daily_signals.parquet`, com IV, previsão, VRP, quantil e peso |
| Simulação com prêmios observados | `src/run_revision.py`; [contratos usados](results/contracts_used.csv), [operações e contabilidade](results/expiry_ledgers.csv) |
| Sharpe, Sortino e drawdown | [Métricas nos vencimentos](results/performance_expiry.csv), na mesma frequência para estratégias e benchmarks |
| Bootstrap financeiro pareado | [Intervalos por cenário e tamanho de bloco](results/bootstrap_paired_monthly.csv), condicionais às operações observadas |

## Reproduzir

O [notebook Colab](https://colab.research.google.com/github/hugogobato/Mestrado_Anna_Julia/blob/main/experiments/putwriting/notebooks/03_qualification_and_putwriting.ipynb) clona o repositório, instala as dependências, executa a revisão e baixa um único ZIP com os resultados. Não precisa de GPU nem de montar Google Drive. Ele usa os arquivos que acompanham este repositório e apresenta as hipóteses antes da execução.

Localmente, a partir da raiz:

```bash
python3 -m venv /tmp/anna-revision-venv
/tmp/anna-revision-venv/bin/python -m pip install -r experiments/putwriting/requirements.txt
/tmp/anna-revision-venv/bin/python experiments/putwriting/src/run_revision.py --assume-am-roll --bootstrap-reps 2000
```

O parâmetro `--assume-am-roll` identifica expressamente o cenário em que os contratos são AM e a nova venda ocorre após a liquidação do contrato anterior na mesma data. Os CSVs não contêm horários que comprovem essa ordem. Sem esse parâmetro, o programa explica a hipótese necessária e interrompe a execução financeira. O sinal sempre usa o fechamento do pregão anterior à entrada.

Os CSVs de previsão publicados em `results/forecast_revision/forecasts/` e `results/neural_purged/forecasts/` foram compactados sem perda em `.csv.gz`, com conferência de conteúdo registrada em `results/csv_compression_manifest.json`. Os Parquets permanecem disponíveis para leitura direta. Uma reexecução gera os CSVs novamente; `src/compress_forecast_csv.py` reproduz a compactação para publicação.

O comando refaz HAR e HAR-VIX nas 16 janelas, executa Clark-West e os 36 cenários financeiros (nove modelos, LIVE/VIX e bid/mid). A entrada `data/strategy_forecasts_h22.parquet` combina seis previsões estatísticas preservadas com LSTM e TSMixerX reestimados após a correção das fronteiras dos alvos. O arquivo `data/archived_forecasts_h22.parquet` mantém as oito previsões anteriores para rastreabilidade. O painel diário preservado está em `data/daily_panel.parquet`; [PROVENIENCIA.json](data/PROVENIENCIA.json) e [strategy_forecasts_provenance.json](data/strategy_forecasts_provenance.json) registram as origens. O Excel dentro do ZIP fornecido é idêntico ao Excel original do repositório.

Para apenas repetir a estratégia e os bootstraps após o ajuste de HAR-VIX, acrescente `--skip-forecast-fit`. Para preservar a execução principal e gerar outra, use `--output /caminho/novo`. A semente padrão é 42. O [manifesto](results/run_manifest.json) registra hashes das entradas, do novo HAR-VIX e do código, versões de bibliotecas, datas, hipóteses e escopo.

A etapa de HAR/HAR-VIX, estratégia e bootstraps levou aproximadamente 29 segundos e usou cerca de 400 MB de RAM nesta máquina, com duas threads numéricas. Esse tempo não inclui instalação, clonagem nem o retreinamento neural. O notebook usa as previsões corrigidas salvas por padrão. A opção `REFIT_NEURAL=True` refaz uma busca de 25 trials e até 400 passos e os 32 ajustes anuais em um diretório inicialmente vazio; ela pode demorar substancialmente mais. O histórico e os logs da execução realizada estão em `results/neural_purged/`.

## Especificação e interpretação

O HAR original permanece uma regressão direta de `log(RV_YZ[t+h])` sobre constante, log-variância diária e médias das log-variâncias de cinco e 22 pregões. O HAR-VIX acrescenta `(VIX[t]/100)^2`. Ambos usam as mesmas origens e a mesma amostra de treino. As previsões são exponenciadas, sem correção de retransformation; portanto são previsões pontuais de variância e não uma demonstração de que estimam a média aritmética condicional exigida na interpretação estrita de VRP. Não alteramos a especificação depois de observar seus maus resultados em 2020.

Clark-West compara exclusivamente o HAR restrito e o HAR-VIX irrestrito. A diferença ajustada é `e_HAR² - e_HAR-VIX² + (f_HAR - f_HAR-VIX)²`. O teste é unilateral, com valores positivos favoráveis ao modelo maior, covariância HAC com pelo menos 21 lags e correção BH entre os 22 horizontes. O resultado principal está nas colunas `CW_log_*`, porque os modelos são lineares e aninhados na escala log. As colunas sem `log` são uma aproximação exploratória na escala de variância, expressamente identificada no CSV. Não foi inventado um teste LR para ordens GARCH que não foram ajustadas.

O sinal é `VRP[t] = (IV_percent[t]/100)^2 - forecast_variance_h22[t]`. A expressão do documento foi interpretada consistentemente em variância: IV precisa ser elevada ao quadrado. O quantil é a fração dos 504 VRPs anteriores menores ou iguais ao atual, sem incluir o próprio dia. Não há negociação com histórico incompleto. Se VRP for não positivo, `w=0`; caso contrário, `w=q`, com teto um. IV ausente não vira zero. LIVE é uma superfície genérica ATM de um mês, cuja equivalência à IV de cada contrato individual ainda precisa ser confirmada.

Os sinais são calculados diariamente, mas as opções são negociadas apenas nas datas mensais em que há contratos e prêmios observados. Os pesos permanecem fixos até o vencimento. Não se abrem 22 posições sobrepostas. Os strikes fornecidos são usados como aproximações ATM em uma grade discreta, sem inventar um strike exatamente igual ao índice. O prazo real de cada contrato pode diferir dos 22 pregões previstos a partir do fechamento anterior.

Cada carteira começa com patrimônio um. A quantidade fracionária normalizada é `w * patrimônio / strike`, equivalente a dividir também pelo multiplicador ao trabalhar com contratos em dólares. O valor máximo de exercício fica integralmente coberto. O prêmio entra no caixa, a obrigação da opção existe até o vencimento, e o pagamento final é `quantidade * max(strike - SOQ, 0)`. O total em caixa, incluindo o prêmio, rende juros. Não há curva diária de patrimônio fabricada entre os vencimentos.

A taxa de caixa é tratada, como hipótese, como rendimento anual simples em porcentagem, ACT/360 e reinvestimento diário. Cada dia usa exclusivamente uma cotação anterior, com idade máxima de sete dias. Isso é uma aproximação de conta remunerada, não uma reconstrução do retorno total de títulos do Tesouro. A identificação do campo Bloomberg e da convenção de desconto/yield ainda precisa ser confirmada.

Há dois benchmarks distintos: `put` é a venda integralmente colateralizada dos mesmos contratos e prêmios da base; `official_PUT_close_proxy` usa os fechamentos do PUT oficial nas mesmas datas. O primeiro não é o índice oficial. A Cboe usa regras próprias de seleção, preços intradiários e caixa em T-bills de um e três meses; por isso não esperamos replicação exata. Fechamentos do PUT e liquidação AM também não são eventos simultâneos.

## O que os bootstraps verificam

O bootstrap diário sorteia blocos contíguos de retornos logarítmicos do SPX com reposição e reconstrói preços positivos. A configuração padrão usa aproximadamente 20 blocos por caminho. Com 6.288 retornos válidos, são blocos de 315 retornos e truncamento do último bloco. Esse exercício descreve trajetórias alternativas do índice, sem recalcular opções, sinais ou previsões.

O bootstrap financeiro sorteia os mesmos blocos para os retornos mensais da estratégia e do benchmark. Usa blocos de um, três e seis períodos e 2.000 réplicas por cenário. Os blocos de seis meses fornecem apenas 18 blocos não sobrepostos, explicitamente sinalizados na saída. Os intervalos são percentis da diferença média de retorno por período. `bootstrap_fraction_mean_diff_le_zero` é uma fração de simulações, não um p-valor de teste de hipótese. Esse exercício condiciona-se às previsões, aos sinais e às operações originais. Nenhum dos dois bootstraps comprova a robustez de toda a estratégia em novos universos de preços de opções.

## Verificação e fontes

```bash
DAILY_H22_DATA_DIR="$PWD/experiments/putwriting/data" /tmp/anna-revision-venv/bin/python -m pytest experiments/daily_h22/tests experiments/putwriting/tests -q
```

O notebook pode ser regenerado por `python3 experiments/putwriting/notebooks/build_notebook.py`. Ele reúne todos os outputs em um ZIP e inclui o fallback seguro de download exigido para Colab.

Clark e West (2007), [Approximately Normal Tests for Equal Predictive Accuracy in Nested Models](https://www.kansascityfed.org/documents/5368/pdf-RWP05-05.pdf), fundamentam o teste aninhado. Künsch (1989), [The Jackknife and the Bootstrap for General Stationary Observations](https://doi.org/10.1214/aos/1176347265), fundamenta a reamostragem em blocos sob condições de dependência e estacionariedade. A [metodologia oficial Cboe S&P 500 PutWrite](https://cdn.cboe.com/resources/indices/governance/Cboe_SP_500_PutWrite_Indices_Methodology.pdf), consultada em 17/11/2026, esclarece SOQ, rolagem, marcação das opções e formação do benchmark. Esses documentos não verificam a identificação dos contratos ou os horários dos CSVs fornecidos.
