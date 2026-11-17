# Análise dos resultados do pipeline de previsão de volatilidade

## 1. Escopo da execução

Esta análise foi elaborada a partir dos arquivos presentes em `results/` após a execução final do notebook. O manifesto registra `run_mode=final`, 200 trials de Optuna, até 1.000 passos de treinamento por ajuste, execução das análises de robustez e MCS oficial com `arch.bootstrap.MCS`.

O período de avaliação fora da amostra contém 171 observações operacionais de 22 pregões, de 7 de janeiro de 2011 a 20 de novembro de 2025. A seleção de hiperparâmetros foi feita em um split cronológico fixo, com oito anos de treino e dois anos de validação. Conforme a decisão metodológica registrada, a busca de hiperparâmetros não utiliza um rolling 4+1+1 interno. Depois da seleção, a avaliação usa as janelas anuais rolling 4+1+1.

O melhor trial registrado escolheu `input_size=1`, três blocos, `ff_dim=32`, dropout de 0,1, learning rate de 0,003, scaler robusto, perda MAE, batch size 16, RevIN ativado e 50 features. Os 50 atributos selecionados incluem variáveis de mercado, estimadores de volatilidade, indicadores técnicos, macroeconomia, stress financeiro e lags do alvo.

## 2. Comparação principal

| Modelo | QLIKE | MSE | MAE | R² fora da amostra | MCS a 5% |
|---|---:|---:|---:|---:|---|
| GARCH(1,1) | 0,5768 | 0,001007 | 0,017686 | 0,6025 | Incluído |
| VIX | **0,2338** | 0,001557 | 0,018985 | 0,3857 | Excluído, p=0,042 |
| TSMixerX | 229.454,62 | 0,002361 | **0,017915** | 0,0683 | Excluído, p=0,008 |
| XGBoost | 0,2836 | 0,002024 | 0,019761 | 0,2012 | Excluído, p=0,008 |
| VIX3M | 0,3402 | 0,001964 | 0,025123 | 0,2250 | Excluído, p<0,001 |
| HAR | 5.642.003,16 | 0,001988 | 0,021037 | 0,2155 | Excluído, p=0,002 |
| RW | 0,2866 | 0,002534 | 0,019892 | 0,0000 | Excluído, p=0,004 |
| LSTM | 0,8534 | 0,009312 | 0,070375 | -2,6744 | Excluído, p<0,001 |

Os resultados indicam que o GARCH(1,1) foi o modelo mais forte no critério de erro quadrático: reduziu o MSE em aproximadamente 60,3% em relação ao random walk e apresentou R² fora da amostra de 0,6025. O VIX também teve desempenho forte, com redução de aproximadamente 38,6% do MSE em relação ao random walk e o menor QLIKE da comparação.

O TSMixerX apresentou resultado intermediário em MSE e MAE. Em relação ao random walk, reduziu o MSE em aproximadamente 6,8% e o MAE em aproximadamente 9,9%, com R² fora da amostra positivo, porém pequeno, de 0,0683. Portanto, os outputs não sustentam a conclusão de que o TSMixerX superou os benchmarks principais nesta execução.

### O desempenho do GARCH é plausível?

Sim. O resultado do GARCH(1,1) não é anômalo. A volatilidade financeira apresenta clustering e persistência, e o GARCH foi desenvolvido especificamente para representar essa dinâmica com poucos parâmetros. Em uma amostra relativamente pequena, um modelo parcimonioso e bem alinhado à estrutura do problema pode superar uma rede neural com dezenas de variáveis e muitos parâmetros.

Também é importante observar que os modelos não recebem exatamente a mesma representação dos dados. O GARCH é ajustado aos retornos diários e produz uma previsão para os 22 pregões seguintes. O TSMixerX usa principalmente features agregadas em blocos de 22 pregões. O alvo e as janelas de avaliação são os mesmos, mas o GARCH observa a sequência diária de retornos de forma mais direta. Essa diferença deve ser descrita como uma característica metodológica da comparação, e não interpretada automaticamente como evidência de superioridade universal do GARCH.

Assim, o resultado sugere que a dinâmica condicional tradicional continua sendo muito competitiva para este problema. Ele não demonstra que o TSMixerX seja inútil, mas mostra que a complexidade adicional não trouxe vantagem clara nesta amostra e configuração. Antes da conclusão final, recomenda-se verificar a implementação e a anualização do GARCH e repetir o MCS com configurações de sensibilidade para confirmar a estabilidade da conclusão estatística.

## 3. Diagnóstico importante do QLIKE do TSMixerX

O QLIKE do TSMixerX ficou extremamente alto por causa de uma previsão praticamente nula. Em 9 de junho de 2014, o realizado foi 0,003924 e a previsão foi limitada pelo piso numérico a `1e-10`, gerando uma contribuição individual de aproximadamente 39,2 milhões para o QLIKE médio. Também houve subestimação relevante durante o episódio de estresse de 2020.

Esse resultado deve ser reportado, não escondido. Ele mostra que o TSMixerX teve desempenho razoável em MSE e MAE, mas ainda não está calibrado de forma robusta para o scoring QLIKE. Antes de usar esse modelo como previsão final na etapa de trading, recomenda-se testar uma restrição de positividade mais estável, uma transformação logarítmica do alvo ou uma perda explicitamente alinhada ao QLIKE, além de reavaliar a previsão de junho de 2014 e as previsões do período da COVID-19.

Por essa razão, o MCS foi calculado com perdas MSE, conforme especificado no desenho do experimento e implementado com `arch.bootstrap.MCS`. A 5%, somente o GARCH(1,1) permaneceu no conjunto de modelos não inferior. Isso não significa que todos os modelos excluídos sejam estatisticamente inúteis, mas que eles foram eliminados pelo procedimento MCS sob a perda MSE e a configuração de bootstrap utilizada.

## 4. Robustez

As análises suplementares foram salvas em `results/robustness/`.

| Variante | N | QLIKE | MSE | MAE |
|---|---:|---:|---:|---:|
| TSMixerX, janela expanding | 171 | 1.030.755,50 | 0,001305 | **0,013700** |
| TSMixerX, seed 42 | 171 | 229.454,62 | 0,002361 | 0,017915 |
| TSMixerX, seed 123 | 171 | **0,3769** | **0,001834** | 0,016748 |
| TSMixerX, seed 2024 | 171 | 1.654.890,62 | 0,002058 | 0,018358 |
| TSMixerX com VIX3M, desde 2007 | 91 | 0,8888 | 0,003831 | 0,025833 |

A sensibilidade às seeds é material: o MSE variou de 0,001834 a 0,002361, enquanto o QLIKE variou de 0,3769 a mais de 1,6 milhão. Isso indica que a estabilidade do treinamento ainda precisa ser tratada antes de uma conclusão definitiva sobre a superioridade do TSMixerX.

A janela expanding apresentou MSE melhor que o modelo principal e o menor MAE de todas as variantes TSMixerX, mas continuou com o problema de QLIKE causado por previsões muito baixas. A versão com VIX3M teve desempenho inferior ao modelo principal em MSE e MAE, embora use uma amostra menor, com 91 observações.

## 5. Relação com o VIX e etapa de trading

A correlação entre `vix_mean` e a volatilidade realizada foi aproximadamente 0,618, coerente com o papel do VIX como indicador de expectativa de volatilidade. No arquivo de previsões TSMixerX, o spread de variância definido pelo pipeline, `VIX² - previsão_TSMixerX`, foi positivo em aproximadamente 84,2% das observações, com média de 0,01173.

Esse spread é apenas um diagnóstico inicial para a etapa de Cash-Secured Put Writing. Ele ainda não constitui uma estratégia de trading: faltam regras de seleção de strikes, maturidade das opções, prêmio observado, custos de transação, margem, sizing por Kelly e avaliação de drawdown.

## 6. Como usar os outputs

Os arquivos principais são:

1. `results/metrics.csv`: tabela consolidada de QLIKE, MSE, MAE, R², p-valores e MCS.
2. `results/forecasts_benchmarks.parquet` e `.csv`: previsões dos benchmarks e valores realizados.
3. `results/forecasts_tsmixerx.parquet` e `.csv`: previsões rolling do TSMixerX, VIX e spread de variância.
4. `results/models/tsmixerx/`: 15 arquivos `.pt`, um por janela rolling, além do manifesto de modelos.
5. `results/models/benchmarks/`: pesos e parâmetros dos modelos treináveis e arquivos de parâmetros do HAR/GARCH.
6. `results/hp_search/best_config.json`: hiperparâmetros e features escolhidos.
7. `results/mcs_pvalues.csv`, `results/mcs_membership.csv` e `results/mcs_config.json`: resultados e configuração do MCS oficial.
8. `results/robustness/`: forecasts e métricas das análises de VIX3M, expanding window e seeds.
9. `results/figures/`: gráficos para inspeção e possível adaptação ao texto da dissertação.

Para a dissertação, a leitura mais defensável neste estágio é: GARCH(1,1) foi o melhor benchmark estatístico no conjunto principal, resultado plausível dada a persistência da volatilidade e a parcimônia do modelo; o VIX foi um benchmark de mercado competitivo; o TSMixerX mostrou ganho modesto contra o random walk em MSE/MAE, mas apresentou instabilidade entre seeds e falha importante no QLIKE. A próxima etapa deve corrigir ou investigar essa instabilidade antes de transformar as previsões em sinais de trading.
