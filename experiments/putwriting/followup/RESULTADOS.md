# Resultados da revisão com os dados complementares

## Previsão

O painel tem 6.539 pregões do SPX. Ao longo das 16 janelas anuais, de 2010 a 2025, cada modelo produz 4.024 previsões para cada horizonte de 1 a 22 pregões. No horizonte 22, 4.002 origens podem ser avaliadas. As últimas 22 previsões continuam disponíveis para uso prospectivo. A checagem por data e horizonte confere as observações com o alvo do painel.

No horizonte 22, com a correção de Duan, o HAR com componentes de Rogers–Satchell tem QLIKE de **0,372** e MSE de **0,001240**. O HAR–VIX com log(VIX²) tem QLIKE de **0,312** e MSE de **0,001119**. O Clark–West para esse par, na escala em que os modelos foram estimados, tem estatística de **8,04** e p ajustado de **6,81×10⁻¹⁶** com HAC de 21 pregões. A evidência é retrospectiva, pois a primeira execução já havia revelado a instabilidade em 2020.

Nas 228 origens avaliáveis de 2025, o HAR–VIX tem MSE ligeiramente menor que o HAR (0,001547 vs. 0,001586) e QLIKE maior (0,375 vs. 0,310). No conjunto completo, o VIX calibrado fica próximo do HAR–VIX em QLIKE (0,313 vs. 0,312). Os resultados não apontam uma vantagem uniforme.

Em 16/03/2020, com variância observada de 0,2853, o HAR–VIX revisado prevê **0,2081**, incluindo smearing. Mantendo os mesmos dados, janelas e as mesmas componentes diárias, a versão com VIX² em nível prevê **24,41**. A variante com as componentes Yang–Zhang suavizadas e VIX² em nível chega a **72,40** após smearing (59,43 antes). Os 64,74 reportados na primeira entrega vêm da execução histórica, com ajustes diferentes. O teste controlado mostra de onde vem a extrapolação e permite comparar as duas especificações em condições iguais. [Tabela do choque](results/har_crisis_h22.csv), [gráfico](results/har_crise_2020.png) e [comparação completa](results/har_specification_metrics.csv).

A tabela de previsão guarda a versão bruta e a corrigida. No HAR–VIX, o smearing reduz o viés médio de −0,00606 para −0,00258 e o QLIKE de 0,376 para 0,312 em H=22. No TSMixerX o QLIKE cai de 0,773 para 0,686, com pequeno aumento do MSE. No LSTM o QLIKE sobe de 0,872 para 0,892. A correção estima a retransposição pela distribuição dos resíduos no treino. Seu desempenho depende do modelo, horizonte e métrica. [Resultados por horizonte](results/forecast_metrics.csv) e [Clark–West](results/clark_west_log.csv).

XGBoost foi reestimado nos 22 horizontes de cada janela. LSTM e TSMixerX foram retreinados nas 16 janelas, com checagem de reabertura de cada um dos 32 checkpoints e hiperparâmetros da busca anterior congelados. Os ajustes limpos e seus resultados estão em [neural_fallback](neural_fallback/README.md). A busca nova iniciada em CPU ficou parcial. O [notebook de GPU](notebooks/neural_gpu.ipynb) permite fazer a busca completa no painel corrigido.

Texto sugerido para a dissertação: “Estimamos regressões diretas para o log da variância anualizada de Yang–Zhang, calculada nos 22 pregões que terminam na data alvo de cada horizonte. Para representar os componentes diário, semanal e mensal do HAR, usamos os logaritmos das médias de 1, 5 e 22 pregões da variância intradiária diária de Rogers–Satchell, observada até o fechamento da origem da previsão. O HAR–VIX acrescenta o log da variância implícita anualizada do VIX, convertido para decimal. Para gerar previsões na escala de variância, multiplicamos a exponencial do valor previsto pelo fator de Duan de cada janela de treino e horizonte.”

## Marcação diária

As 120 entradas têm bid, ask e mid iguais aos da rolagem original, e as 120 liquidações coincidem com o SOQ na data efetiva. O strike das opções fica a no máximo 0,116% do fechamento do SPX na entrada. A IV do contrato supera a LIVE por 0,079 p.p. em média, com desvio padrão de 0,304 p.p. Isso sustenta seu uso como aproximação para o sinal, com o reconhecimento de que a LIVE tem moneyness e prazo fixos interpolados.

O backtest vai de 17/12/2015 a **19/12/2025**, último vencimento entregue. O Excel e os novos arquivos vão até 31/12/2025, mas os 120 contratos incluem como última entrada a de 21/11/2025. Para simular exposição a opções após o vencimento de dezembro, falta a rolagem de 19/12/2025 e o contrato correspondente. O retorno desde o fechamento de 31/12/2024 até o último vencimento está no recorte `2025_to_last_expiry`.

O VIX bruto continua como referência preditiva. Como sinal de negociação, ficou **120 de 120 rolagens em caixa** com as duas fontes de IV. Usar VIX nos dois lados deixa o VRP identicamente zero. Com LIVE, há dois dias de VRP positivo, em 20/09/2022 e 27/10/2022. Nenhum coincide com a véspera de uma entrada. O ponto de 20/09/2022, com LIVE de 47,2841% e VIX de 27,16%, merece conferência no terminal. Ele foi mantido como recebido.

Em 23 pregões do SPX falta o mid do contrato. Em 18 há ask do dia e em cinco faltam bid e ask. A análise com o mid observado preserva essas lacunas. A triagem também encontrou 60 mids e 26 asks abaixo de um limite inferior aproximado, calculado com o fechamento do índice e o USGG1M defasado. Isso é um sinal para conferir sincronização e qualidade de marcação. Sem timestamp e sem a convenção exata da taxa, a triagem não identifica uma oportunidade de arbitragem. [Auditoria dos dados](results/data_audit/AUDITORIA.md) e [conferência das marcas](results/data_audit/panel_and_marks.json).

Os 36 cenários cruzam nove modelos de previsão, duas fontes de IV e execução no bid ou no mid. A tabela abaixo usa LIVE, execução no bid e marcação no mid observado. O número de retornos diários válidos varia de acordo com a posição vendida. O drawdown usa somente os fechamentos disponíveis.

| Estratégia ou referência | Retorno total | Sharpe diário anualizado | Sortino diário anualizado | Drawdown observado |
|---|---:|---:|---:|---:|
| HAR | 82,48% | 0,66 | 0,89 | −15,63% |
| HAR–VIX | 66,78% | 0,48 | 0,67 | −14,23% |
| Passeio aleatório | 92,03% | 0,84 | 1,17 | −10,32% |
| TSMixerX | 95,39% | 0,75 | 0,98 | −18,38% |
| Venda integral dos mesmos contratos | 117,03% | 0,60 | 0,80 | −26,57% |
| PUT oficial | 115,35% | 0,50 | 0,66 | −28,93% |
| Caixa (também o caso VIX bruto) | 23,75% | n.d. | n.d. | −0,002% |

O HAR–VIX melhora o QLIKE em relação ao HAR na amostra completa e tem Sharpe menor neste cenário de investimento. A marcação aproximada que preenche só as lacunas reduz o Sharpe do HAR de 0,66 para 0,60 e o da venda integral de 0,60 para 0,53. Os retornos totais ficam iguais porque os fechamentos inicial e final têm marcas observáveis. Em 2025 há 11 entradas e 12 vencimentos, incluindo a posição de dezembro de 2024. [Métricas diárias completas](results/performance_daily.csv), [resultados nos vencimentos](results/performance_expiry.csv) e [gráfico de patrimônio](results/patrimonio_diario_live_bid.png).

O bootstrap pareado reamostra retornos de vencimento a vencimento em blocos de 1, 3 e 6 rolagens. Os intervalos condicionam aos contratos, sinais, modelos ajustados e retornos já observados. O bootstrap separado usa o histórico do SPX de 03/01/2000 a 31/12/2025 para gerar caminhos diagnósticos, sem reprecificação de opções e sem refazer a estratégia. [Bootstrap pareado](results/bootstrap_paired_expiry.csv) e [metadados do SPX](results/bootstrap_index_metadata.json).
