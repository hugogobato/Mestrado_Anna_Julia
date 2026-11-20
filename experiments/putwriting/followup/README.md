# Revisão dos modelos e backtest diário até 2025

Os [resultados](RESULTADOS.md) reúnem as novas tabelas de previsão, a comparação das especificações HAR, a marcação diária e os limites de interpretação. A [auditoria](results/data_audit/AUDITORIA.md) descreve o conteúdo dos dois ZIPs e as conferências. Os resultados da primeira entrega continuam na [pasta anterior](../results/).

## Revisão da previsão

O alvo continua sendo a variância anualizada Yang–Zhang dos 22 pregões que terminam em `t+h`. No horizonte 22, toda essa janela está depois da origem da previsão. O HAR principal agora usa o log das médias de 1, 5 e 22 pregões de uma variância intradiária diária de Rogers–Satchell, calculada com OHLC do SPX. Isso adapta a estrutura multiescala de Corsi ao alvo Yang–Zhang e à disponibilidade de dados diários. A medida Rogers–Satchell usada nos regressores termina no fechamento e não inclui o retorno overnight. O HAR–VIX acrescenta `log((VIX/100)^2)` na origem.

As seis variantes da [comparação](results/har_specification_metrics.csv) cruzam a fonte das três componentes (Rogers–Satchell diário ou a série Yang–Zhang suavizada da implementação anterior) com três formas de incluir o VIX (sem VIX, com VIX² em nível ou com log(VIX²)). Todas usam as mesmas origens de treino e teste. A comparação de nível e log mantém as outras decisões fixas. O [teste Clark–West](results/clark_west_log.csv) usa as previsões lineares em log, com HAC de 21 pregões, sensibilidade de 42, alternativa unilateral e ajuste Benjamini–Hochberg nos 22 horizontes de cada par. A revisão foi motivada pela instabilidade observada na análise anterior, então o período já examinado é uma comparação retrospectiva.

XGBoost, LSTM e TSMixerX refazem a seleção de variáveis na amostra de treino de cada janela, após remover `ret_nky_simple`, `ret_ukx_simple` e `ret_wti_simple`. A remoção de `vix_term_spread` fecha uma via indireta para o VIX3M, que a especificação anterior excluía. Os log-retornos permanecem com o tratamento da base anterior. As defasagens macro continuam sendo aproximações, pois o Excel não inclui datas de publicação e séries de vintages.

Para cada regressão ajustada em log, a previsão em variância é `exp(previsão_log) × média(exp(resíduo_log))`. A média de Duan usa resíduos do próprio modelo nas origens de treino e é calculada por janela e horizonte. A validação neural escolhe a parada pelo QLIKE da previsão sem smearing. Os resíduos dentro da amostra podem ser pouco dispersos em modelos flexíveis e o fator agregado pode não capturar mudanças de dispersão entre regimes. As [métricas](results/forecast_metrics.csv) comparam as duas retransposições. Passeio aleatório, VIX bruto, VIX calibrado e GARCH operam diretamente em variância.

Os alvos do treino e da validação terminam antes da partição seguinte. Todos os modelos geram sinais até o fechamento de 31/12/2025. No horizonte 22, a última origem com alvo observado é 28/11/2025. Os 22 sinais posteriores podem ser usados para decisões após a respectiva origem, mas ficam fora das métricas de previsão H=22.

## Relógio e marcação do backtest

O patrimônio começa no fechamento de 17/12/2015, antes da primeira venda, e termina no fechamento de 19/12/2025, data do último SOQ disponível para os 120 contratos. O sinal para cada rolagem vem do pregão SPX imediatamente anterior. A alocação usa o VRP em variância anualizada e a posição no histórico dos 504 pregões anteriores. A put antiga é liquidada pelo SOQ pela manhã e a nova posição é aberta com as cotações de fim do dia. Como as cotações não têm timestamps intradiários, bid e mid são cenários de execução, sem taxas, impacto ou slippage adicional. As quantidades fracionárias ficaram limitadas pelo caixa disponível antes de receber o prêmio.

O cenário principal marca a obrigação vendida pelo mid observado. Quando falta mid em uma data com posição, o NAV e os retornos que dependem desse fechamento ficam ausentes. O drawdown usa os fechamentos disponíveis e pode perder perdas em dias sem marca. Duas sensibilidades aproximadas usam (1) o ask do dia quando falta mid, seguido da última marca válida quando faltam ambos e (2) ask sempre que disponível. O último preço negociado não é usado, pois pode ser antigo. Sharpe e Sortino usam excesso sobre o caixa e fator de 252 pregões. A coluna `common_days_sharpe_annualized` usa somente as datas com retorno diário disponível também para a estratégia daquela linha de cenário. Em `2025_to_last_expiry`, a base é o fechamento de 31/12/2024 e mantém a posição que já estava aberta.

O USGG1M é rendimento anual em porcentagem. O caixa usa a última observação estritamente anterior ao dia de accrual e capitaliza o rendimento simples ACT/360 por dia corrido. É uma aproximação até confirmar a convenção na tela DES. O benchmark de venda integral vende os mesmos contratos. O PUT oficial usa retornos do índice no fechamento, com regras e relógio próprios. Os retornos entre datas de vencimento são medidos no fechamento e incluem o custo de abrir a próxima posição naquele dia quando aplicável.

## Reproduzir

Com Python 3.12, execute a partir da raiz do repositório. O backtest usa os dois ZIPs locais enviados pela Anna. A etapa neural usa o Excel e o painel preparado localmente. O [notebook opcional para Colab com GPU](notebooks/neural_gpu.ipynb) recria o painel a partir do Excel, roda nova busca e novo treino e baixa os resultados em um ZIP. Para repetir o backtest com uma nova execução neural, use o par de Parquets dessa mesma execução.

```bash
python -m pip install -r experiments/putwriting/followup/requirements.txt
python -m experiments.putwriting.followup.panel
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python -m experiments.putwriting.followup.statistical

python -m experiments.putwriting.followup.run \
  --original-zip 'BASES DE DADOS-20261117T124409Z-1-001.zip' \
  --followup-zip 'DADOS FALTANTES-20261120T201147Z-1-001.zip' \
  --neural-forecasts \
      experiments/putwriting/followup/neural_fallback/results/forecasts/lstm.parquet \
      experiments/putwriting/followup/neural_fallback/results/forecasts/tsmixerx.parquet
python -m experiments.putwriting.followup.figures
```

A execução entregue refaz todos os ajustes neurais na base limpa com os [hiperparâmetros da busca anterior](neural_fallback/README.md) congelados. Assim, repetir o backtest com os Parquets salvos não exige GPU. A [nova busca em Colab](notebooks/neural_gpu.ipynb) é opcional. O [manifesto](results/run_manifest.json) registra os hashes dos arquivos executados.

## Referências

- Componentes em múltiplos horizontes e variância diária de OHLC. [Corsi (2009)](https://doi.org/10.1093/jjfinec/nbp001) e [Rogers e Satchell (1991)](https://doi.org/10.1214/aoap/1177005835).
- Correção de retransposição e comparação de modelos aninhados. [Duan (1983)](https://doi.org/10.1080/01621459.1983.10478017) e [Clark e West (2007)](https://www.kansascityfed.org/documents/5368/pdf-RWP05-05.pdf).
- Liquidação AM por SOQ nas opções SPX padrão e regras do benchmark. [Liquidação de SPX padrão, Cboe](https://cdn.cboe.com/resources/spx/Settlement_of_Standard_AM_Settled_SP_500_Index_Options.pdf) e [metodologia Cboe PutWrite](https://cdn.cboe.com/resources/indices/governance/Cboe_SP_500_PutWrite_Indices_Methodology.pdf).
