"""Write a Portuguese reading guide from the actual persisted results."""
import json
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]


def num(value, digits=3):
    return 'não definido' if pd.isna(value) else f'{value:.{digits}f}'


def run():
    results = BASE / 'results'
    manifest = json.loads((results / 'run_manifest.json').read_text())
    metrics = pd.read_csv(results / 'performance_expiry.csv')
    primary = metrics.query("iv_source == 'iv' and premium_source == 'bid'")
    strategy = primary.query("portfolio == 'strategy'")
    rows = []
    for row in strategy.itertuples(index=False):
        rows.append(f'| {row.model} | {100*row.total_return:.2f}% | {num(row.expiry_sharpe_annualized)} | {num(row.expiry_sortino_annualized)} | {100*row.expiry_max_drawdown:.2f}% |')
    for portfolio, label in [('put', 'Mesmos contratos, colateral 100%'),
                             ('official_PUT_close_proxy', 'PUT oficial, proxy por fechamentos'),
                             ('cash', 'Caixa aproximado')]:
        row = primary[primary.portfolio == portfolio].iloc[0]
        rows.append(f'| {label} | {100*row.total_return:.2f}% | {num(row.expiry_sharpe_annualized)} | {num(row.expiry_sortino_annualized)} | {100*row.expiry_max_drawdown:.2f}% |')
    cw = pd.read_csv(results / 'forecast_revision/metrics/clark_west_har_vix.csv').query('horizon == 22').iloc[0]
    fm = pd.read_csv(results / 'forecast_metrics_h22.csv').set_index('model')
    bootstrap = pd.read_csv(results / 'bootstrap_paired_monthly.csv')
    selected_boot = bootstrap.query("model == 'HAR' and iv_source == 'iv' and premium_source == 'bid' and comparator == 'official_put_return'")
    zero_inside = (selected_boot.ci_low <= 0) & (selected_boot.ci_high >= 0)
    bootstrap_sentence = ('No HAR com LIVE/bid contra o PUT oficial, os intervalos abrangem zero nas três configurações.'
                          if len(selected_boot) == 3 and zero_inside.all()
                          else 'Consulte os intervalos de cada cenário no CSV; sua interpretação depende da configuração de blocos.')
    text = f'''# Resultados executados e pendências para a dissertação

Esta revisão responde ao documento enviado após a qualificação. O código e os resultados estão em `experiments/putwriting/`; o [README](README.md) explica a reprodução e cada arquivo. As saídas anteriores foram preservadas. A execução financeira é um cenário de pesquisa provisório com preços de entrada observados, e não um backtest diário finalizado.

## Execução e correção adicional

HAR e HAR-VIX foram ajustados nas 16 janelas anuais 8/2/1. Cada um tem 4.002 previsões fora do treino por horizonte. O HAR-VIX usa VIX contemporâneo na origem, em variância anualizada decimal; o teste aninhado usa exclusivamente o par HAR/HAR-VIX, com HAC e correção de multiplicidade.

A revisão final encontrou um problema anterior no treinamento neural: as últimas 22 origens de validação geravam alvos dentro do teste. Na primeira janela, isso representava 253 entradas de alvo em 22 datas de teste, de 04/01/2010 a 03/02/2010. Elas podiam influenciar a busca Optuna e a seleção dos pesos. O código passou a remover as origens que cruzam cada fronteira de treino/validação e validação/teste. A nova busca e os ajustes neurais foram executados em `results/neural_purged/`, separados dos resultados antigos, usando um estudo novo e arquivos com identificação da política temporal. Os resultados financeiros desta pasta usam essas novas previsões de LSTM e TSMixerX; as seis previsões estatísticas anteriores são preservadas, pois seus treinos terminam dois anos antes do teste. Há controles para impedir mistura de checkpoints incompatíveis na retomada.

Foram simulados {manifest['contracts']} contratos, com entradas de {manifest['first_entry']} até 20/12/2024 e último vencimento em {manifest['last_expiry']}. Os nove modelos foram combinados com LIVE/VIX e bid/mid, formando 36 cenários. As 11 entradas de 2025 foram excluídas porque SPX, LIVE e VIX terminam em 31/12/2024. O vencimento de janeiro de 2025 pertence à posição aberta em dezembro de 2024. Os arquivos de previsão ainda contêm origens de 2025, mas elas não geram posições.

## O HAR-VIX não mostrou melhora uniforme

Em h=22, o MSE do HAR é {fm.loc['HAR','MSE']:.8f}, e o do HAR-VIX é {fm.loc['HAR-VIX','MSE']:.8f}. O QLIKE é {fm.loc['HAR','QLIKE']:.6f} e {fm.loc['HAR-VIX','QLIKE']:.6f}, respectivamente. O Clark-West principal na escala log tem p unilateral {cw.CW_log_pvalue_one_sided:.8f} e p ajustado BH {cw.CW_log_pvalue_BH_across_horizons:.8f}. A aproximação exploratória na escala de variância tem p {cw.CW_pvalue_one_sided:.6f}. Esses testes usam perdas diferentes; o resultado em log não demonstra melhora em MSE de variância.

O maior erro vem da extrapolação da especificação log-linear em março de 2020. Em 16/03/2020, VIX=82,69 implica variância 0,68376361, contra máximo 0,2304 no treino de 2010 a 2017. O coeficiente VIX em h=22 é aproximadamente 11,8079; sua contribuição ao preditor log é 8,0738, e a previsão exponenciada chega a 64,7427 para realização 0,2853. A conversão percentual foi conferida e está correta. A regressão é instável fora da faixa de treino. Os [maiores valores previstos](results/har_vix_extreme_forecasts.csv) foram preservados, sem limitar previsões depois de observar o teste. Uma transformação alternativa deve ser escolhida no treino/validação em outra especificação, com transparência sobre essa decisão.

## Cenário principal: LIVE e prêmio bid

Todos os valores abaixo são observados apenas nas datas de vencimento. Sharpe e Sortino usam o retorno do caixa do mesmo período e anualização aproximada por raiz de 12. O retorno acumulado cobre o período completo, não um ano. O drawdown pode deixar de capturar perdas maiores entre vencimentos.

| Carteira ou modelo | Retorno acumulado | Sharpe | Sortino | Drawdown nos vencimentos |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

Não selecionar um modelo para implantação apenas por essa tabela. Bid e mid são dois cenários de preço observado, sem taxas ou impostos; não há comprovação de execução intradiária. O benchmark oficial usa fechamentos nas mesmas datas, que não são simultâneos à liquidação AM. A venda integral dos contratos fornecidos é um benchmark próprio, distinto do índice Cboe.

As [curvas LIVE/bid](results/equity_expiry_iv_bid.png), [curvas VIX/bid](results/equity_expiry_vix_bid.png), [operações completas](results/expiry_ledgers.csv) e [métricas de todos os cenários](results/performance_expiry.csv) permitem examinar a relação entre retorno e exposição. No cenário em que o próprio VIX é previsão e IV, o VRP é zero por construção e a carteira fica no caixa.

## Bootstrap e alcance da evidência

O bootstrap diário do SPX gera 2.000 trajetórias, com aproximadamente 20 blocos de retornos contíguos por trajetória, e preserva cinco caminhos completos para inspeção. A observação sem preço em 11/09/2001 foi explicitamente registrada e excluída do calendário de preços válidos; não foi preenchida. Esse exercício não reprecifica opções nem testa sozinho a estratégia.

O bootstrap financeiro gera intervalos da diferença média de retorno entre estratégia e benchmark a partir dos mesmos blocos mensais de retornos pareados. Há 2.000 réplicas com blocos de um, três e seis meses, com aviso quando o número de blocos não sobrepostos disponíveis é pequeno. {bootstrap_sentence} A fração de simulações com diferença não positiva não é um p-valor. Não há evidência aqui que autorize chamar a estratégia de comprovadamente superior ou robusta a toda trajetória alternativa do mercado. Reamostrar o pipeline completo exigiria hipóteses adicionais e dados para reconstruir a cadeia de opções.

## Dados que precisam ser confirmados ou complementados

1. Identificação contratual AM/PM e timestamps das cotações, para verificar a ordem de liquidação e da nova entrada. Todos os 120 valores informados de liquidação conferem com SPX_SOQ pela data efetiva, inclusive feriados; isso não substitui o cadastro dos contratos.
2. Bid/ask ou preços de marcação diários dos mesmos contratos, entre entrada e vencimento. Sem eles não calculamos Sharpe, Sortino ou drawdown diários da estratégia. O [modelo de campos necessários](results/data_audit/missing_daily_option_quotes_template.csv) facilita a coleta.
3. Identificação do campo Bloomberg de TBILL1M, convenção de yield/desconto e day count. A conta atual assume yield simples ACT/360 e reinvestimento diário, usando exclusivamente a cotação conhecida antes de cada dia; não replica retorno total de T-bills.
4. Extensão das séries SPX, LIVE e VIX para 2025, se esse período entrar na estratégia, e confirmação de que a IV LIVE representa a opção efetivamente negociada.
5. Revisão da interpretação da previsão pontual como média condicional de variância. Exponenciar previsão em log sem correção não garante essa equivalência. Datas de divulgação e vintages das variáveis macro também não foram fornecidas; os atrasos existentes são aproximações, não dados point-in-time comprovados.

O [manifesto](results/run_manifest.json) identifica dados, versões, código e hipóteses; [verification.json](results/verification.json) registra as reconciliações determinísticas da execução. As fontes metodológicas e o comando dos testes estão no README. Os artefatos não substituem essas confirmações de mercado.
'''
    (BASE / 'RESULTADOS_E_PENDENCIAS.md').write_text(text, encoding='utf8')


if __name__ == '__main__':
    run()
