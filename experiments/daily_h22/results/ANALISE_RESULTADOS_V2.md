# Análise dos resultados v2 — Previsões diárias H=22

> **Documento autocontido** que descreve o contexto, as escolhas metodológicas,
> as perguntas respondidas, os resultados numéricos e a localização de cada
> artefacto gerado na execução final dos notebooks Colab para previsão diária
> de volatilidade com horizonte h=1..22.

---

## Índice

1. [Contexto e motivação](#1-contexto-e-motivação)
2. [Definições metodológicas](#2-definições-metodológicas)
3. [Estrutura dos ficheiros (guia de referência)](#3-estrutura-dos-ficheiros-guia-de-referência)
4. [O que foi executado — pipeline completo](#4-o-que-foi-executado--pipeline-completo)
5. [Respostas às perguntas do Flavio](#5-respostas-às-perguntas-do-flavio)
6. [Análise dos resultados](#6-análise-dos-resultados)
7. [Lições e próximos passos](#7-lições-e-próximos-passos)

---

## 1. Contexto e motivação

### 1.1 Da v1 para a v2

A **v1** (experiência mensal com blocos de 22 pregões) produzia uma única
previsão por bloco, comprimindo a frequência dos dados. O Flavio pediu:

- Previsões em **frequência diária** (uma nova previsão em cada pregão);
- **22 horizontes** por origem (h=1 a h=22);
- Janelas **8/2/1** (8 anos treino, 2 validação, 1 teste);
- Comparação clara com o **VIX**, incluindo uma versão calibrada;
- Métricas, testes estatísticos e explicabilidade por horizonte.

Este documento descreve a **v2**, que implementa integralmente esses requisitos.

### 1.2 O que é o VIX calibrado?

O VIX é a volatilidade implícita a 30 dias do S&P 500, calculada a partir de
opções. Contém um prémio de risco de variância positivo, o que o torna um
estimador enviesado (tipicamente acima) da volatilidade realizada futura.

Usamos duas versões:

| Modelo | Definição | Interpretação |
|--------|-----------|---------------|
| VIX_raw (VIX bruto) | (VIX_t / 100)² | Previsão risk-neutral, sem ajuste. Preserva a interpretação económica. |
| VIX_calibrated | α + β · (VIX_t / 100)² | Regressão linear estimada apenas no treino de cada janela 8/2/1. Remove o viés médio (prémio de risco). |

Os coeficientes α, β são reestimados anualmente. Esta distinção responde à
pergunta "Should the VIX not be better?" (secção 5.5).

### 1.3 O que é o alvo (target)?

A variável a prever é baseada na **volatilidade realizada Yang–Zhang**,
calculada numa janela móvel de 22 pregões utilizando preços OHLC
(Yang & Zhang, 2000). O cálculo combina três componentes:

1. **Variância overnight** (close_{t-1} → open_t)
2. **Variância open-to-close** (open → close)
3. **Componente Rogers–Satchell** (high/low, drift-independent)

Com o parâmetro de ponderação k = 0.34 / (1.34 + (window+1)/(window-1)),
a variância diária YZ é:

```
YZ_var_diária = v_o + k × v_c + (1 - k) × v_rs
```

No pipeline usamos **variância anualizada** como alvo:

```
y[t] = 252 × YZ_var_diária(t-21, t)
```

Para cada horizonte h: target[t+h] = y[t+h].

O VIX, por seu lado, é uma **volatilidade implícita anualizada em percentagem**
(30 dias corridos). Para comparar as duas grandezas diretamente:

| Grandeza | Definição | Exemplo |
|----------|-----------|---------|
| y (variância anualizada, decimal) | `252 × YZ_var_diária` | 0,0214 |
| Volatilidade anualizada (%) | `100 × sqrt(y)` | 14,6% |
| VIX (vol. implícita anualizada %) | `VIX_t` | 18,4% |

O modelo recebe features[t] e prevê diretamente os 22 horizontes da
variância anualizada y:

```
y_hat[t+1], y_hat[t+2], ..., y_hat[t+22]
```

A métrica **headline** é h=22 (horizonte mensal comparável ao VIX).

---

## 2. Definições metodológicas

### 2.1 Janelas 8/2/1 rolling

Cada janela anual:

1. **Treino**: 8 anos de dados diários
2. **Validação**: 2 anos seguintes (para early stopping e selecção de features)
3. **Teste**: 1 ano seguinte (origens diárias com horizonte até h=22)

A primeira janela: treino 2000–2007, validação 2008–2009, teste 2010. A
última janela: teste 2025. Total: **16 janelas**.

### 2.2 Modelos implementados

| Modelo | Tipo | Descrição |
|--------|------|-----------|
| RW | Benchmark | Random walk: y_hat[t+h] = y[t] |
| HAR | Benchmark | Heterogeneous Auto-Regressive com médias 1, 5, 22 dias |
| GARCH(1,1) | Benchmark | Variância condicional com previsão multi-horizonte |
| VIX_raw | Benchmark | (VIX_t/100)² directo |
| VIX_calibrated | Benchmark | Regressão linear calibrada no treino |
| XGBoost | ML | 22 regressores directos, um por horizonte |
| LSTM | Neural | LSTM com 22 saídas directas, validação e early stopping |
| TSMixerX | Neural | Arquitectura MLP multi-horizonte (Chen et al., 2023) |

### 2.3 Métricas e testes estatísticos

- **QLIKE**: y/y_hat - log(y/y_hat) - 1 — loss robusta para variância.
- **MSE**: erro quadrático médio.
- **MAE**: erro absoluto médio.
- **R² OOS**: 1 - MSE_modelo / MSE_RW (Campbell & Thompson, 2008).
  *Pode ser negativo*: significa que o modelo erra mais que o RW.
- **DM** (Diebold–Mariano): teste de igualdade preditiva contra RW, com
  correcção HAC (lags=21 para h=22) e ajuste FDR para 22 horizontes.
- **GW** (Giacomini–White): teste conditional de igualdade preditiva.
- **MCS** (Model Confidence Set): arch.bootstrap.MCS, loss MSE, 1000
  réplicas bootstrap em blocos (block_size=22), α=5%.

### 2.4 Busca de hiperparâmetros (Optuna)

Executada apenas na **janela 0** (treino 2000–2007, validação 2008–2009).
25 trials, 18 completos (7 pruning real — trial.report + should_prune).
Pruner: MedianPruner. O objetivo foi a QLIKE média de validação nos 22
horizontes. O melhor trial (trial 12, best_validation_qlike=0,2541):

| Parâmetro | Valor |
|-----------|-------|
| n_features | 20 (de 58 candidatas) |
| input_size | 22 pregões |
| n_block | 4 |
| ff_dim | 64 |
| dropout | 0.1 |
| learning_rate | 0.001 |
| batch_size | 64 |
| loss | MAE |
| target_transform | log variance |

Ref: `results/hp_search/best_config.json` e `results/hp_search/tsmixerx_daily.db`.

### 2.5 Pipeline de feature selection

Para cada janela, usando apenas os 8 anos de treino:

1. Remoção de features com >50% missing (VIX3M removido inteiramente)
2. Remoção de variância quase nula
3. Remoção de colinearidade (correlação >0.95)
4. Selecção supervisionada: SelectKBest com f_regression, top-20

As 20 features seleccionadas na janela 0 (congeladas para todas as janelas):
rv_yz_22, vix, atr_14, bollinger_width_20, baa, stress_index, true_range,
aaa, dxy, parkinson, macd, garman_klass, cpi, rsi_14, rogers_satchell,
ret_spx_abs, gold_close, industrial_production, wti_close, rv_diaria,
vix_term_spread (21 itens — rv_yz_22 é a primeira lag do alvo).

### 2.6 Explicabilidade

Usámos **Integrated Gradients** (método de atribuição de gradientes integrados,
não SHAP) para o TSMixerX na janela 0, horizontes h=1 e h=22, com 16 passos
de integração. Para comparação, também calculámos **Shapley Value Sampling**
(10 amostras) na janela 0.

Os ficheiros de saída distinguem claramente integrated_gradients_* de
shapley_*.

Ref: `results/explainability/explainability_manifest.json`.

### 2.7 Referências bibliográficas

- Campbell, J. Y., & Thompson, S. B. (2008). Predicting Excess Stock Returns
  Out of Sample: Can Anything Beat the Historical Average? *Review of Financial
  Studies*, 21(4), 1509–1531.
- Chen, S.-A., Li, C.-L., Yoder, N., Arik, S. O., & Pfister, T. (2023).
  TSMixer: An All-MLP Architecture for Time Series Forecasting. *TMLR*.
- Yang, D., & Zhang, Q. (2000). Drift-Independent Volatility Estimation Based
  on High, Low, Open, and Close Prices. *Journal of Business*, 73(3), 477–491.
- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna:
  A Next-generation Hyperparameter Optimization Framework. *KDD 2019*.
- Bento, J., Saleiro, P., Cruz, A. F., Figueiredo, M. A. T., & Bizarro, P.
  (2021). TimeSHAP: Explaining Recurrent Models through Sequence Perturbations.
  *KDD 2021*.
- Raykar, V. C. et al. (2023). TsSHAP: Robust Model Agnostic Feature-based
  Explainability for Time Series Forecasting. arXiv:2303.12316.
- Xu, J. et al. (2025). A Unified Hyperparameter Optimization Pipeline for
  Transformer-Based Time Series Forecasting Models. arXiv:2501.01394.

---

## 3. Estrutura dos ficheiros (guia de referência)

```
experiments/daily_h22/results/
│
├── evaluation_manifest.json            → Metadados globais
│                                         (modelos, cobertura, MCS, VIX-RV)
│
├── metrics/
│   ├── metrics_h22.csv                 → Métricas headline h=22, 8 modelos
│   ├── metrics_aggregate.csv           → Média macro h=1..22
│   ├── metrics_by_horizon.csv          → QLIKE/MSE/MAE/R²/bias por h=1..22
│   ├── dm_vs_rw_by_horizon.csv         → Diebold–Mariano vs RW, todos h
│   ├── gw_vs_rw_by_horizon.csv         → Giacomini–White vs RW, todos h
│   └── mcs_h22_membership.csv          → Model Confidence Set, h=22
│
├── forecasts/
│   ├── rw.csv / rw.parquet             → Previsões RW
│   ├── har.csv / har.parquet           → Previsões HAR
│   ├── garch_11.csv / garch_11.parquet → Previsões GARCH(1,1)
│   ├── vix.csv / vix.parquet           → Previsões VIX bruto
│   ├── vix_calibrated.csv / vix_calibrated.parquet → Previsões VIX calibrado
│   ├── xgboost.csv / xgboost.parquet   → Previsões XGBoost
│   ├── lstm.csv / lstm.parquet         → Previsões LSTM
│   ├── tsmixerx.csv / tsmixerx.parquet → Previsões TSMixerX
│   ├── all_models.parquet              → Todos os modelos (18 MB)
│   └── partial/                        → Previsões parciais por janela
│
├── inputs/
│   ├── lstm_window_*.parquet           → Inputs X por janela, formato longo
│   ├── tsmixerx_window_*.parquet       → (origin_date, input_date, feature, value)
│   └── xgboost_window_*.parquet
│
├── models/
│   ├── garch/                          → Ajustes GARCH por janela (.pkl)
│   ├── har/                            → Coeficientes HAR por janela
│   ├── lstm/lstm_window_*.pt           → Pesos LSTM, 16 ficheiros (~100 KB)
│   ├── tsmixerx/tsmixerx_window_*.pt   → Pesos TSMixerX, 16 (~150 KB)
│   ├── vix_calibrated/                 → Coeficientes α,β por janela
│   └── xgboost/                        → Modelos XGBoost por janela
│
├── hp_search/
│   ├── best_config.json                → Configuração vencedora (trial 12)
│   └── tsmixerx_daily.db               → Base SQLite Optuna (25 trials)
│
├── feature_selection/
│   ├── selection_stability.csv         → Taxa de selecção (16 janelas)
│   ├── ranking_tsmixerx_window_*.csv
│   └── ranking_lstm_window_*.csv
│
├── training_history/
│   ├── tsmixerx_window_*.csv           → Curvas treino/validação, 16 janelas
│   └── lstm_window_*.csv               → step, train_loss, validation_qlike,
│                                          best_step, stopped_early
│
├── explainability/
│   ├── explainability_manifest.json
│   ├── integrated_gradients_feature_horizon.csv
│   ├── integrated_gradients_lag_horizon.csv
│   ├── temporal_attributions.csv
│   └── temporal_attributions.parquet
│
├── figures/ (20 ficheiros)
│   ├── forecasts_h22.png               → Previsões vs realizado, h=22
│   ├── mae_by_horizon.png / mse_by_horizon.png / qlike_by_horizon.png
│   ├── r2_oos_by_horizon.png
│   ├── vix_vs_rv_volatility.png        → VIX e RV vol. anualizada
│   ├── vix_rv_scatter.png              → Dispersão VIX vs RV
│   ├── vix_rv_rolling_correlation.png  → Correlação rolling VIX-RV
│   ├── convergence_lstm.png / convergence_tsmixerx.png
│   ├── feature_selection_flow.png / feature_ranking_top30.png
│   ├── feature_correlation_clusters.png / feature_family_counts.png
│   ├── feature_selection_stability.png
│   ├── temporal_importance_h??.png     → h=1, 5, 10, 22
│   └── temporal_lag_horizon_heatmap.png
│
└── ANALISE_RESULTADOS_V2.md            → Este documento
```

### 3.1 Schema comum dos forecasts

Cada CSV/Parquet em forecasts/ usa:

| Campo | Descrição |
|-------|-----------|
| model | Nome do modelo |
| window_id | Janela 8/2/1 (0..15) |
| origin_date | Data da previsão (informação disponível) |
| target_date | Data do valor previsto |
| horizon | 1..22 |
| y | Valor realizado (RV_YZ anualizada) |
| y_hat | Previsão do modelo |

---

## 4. O que foi executado — pipeline completo

### Fase 0: Definições e data prep

- Alvo: Yang–Zhang corrigido com componentes overnight, open-to-close e
  Rogers–Satchell (Yang & Zhang, 2000), anualizado x252, janela móvel 22
  pregões.
- Features históricas apenas (hist_exog). VIX3M excluído da amostra principal.
  Macro alinhada por data de publicação.
- Lags traduzidos para o ritmo diário: {22, 66, 132, 264, 528} pregões.
- Input_size = 22 (seleccionado pelo Optuna).

### Fase 1: Geração de janelas 8/2/1

- 16 janelas anuais (teste 2010–2025).
- Split manifesto implícito: treino em 8 anos, validação em 2, teste em 1.
- Todas as origens têm 22 targets futuros disponíveis.

### Fase 2: Benchmarks diários multi-horizonte

- RW, HAR (componentes 1, 5, 22), GARCH(1,1), XGBoost (22 regressores
  directos), LSTM (saída 22, validação, early stopping), VIX_raw, VIX_calibrated.
- Todos produzem o mesmo schema de 22 horizontes por origem.

### Fase 3: TSMixerX e Optuna

- TSMixerX com 22 saídas, 20 features, ff_dim=64, dropout 0.1, loss MAE,
  target_transform = log variance.
- Optuna na janela 0: 25 trials, 18 completos, pruning funcional.
- Configuração congelada nas 16 janelas; weights reajustados anualmente.

### Fase 4: Treino e convergência

- 400 passos máximos, early stopping com paciência de 10 checks (200 passos
  sem melhora).
- Métrica de validação: QLIKE (média h=1..22).
- Históricos salvos em training_history/ (step, train_loss, validation_qlike,
  best_so_far, bad_checks, stopped_early).
- Todos os 32 modelos neurais verificados com reload.

### Fase 5: Feature selection e explicabilidade

- Ranking de features por janela (SelectKBest + f_regression).
- Integrated Gradients + Shapley Value Sampling para TSMixerX.
- Outputs salvos em explainability/ e figures/.

### Fase 6: Avaliação por horizonte

- QLIKE, MSE, MAE, R² OOS para cada h=1..22.
- Macro-média sobre 22 horizontes.
- DM e GW com HAC (lags = horizon, mínimo 5, máximo 21).
- FDR para 22 horizontes.
- MCS com bootstrap em blocos, block_size=22, 1000 reps.

### Fase 7: Outputs

- CSV por modelo, Parquet consolidado (all_models.parquet).
- Figuras de diagnóstico, curvas de convergência, heatmaps temporais.
- Manifesto de avaliação (evaluation_manifest.json) com cobertura, MCS e
  correlação VIX-RV.
- Neural manifest (neural_manifest.json) com 32 checkpoints verificados.

---

## 5. Respostas às perguntas do Flavio

### 5.1 "H=22 and daily, but daily predictions"

A v2 implementa exactamente isto: cada origem emite 22 previsões, uma para
cada horizonte h=1..22, com step_size=1 (todos os pregões). O ficheiro
forecasts/all_models.parquet contém origin_date, target_date e horizon,
permitindo rastrear cada previsão individual.

### 5.2 "Was Realized Volatility the average of 22 days or the accumulated?"

A variável alvo é a variância Yang–Zhang anualizada, calculada numa janela
móvel de 22 pregões:

```
RV_YZ[t] = 252 × YZ_variance(t-21, t)
```

Isto é uma **variância anualizada** da janela de 22 dias (equivalente a
252 × mean daily variance). O ficheiro metrics_by_horizon.csv mostra mean_y
(~0,0214, ou 14,6% em volatilidade anualizada). As unidades estão sempre
identificadas.

### 5.3 "Gráfico VIX e Realized Volatility together"

Foram criados 4 gráficos específicos:

1. figures/vix_vs_rv_volatility.png — VIX_t e 100×sqrt(RV_YZ[t+22]) no
   mesmo eixo, ambos em volatilidade anualizada percentual.
2. figures/vix_rv_scatter.png — dispersão entre VIX e RV, com recta 45°,
   regressão calibrada no treino, destaque 2008/2020.
3. figures/vix_rv_rolling_correlation.png — correlação rolling.
4. A secção 6.9 resume as correlações numéricas.

### 5.4 "Investigate VIX and Realized Volatility correlation"

Correlações na amostra completa (4002 obs, 2010–2025):

| Métrica | Valor |
|---------|-------|
| Pearson (VIX vs RV vol.) | 0,620 |
| Spearman (VIX vs RV vol.) | 0,669 |
| Pearson (VIX² vs RV var.) | 0,504 |
| VIX médio | 18,43% |
| RV vol. média | 12,81% |
| Variance Risk Premium médio | 0,0173 |

Ref: evaluation_manifest.json — secção vix_correlation.

### 5.5 "Should the VIX not be better?"

**VIX calibrado** é o melhor modelo em h=22 (R²=0,396). O VIX bruto tem
R²=0,057 porque o prémio de risco o torna sistematicamente altista (bias
= +0,0173). A calibração remove esse viés. Ver secção 5.1 para a distinção
entre VIX_raw e VIX_calibrated.

### 5.6 "8/2/1 window"

Implementado: cada janela usa 8 anos treino, 2 anos validação, 1 ano teste.
16 janelas no total (teste 2010–2025). O Optuna foi executado apenas na
janela 0. A configuração foi congelada e os pesos reajustados em cada janela.

### 5.7 "Try to show convergence with training and validation loss graph"

Os históricos de treino foram salvos em training_history/ com step,
train_loss, validation_qlike, best_so_far, bad_checks, stopped_early.
Foram gerados:

- figures/convergence_lstm.png — convergência LSTM
- figures/convergence_tsmixerx.png — convergência TSMixerX

Cada gráfico mostra treino e validação, com linha vertical no melhor
checkpoint e indicação de early stopping.

### 5.8 "Does it have early stoppage?"

Sim. TSMixerX e LSTM têm early stopping com:
- Validação a cada 20 passos
- Paciência: 10 checks sem melhora (200 passos)
- Métrica monitorada: QLIKE de validação
- Melhor estado restaurado ao final

O training_history/ regista stopped_early=True/False e best_step.
Exemplo: TSMixerX janela 0 — best_step=200, stopped_early=True (treinou
até ao limite de 400). LSTM janela 0 — best_step=200, stopped_early=True.

### 5.9 "Check whether people from the literature are using Optuna"

Sim. Optuna (Akiba et al., 2019, KDD) é usado em pipelines de forecasting
académicos, incluindo trabalhos que modelam séries temporais com busca
automática (ex. Xu et al., 2025). A interface NeuralForecast também suporta
Optuna nos seus AutoModels. O uso é metodologicamente aceitável e está
documentado no best_config.json com o espaço de busca, sampler, pruner,
número de trials e função objectivo.

### 5.10 "Does it have temporal SHAP values? If so, let's use it"

Usámos **Integrated Gradients** (não SHAP) para atribuições temporais do
TSMixerX, por ser mais rápido e manter a dimensão lag×feature×horizonte.
Para comparação, calculámos também Shapley Value Sampling (10 amostras).
Os ficheiros em explainability/ nomeiam correctamente cada método
(integrated_gradients_* vs. temporal_attributions).

Os heatmaps temporais (figures/temporal_importance_h*.png e
figures/temporal_lag_horizon_heatmap.png) mostram a decomposição por lag
e feature para h=1, 5, 10, 22.

### 5.11 "Create a visualization of the feature selection part"

Foram gerados 5 gráficos de selecção:

1. figures/feature_selection_flow.png — pipeline (missing, low-var,
   colinearidade, SelectKBest)
2. figures/feature_ranking_top30.png — top-30 com score e p-valor
3. figures/feature_correlation_clusters.png — matriz de correlação
   ordenada por clusters
4. figures/feature_family_counts.png — famílias (mercado, volatilidade,
   técnicas, macro, juros, internacionais, lags)
5. figures/feature_selection_stability.png — estabilidade entre 16 janelas

### 5.12 "For forecasts, add the external variables you used for model"

As variáveis exógenas estão disponíveis em inputs/ no formato longo:
origin_date, input_date, input_lag, feature, value. Isto permite reconstruir
exactamente a matriz fornecida a cada modelo. O schema inclui:

- LSTM: 32 ficheiros lstm_window_*.parquet (16 janelas × 2 modelos)
- TSMixerX: 16 ficheiros tsmixerx_window_*.parquet
- XGBoost: 16 ficheiros xgboost_window_*.parquet

As variáveis incluem VIX, VIX term spread, ATR, Bollinger width, Garman–Klass,
Parkinson, Rogers–Satchell, retornos, indicadores técnicos, macro (CPI,
unemployment, industrial production), taxas de juro (Fed Funds, UST 3M/10Y),
índices internacionais, commodities, etc.

### 5.13 "Also, create a .csv results per model"

A estrutura de forecasts/ tem exactamente isto: 8 CSVs (rw, har, garch_11,
vix, vix_calibrated, xgboost, lstm, tsmixerx) + 1 Parquet consolidado
(all_models.parquet). Cada CSV segue o schema da secção 3.1.

### 5.14 "R2_oos of LSTM is negative, but why?"

O R² OOS não é limitado a [0, 1]:

```
R²_OOS = 1 − MSE_modelo / MSE_RW
```

Valor negativo significa que o modelo erra mais que o RW. Na v1 com amostra
mensal, o R² do LSTM era -2,67, indicando forte viés altista (previa
variância média de 0,089 vs alvo de 0,031). Na v2, o LSTM tem R²=0,26
em h=22 — positivo e competitivo, graças à maior amostra diária, early
stopping, transformação log e validação.

### 5.15 "How is the VIX model used?"

VIX_raw: y_hat[t+h] = (VIX_origin_date/100)², sem coeficientes nem treino.
VIX_calibrated: y_hat[t+h] = α + β × (VIX_origin_date/100)², onde α, β são
estimados por regressão OLS nos 8 anos de treino de cada janela, com h=22
como alvo. Os coeficientes estão em models/vix_calibrated/. Ambos usam o
**VIX spot na origin_date**, não a média do bloco anterior.

O forecasts/vix_calibrated.csv inclui vix_t, vix_variance_t, rv_yz_target e
vrp = vix_variance_t − y_hat.

---

## 6. Análise dos resultados

### 6.1 Escopo da execução

| Parâmetro | Valor |
|-----------|-------|
| Janelas completadas | 16 de 16 (todos os 8 modelos) |
| Observações na avaliação | 4002 (teste ~2010–2025) |
| Trials Optuna | 25 (18 completos, 7 pruning) |
| Passos máximos por ajuste | 400 |
| Réplicas MCS | 1000 |
| Modelos neurais treinados | 32 (16 TSMixerX + 16 LSTM) |

### 6.2 Estabilidade da selecção de features

Apenas 8 features foram seleccionadas em todas as 16 janelas:

| Feature | Taxa de selecção |
|---------|:----------------:|
| atr_14 | 100% |
| bollinger_width_20 | 100% |
| garman_klass | 100% |
| parkinson | 100% |
| ret_spx_abs | 100% |
| rv_diaria | 100% |
| vix | 100% |
| vix_term_spread | 100% |

17 variáveis de retorno (ret_nky_simple, ret_ukx_log, ret_wti_simple, etc.)
têm 0% de selecção — nunca ajudaram em nenhuma janela. A selecção é dominada
por estimadores de volatilidade realizada e pelo VIX.

Ref: feature_selection/selection_stability.csv.

### 6.3 Comparação principal — horizonte 22 (headline)

4002 observações, h=22, 16 janelas.

| Modelo | QLIKE | MSE | MAE | R² OOS | Viés |
|--------|:----:|:---:|:---:|:------:|:----:|
| **VIX calibrado** | **0,3111** | **0,001123** | **0,01175** | **0,3956** | −0,0025 |
| HAR | 0,3876 | 0,001271 | 0,01195 | 0,3163 | −0,0058 |
| LSTM | 0,4988 | 0,001376 | 0,01260 | 0,2598 | −0,0059 |
| TSMixerX | 0,8513 | 0,001469 | 0,01398 | 0,2094 | −0,0061 |
| XGBoost | 0,9057 | 0,001754 | 0,02139 | 0,0561 | 0,0038 |
| VIX bruto | 0,4091 | 0,001752 | 0,02479 | 0,0575 | 0,0173 |
| GARCH(1,1) | 0,3365 | 0,001773 | 0,01787 | 0,0459 | 0,0070 |
| RW | 0,4181 | 0,001859 | 0,01492 | 0,0000 | ≈0 |

Ref: metrics/metrics_h22.csv.

**Observações principais:**

1. **VIX calibrado domina isoladamente** em h=22: MSE 39,6% menor que RW,
   R²=0,396, QLIKE mais baixo (0,311). A calibração linear corrige o viés
   altista do VIX bruto (−0,0025 vs. +0,0173).

2. **HAR é segundo melhor e mais consistente**: R²=0,316 em h=22 e o único
   com R² positivo em **todos** h=1..22 (ver 6.5).

3. **LSTM e TSMixerX positivos, mas atrás**: R² 0,210–0,260, ambos com viés
   negativo (~−0,006). LSTM supera TSMixerX em MSE e QLIKE.

4. **XGBoost, VIX bruto e GARCH próximos do RW**: R² entre 0,046 e 0,058.

5. **VIX calibrado tem problema grave de QLIKE em horizontes curtos**:
   A QLIKE macro é 1,9 milhões porque, em h=1 a h=16, a variância prevista
   aproxima-se de zero para algumas observações (ver 6.6).

### 6.4 Agregado macro (média h=1..22)

| Modelo | QLIKE macro | MSE macro | MAE macro | R² OOS macro |
|--------|:----------:|:---------:|:---------:|:------------:|
| **HAR** | **0,1532** | **0,000608** | **0,00736** | **0,2970** |
| GARCH(1,1) | 0,1735 | 0,001295 | 0,01318 | −5,5419 |
| RW | 0,1744 | 0,000842 | 0,00879 | 0,0000 |
| LSTM | 0,2931 | 0,001220 | 0,01013 | −6,3337 |
| TSMixerX | 0,3954 | 0,001204 | 0,01070 | −5,6438 |
| VIX bruto | 0,3403 | 0,001000 | 0,02136 | −4,6273 |
| VIX calibrado | 1,9×10⁶ | 0,000730 | 0,00977 | −3,1663 |
| XGBoost | 0,3580 | 0,001271 | 0,01416 | −4,6907 |

Ref: metrics/metrics_aggregate.csv.

Na média macro, o **HAR** é o melhor modelo — consistente em todos os
horizontes com o menor MSE (0,000608), MAE (0,00736) e QLIKE (0,1532),
além de R² macro positivo (0,297). Todos os demais têm R² macro negativo,
indicando que o RW é competitivo nos horizontes curtos.

### 6.5 Curvas por horizonte

**R² OOS por horizonte (selecção):**

| h | HAR | VIX_cal | LSTM | TSMixerX | XGBoost | GARCH | VIX |
|:-:|:---:|:-------:|:----:|:--------:|:-------:|:-----:|:---:|
| 1 | 0,369 | −47,81 | −83,22 | −70,74 | −56,64 | −69,49 | −65,80 |
| 5 | 0,342 | −1,72 | −4,48 | −4,34 | −3,58 | −3,96 | −2,50 |
| 10 | 0,264 | 0,063 | −0,90 | −0,84 | −0,85 | −0,90 | −0,19 |
| 15 | 0,252 | 0,339 | −0,054 | −0,109 | −0,294 | −0,213 | 0,114 |
| 22 | 0,316 | **0,396** | **0,260** | 0,209 | 0,056 | 0,046 | 0,057 |

Ref: metrics/metrics_by_horizon.csv.

- HAR tem R² positivo em **todos** os horizontes — desempenho notável.
- VIX calibrado só passa a ter R² positivo a partir de h=10, mas dispara nos
  horizontes longos.
- LSTM e TSMixerX cruzam o zero em h=17.
- XGBoost, GARCH e VIX bruto só têm R² positivo a partir de h=20.

**MSE por horizonte:** cresce monotonicamente com h para todos. HAR tem o
menor MSE em todos os horizontes excepto h=22, onde o VIX calibrado o supera.

### 6.6 O problema do QLIKE do VIX calibrado

O VIX calibrado produz previsões muito próximas de zero em horizontes curtos
(h=1 a h=16) para observações com variância realizada baixa. A QLIKE é:

```
QLIKE = y/y_hat − log(y/y_hat) − 1
```

Quando y_hat → 0, y/y_hat → ∞ e a QLIKE explode. A QLIKE macro atinge
1,9 milhões, mas a partir de h=17 normaliza. Isto **não afecta MSE nem MAE**.
Uma solução possível é usar transformação log no target ou uma loss robusta.

### 6.7 Testes Diebold–Mariano vs. RW (h=22)

| Modelo | DM stat | p-valor | p-FDR (22 h) | Significância |
|--------|:-------:|:-------:|:------------:|:-------------:|
| **VIX calibrado** | 2,900 | **0,0037** | **0,0081** | DM*** e FDR*** |
| XGBoost | −1,989 | 0,0467 | 0,0672 | DM* (não após FDR) |
| TSMixerX | −1,780 | 0,0751 | 0,0996 | Não signif. |
| GARCH(1,1) | 1,580 | 0,1141 | 0,1429 | Não signif. |
| LSTM | −0,801 | 0,4232 | 0,4525 | Não signif. |
| HAR | 0,943 | 0,3455 | 0,3828 | Não signif. |
| VIX bruto | 0,141 | 0,8880 | 0,8997 | Não signif. |

Ref: metrics/dm_vs_rw_by_horizon.csv (h=22).

Apenas o **VIX calibrado** rejeita a nula a 5% e sobrevive à correcção FDR
para 22 horizontes. O GW (metrics/gw_vs_rw_by_horizon.csv) confirma: VIX
calibrado rejeita em h=18..22, HAR rejeita em h=1..18 (ambos após FDR).

### 6.8 Model Confidence Set (h=22, loss MSE)

Configuração: arch.bootstrap.MCS, método R, 1000 réplicas, block_size=22,
α=5%.

| Modelo | Incluído | p-valor |
|--------|:--------:|:-------:|
| VIX calibrado | Sim | 1,00 |
| GARCH(1,1) | Sim | 0,30 |
| HAR | Sim | 0,30 |
| LSTM | Sim | 0,30 |
| RW | Sim | 0,30 |
| TSMixerX | Sim | 0,12 |
| VIX bruto | Sim | 0,06 |
| **XGBoost** | **Não** | **0,003** |

Ref: metrics/mcs_h22_membership.csv.

Apenas o XGBoost é excluído a 5%. O VIX calibrado tem o maior p-valor
(1,00). Os restantes são estatisticamente indistinguíveis na perda MSE.

### 6.9 Relação VIX vs. volatilidade realizada (2010–2025)

Correlações sobre 4002 observações (do evaluation_manifest.json):

| Métrica | Valor | Smoke (1 janela, 2010) |
|---------|:-----:|:----------------------:|
| Pearson (VIX vs RV vol.) | **0,620** | 0,229 |
| Spearman (VIX vs RV vol.) | **0,669** | 0,419 |
| Pearson (VIX² vs RV var.) | **0,504** | 0,135 |
| VIX médio | 18,43% | 22,55% |
| RV vol. média | 12,81% | 14,53% |
| Variance Risk Premium | 0,0173 | 0,0288 |

A correlação é muito maior na amostra completa (16 anos) do que no smoke de
1 ano (2010). O VRP médio (0,0173) é coerente com a literatura.

### 6.10 Explicabilidade (Integrated Gradients)

Atribuições para TSMixerX, janela 0, h=1 e h=22, 16 passos de integração.

**Top-5 atribuições absolutas em h=22 (feature level):**

| Feature | Atribuição |
|---------|:----------:|
| CPI | 0,0138 |
| Unemployment | 0,0101 |
| UST 3M | 0,0038 |
| Fed Funds | 0,0033 |
| Volume Médio 22d | 0,0030 |

Macro variáveis (CPI, Unemployment, UST 3M) dominam as atribuições em h=22.
O VIX, que lidera o ranking F-score univariado, tem atribuição modesta (0,0005).
Isto é esperado: F-score mede poder linear isolado; IG captura contribuições
marginais não lineares no modelo completo.

Ref: explainability/integrated_gradients_feature_horizon.csv.

### 6.11 Convergência do treinamento

Com 400 passos máximos e early stopping:

| Modelo | Janelas com early stopping | Best_step médio |
|--------|:--------------------------:|:---------------:|
| TSMixerX | 12/16 (75%) | ~150 |
| LSTM | 14/16 (88%) | ~130 |

Os 4 treinos do TSMixerX que não pararam cedo (janelas 1, 4, 7, 11, 12, 13)
beneficiaram de mais passos para convergir. A configuração com loss MAE e
dropout 0.1 produziu treinos mais estáveis que a configuração do smoke test.

Ref: training_history/ e neural_manifest.json.

---

## 7. Lições e próximos passos

### 7.1 Observações-chave

1. **VIX calibrado** é o melhor modelo em h=22 (R²=0,396, DM significativo,
   MCS incluído). No entanto, sofre de instabilidade numérica no QLIKE em
   horizontes curtos.

2. **HAR** é o melhor modelo na média macro (R² macro=0,297, positivo em
   todos os horizontes). Simplicidade e consistência notáveis.

3. **LSTM e TSMixerX** mostram ganho contra RW em h=22 (R²~0,21–0,26), mas
   ficam atrás dos líderes. A frequência diária e o early stopping
   melhoraram substancialmente o desempenho em relação à v1.

4. **XGBoost, GARCH e VIX bruto** não superam o RW de forma relevante em h=22.

5. **Contraste com a v1**: Na v1 (blocos mensais), o GARCH dominava. Aqui,
   na estrutura diária com 22 saídas directas, o GARCH é um dos piores
   modelos. Modelos que exploram múltiplas features (VIX calibrado, HAR) ou
   estruturas não lineares (LSTM, TSMixerX) beneficiam-se mais do formato
   diário multi-horizonte.

### 7.2 Recomendações

1. **Resolver o QLIKE do VIX calibrado** em horizontes curtos (poda de
   previsões extremas, transformação log ou loss robusta).

2. **Testar ensembles**: HAR + VIX calibrado (complementaridade nos
   horizontes) e HAR + LSTM.

3. **Análise de robustez** com múltiplas seeds para os modelos neurais.

4. **Incorporar VIX3M** como análise suplementar a partir de 2007 (excluído
   da amostra principal, como solicitado).

5. **Preparar previsões** do VIX calibrado e do HAR para a etapa de
   estratégia de Cash-Secured Put Writing.

6. **Validar Yang–Zhang** contra implementação independente e dados
   sintéticos (pendente da Fase 0 original).

### 7.3 Como usar os outputs

| O quê | Onde está |
|-------|-----------|
| Métricas headline h=22 | metrics/metrics_h22.csv |
| Métricas macro médias | metrics/metrics_aggregate.csv |
| Métricas por horizonte | metrics/metrics_by_horizon.csv |
| Teste DM vs RW | metrics/dm_vs_rw_by_horizon.csv |
| Teste GW vs RW | metrics/gw_vs_rw_by_horizon.csv |
| Model Confidence Set | metrics/mcs_h22_membership.csv |
| Previsões (CSV por modelo) | forecasts/{modelo}.csv |
| Previsões (consolidadas) | forecasts/all_models.parquet |
| Inputs X (formato longo) | inputs/{modelo}_window_*.parquet |
| Pesos LSTM | models/lstm/lstm_window_*.pt |
| Pesos TSMixerX | models/tsmixerx/tsmixerx_window_*.pt |
| Configuração vencedora | hp_search/best_config.json |
| Base Optuna (trials) | hp_search/tsmixerx_daily.db |
| Estabilidade features | feature_selection/selection_stability.csv |
| Rankings por janela | feature_selection/ranking_*.csv |
| Históricos treino | training_history/{modelo}_window_*.csv |
| Atribuições IG | explainability/integrated_gradients_*.csv |
| Atribuições Shapley | explainability/temporal_attributions.csv |
| Figuras | figures/*.png (20 ficheiros) |
| Manifesto global | evaluation_manifest.json |
| Manifesto modelos neurais | models/neural_manifest.json |
