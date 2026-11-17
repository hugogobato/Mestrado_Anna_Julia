**ALTERAÇÕES NO QUE JÁ FOI FEITO:**

**1\. Inclusão do VIX como Regressor no Modelo HAR (HAR-VIX)**

* **Implementação:** Adicionar a variável de volatilidade implícita (VIX) como um regressor extra na equação do modelo HAR existente.  
    
* **Justificativa:** O modelo HAR baseia-se em dados do passado, capturando resultados inerciais (*backward looking*), ao passo que o VIX traz a expectativa futura dos agentes no mercado de opções (*forward looking*). A combinação de ambas as informações (efeito inercial \+ olhar para frente) pode gerar um modelo preditivo superior, inclusive em relação a algoritmos mais complexos.


**2\. Testes de Comparação para Modelos Aninhados**

* **Implementação:** Incluir testes estatísticos específicos para modelos aninhados. A sugestão do Prof. Julio foi utilizar *Clark-West* ou testes de *Likelihood Ratio* na rotina de avaliação de acurácia.  
    
* **Justificativa:** Ao comparar modelos da mesma família em que um possui mais parâmetros que o outro (por exemplo, um GARCH 1-1 contra um GARCH 2-1), há um custo adicional de parametrização. O teste pondera e desconta esse custo para verificar se a inclusão de mais parâmetros resulta, de fato, em um ganho real de acurácia na previsão, evitando *overfitting*.


**3\. Teste de Robustez via *Bootstrapping* de Séries Temporais**

* **Implementação:** Desenvolver uma rotina de *bootstrapping* em blocos para a série temporal do índice. A série original deve ser dividida em vários blocos (por exemplo, 20 blocos não muito pequenos para não perder a dependência da série) e, em seguida, reorganizada interativamente.  
    
* **Justificativa:** O uso de uma única série temporal deixa a pesquisa refém daquela trajetória específica, que é apenas um dos caminhos possíveis que o mercado poderia ter tomado. O *bootstrapping* permite testar a estratégia nesses "universos alternativos" reorganizados, comprovando cientificamente que os resultados do *backtest* são robustos e não um mero acaso da série histórica utilizada.

**IMPLEMENTAÇÕES QUE AINDA FALTAM FAZER:**

Precisamos implementar a próxima etapa do trabalho, que é simular o funcionamento da estratégia de Cash-Secured Put Writting orientada pelos outputs dos modelos preditivos.  
Para cada um dos modelos preditivos, será necessário:

### **Passo 1: Dados Necessários (Já Coletados)**

* **Dados de Preço e Opções:** Série histórica do S\&P 500 (SPX) e os prêmios reais de mercado (cotações de entrada e vencimento) das opções de venda (*puts*) do S\&P 500, estritamente no nível *At-The-Money* (ATM) e com vencimento mais próximo a 1 mês (22 dias úteis).  
    
* **Dados de Volatilidade Implícita:** Série diária de volatilidade implícita dessa mesma opção (put ATM de 1 mês) proveniente do método LIVE da Bloomberg (ou alternativamente o VIX, para teste de robustez).  
    
* **Remuneração do Caixa:** Série histórica de rendimentos das letras do Tesouro norte-americano de 1 mês (*T-bills*).  
    
* **Benchmark:** Série histórica de retornos do índice CBOE S\&P 500 PutWrite Index (PUT Index) para efeito de comparação passiva.


### **Passo 2: Cálculo do Prêmio de Risco de Variância (VRP) \- O Sinal**

* **Conversão de Bases:** Como a volatilidade implícita (IV) vem anualizada e em percentual, é necessário dividi-la por 100 para transformar em decimal, e depois elevá-la ao quadrado para transformá-la em variância ($I{V}^{2}$). A previsão de variância realizada (RV), gerada pelo modelo preditivo para 22 dias à frente, já estará em variância anualizada decimal, não precisando de conversão.  
    
* **Fórmula do VRP:** É preciso calcular diariamente o prêmio de risco de variância subtraindo a variância realizada esperada da variância implícita: $VR{P}_{t}=I{V}_{t}-{E}_{t}(R{V}_{t+22})$.


### **Passo 3: Dimensionamento Dinâmico da Posição (Regra de Alocação)**

* **Cálculo do Quantil Histórico:** Para cada dia $t$, deve-se calcular a posição relativa (quantil ${q}_{t}$, variando de 0 a 1\) do $VR{P}_{t}$ em relação à sua própria distribuição em uma janela móvel dos últimos 2 anos (504 pregões). *Atenção:* O quantil deve ser calculado apenas com as observações que antecedem $t$, sem incluir o próprio dia ou dados futuros para evitar viés de *look-ahead*.

* **Definição do Peso da Posição (${w}_{t}$):**

  * Se $VR{P}_{t}\leq 0$: O peso será ${w}_{t}=0$. Não há venda de opções e o capital fica 100% no caixa rendendo a taxa livre de risco (*T-bills*).  
      
  * Se $VR{P}_{t}>0$: O peso será ${w}_{t}={w}_{max}\cdot {q}_{t}$. A fração do colateral comprometida na venda cresce linearmente com a força do quantil.  
      
  * O ${w}_{max}$ é igual a 1, significando que o colateral é comprometido no máximo até 100%, caracterizando a estratégia *cash-secured* (sem alavancagem).  
    

### **Passo 4: Simulação Financeira da Operação (*Cash-Secured Put Writing*)**

* **Lógica de Entrada:** Com base no peso ${w}_{t}$, é necessário simular a venda da opção *put* ATM e o recebimento do prêmio de mercado. O capital exigido para cobrir o exercício (*strike*) deve ser depositado em caixa.

* **Remuneração do Colateral:** O dinheiro mantido em caixa (100% da carteira quando ${w}_{t}=0$, ou o valor do colateral quando vendido na opção) deve ser rentabilizado diariamente pela taxa da *T-bill* de 1 mês.

* **Lógica de Saída/Vencimento:**

  * Se a opção expirar "fora do dinheiro" (preço do S\&P 500 \> strike): O prêmio recebido é retido integralmente como lucro.

  * Se a opção expirar "dentro do dinheiro" (preço do S\&P 500 \< strike): Ocorre o exercício. É preciso calcular o prejuízo financeiro com base na diferença entre o preço do ativo subjacente e o *strike*, amortecido pelo prêmio recebido na entrada.

### **Passo 5: Avaliação de Desempenho e Geração de Métricas**

* Ao final de todo o período da amostra, é necessário calcular e exportar as seguintes métricas de performance da curva de capital da estratégia dinâmica para comparar com o índice passivo (PUT Index):

  * **Índice de Sharpe:** Mede o retorno excedente por unidade de risco total (desvio padrão).  
      
  * **Índice de Sortino:** Substitui o desvio padrão da fórmula do Sharpe apenas pelo desvio das perdas (abaixo do retorno de referência), sendo ideal para o *payoff* assimétrico da venda de *puts*.  
      
  * **Maximum Drawdown:** Mede a maior queda acumulada do topo ao fundo da curva de capital, para avaliar a proteção contra eventos de cauda (risco de perdas concentradas).