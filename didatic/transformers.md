### Etapa 1: **Tokenização:**
O texto de entrada é dividido em tokens via Byte Pair Encoding (cria "pedacinhos" de palavras). Cada token recebe um ID inteiro; esse ID é apenas um rótulo e não carrega significado.

### Etapa 2: **Embedding:**
 Cada ID é mapeado para um vetor denso, e esse vetor é uma linha na matriz de embeddings $E$. O tamanho do vetor é representado por $d_{model}$ (e, por consequência, a quantidade de colunas da matriz) e é um hiperparâmetro (usado 512 no paper). A operação equivale a multiplicar um vetor one-hot por $E$ e é implementada como um lookup (ao invés de multiplicar diretamente, basta olhar a linha do ID correspondente). A matriz de embeddings $E$ é inicializada aleatoriamente, e o modelo usa esses valores para produzir previsões. A função de loss (cross-entropy) mede o erro da previsão e, por backpropagation, calcula os gradientes e atualiza os parâmetros. A cada passo, só as linhas dos tokens presentes no batch são atualizadas. Ao fim do treino, tokens usados em contextos semelhantes ficam com vetores próximos no espaço. Esse vetor é fixo e independente do contexto. No paper, $E$ é compartilhada entre os embeddings do encoder, do decoder e a camada linear de saída.

### Etapa 3: **Escala:** 
O embedding é multiplicado por $\sqrt{d_{model}}$ (paper usa $\approx 22{,}6$). Como $E$ é inicializada com valores pequenos (norma $\approx 1$) e o positional encoding tem norma $\sqrt{d_{model}/2} = 16$, a escala coloca as duas informações na mesma grandeza, evitando que a posição da palavra no texto domine o significado.

### Etapa 4: **Positional Encoding (PE):** 
Como a self-attention é invariante à ordem dos tokens, a posição é injetada explicitamente (somando a cada vetor um encoding senoidal) 


$$PE_{(pos,\,2i)} = \sin\!\left(\tfrac{pos}{10000^{2i/d_{model}}}\right), \quad PE_{(pos,\,2i+1)} = \cos\!\left(\tfrac{pos}{10000^{2i/d_{model}}}\right)$$

Cada par de dimensões oscila numa frequência diferente, das rápidas às lentas, como os ponteiros de um relógio, o que dá a cada posição uma assinatura única. A matriz PE é calculada uma vez, não tem parâmetros treináveis e é somada elemento a elemento (a linha $pos$ ao token na posição $pos$), igual para todas as frases do batch. Propriedades: generaliza para sequências mais longas que as do treino, e a posição relativa vira uma transformação linear ($PE_{pos+k}$ é uma rotação de $PE_{pos}$ que depende só de $k$).

### Etapa 5: **Dropout:**
 Durante o treino, cada elemento da matriz tem probabilidade 0.1 de ser zerado (e os demais são divididos por 0.9, o que preserva o valor esperado). O dropout não altera os parâmetros de $E$, apenas a cópia que passa pela rede naquele forward. Isso funciona como regularização, forçando representações redundantes. Zerar elementos aleatoriamente força a "espalhar" a informação, e isso reduz o overfitting.

Resultado: Cada linha combina “qual token é” e “onde ele está”, e $X$ (A matriz resultante após todos os processos) é a entrada da primeira camada do encoder. O decoder aplica o mesmo processo à sequência de saída deslocada (iniciada por `<bos>`).

===========================================================================================================================================

### Etapa 6: **Encoder:**
 O encoder é uma pilha de $N$ camadas (paper usa $N = 6$) com pesos independentes. Uma camada (layer) é um conjunto de operações matemáticas organizadas. Cada camada recebe uma matriz $n \times d_{model}$ (mesmo tamanho da matriz $X$) e devolve outra de mesma forma. Cada camada contém duas subcamadas, com os seguintes processos:

#### **(a) Multi-head self-attention**

- **Uma attention head é um mecanismo que aprende a identificar relações entre os tokens de uma sequência. Cada head pode aprender a prestar atenção em diferentes tipos de relações (muitas vezes a relação aprendida não é específica e bem definida);**
- **No Transformer, cada token possui sua própria Query, Key e Value em cada head de cada camada**

**O mecanismo de atenção permite que cada token considere informações dos outros tokens da sequência. Isso é importante porque o significado de uma palavra geralmente depende do contexto**. Para isso, cada head utiliza três matrizes de pesos aprendidas $(W^Q), (W^K) e (W^V)$, que transformam a matriz de entrada da camada (na primeira camada, $X$) em três representações: $Query (Q)$, $Key (K)$ e $Value (V)$, de tamanho $n \times 64$. Por enquanto, 1 matriz entra e 3 matrizes são produzidas. A matriz $Q$ representa o que um token procura nos demais, a $K$ representa as características pelas quais um token pode ser identificado como relevante e $V$ contém a informação que será utilizada na combinação final.

O mecanismo calcula a compatibilidade entre as Queries e as Keys por meio de um produto escalar, dividido por $\sqrt{d_k}$ (no paper, $\sqrt{64} = 8$) para evitar que o Softmax sature: $(S = Q \cdot K^T / \sqrt{d_k})$. Em seguida, aplica a função Softmax em cada linha para transformar esses valores em pesos de atenção ($A$), que indicam o quanto cada token deve considerar cada um dos tokens da sequência (cada linha soma 1). Esses pesos são utilizados para calcular uma média ponderada dos Values ($A \cdot V$), produzindo a saída de atenção para cada token (depois desse processo, as 3 matrizes voltam a ser apenas uma). 

- É interessante pensar que **cada layer incorpora mais informação contextual**, gerando uma nova representação no espaço (todas as camadas enxergam a sequência inteira). **O que muda ao longo da pilha é o nível de abstração**: cada camada transforma mais as representações, permitindo compor relações mais elaboradas. Empiricamente, **camadas iniciais tendem a capturar mais informação sintática/superficial e camadas finais mais informação semântica, mas sem uma divisão explícita de responsabilidades**.

No Multi-Head Attention, esse processo acontece em paralelo por meio de attention heads (no paper, $h=8$). Cada head possui suas próprias matrizes de pesos e pode aprender diferentes relações entre os tokens. No paper, cada head trabalha com vetores de 64 dimensões.

Finalmente, as saídas das oito heads são concatenadas, formando novamente uma matriz de dimensão $n \times 512$ (no caso do paper, $64 \times 8 = 512$). Essa matriz passa por uma última transformação utilizando a matriz de pesos $W^O$, que combina as informações produzidas pelas diferentes heads.

No Encoder, utiliza-se uma máscara de padding para impedir que os tokens direcionem atenção às posições artificiais (`<pad>`) utilizadas para completar sequências de tamanhos diferentes, atribuindo $-\infty$ aos scores dessas posições antes do Softmax.

------------------------ CONTINUAR AQUI---------------------------------------------------------


#### **(b) Feed-Forward Network (FFN):**
Uma rede de duas camadas, $\max(0, zW_1 + b_1)W_2 + b_2$ ($512 \to 2048 \to 512$), aplicada a cada token de forma independente, com os mesmos pesos para todas as posições. A atenção é o único ponto em que os tokens trocam informação; a FFN processa de forma não linear a informação que cada token recebeu. Ela concentra cerca de 2/3 dos parâmetros da camada.

Add & Norm. A saída de cada subcamada passa por dropout (0,1), é somada à entrada da subcamada (conexão residual) e normalizada por LayerNorm: $\text{LN}(x + \text{Dropout}(\text{Sublayer}(x)))$. O residual faz a subcamada aprender apenas um ajuste sobre a entrada, preservando a identidade do token, e garante um caminho direto para o gradiente. Por isso, toda subcamada devolve $d_{model}$ dimensões. A LayerNorm normaliza cada token ao longo das features (média 0, variância 1, seguida de $\gamma$ e $\beta$ aprendidos), estabilizando as ativações ao longo das camadas.

Resultado: a saída da 6ª camada é $H \in \mathbb{R}^{n \times 512}$, um vetor contextualizado por token da frase de entrada. $H$ é usada como keys e values da cross-attention em todas as camadas do decoder.