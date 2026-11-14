# Entendendo a arquitetura do Encoder-Decoder
## Encoder

### Etapa 0: **Conhecimento prévio (pré-encoder):**
$E$ é uma matriz grande (de tamanho $|V|$ por $d_{model}$, onde $|V|$ é o total de tokens do vocabulário e $d_{model}$ é um hiperparâmetro). $E$ atua como um “dicionário”, e sua função é mapear cada token do vocabulário para seu respectivo vetor (cada token possui um ID de identificação). Esses vetores são inicializados aleatoriamente e, ao longo do treino, se tornam boas representações dos tokens no espaço ($E$ é um parâmetro do modelo e vai sendo ajustado).

Supondo que o modelo é um tradutor, a entrada seria o texto a ser traduzido (que precisa ser processado antes de entrar no encoder). A seguir são explicadas as etapas do "pré-processamento":

### Etapa 1: **Tokenização (pré-encoder):**
O texto é dividido em tokens (palavras ou “pedacinhos” de palavras) por algum algoritmo, e.g. BPE, e o resultado são $n$ tokens. O tokenizador está ligado ao "dicionário" (usa o mesmo vocabulário de $E$), então todo token produzido tem uma linha correspondente na matriz.

### Etapa 2: **Embedding:**
Cada um desses $n$ tokens recebe seu ID (igual ao da matriz $E$) e é mapeado para seu respectivo vetor. Assim se forma uma matriz menor (a matriz $M$, de tamanho $n \times d_{model}$). Esse mapeamento equivale a multiplicar um vetor one-hot por $E$ e é implementado como um lookup (ao invés de multiplicar diretamente, basta olhar a linha do ID do token na matriz $E$). Após fazer isso para cada token, a matriz $M$ está montada e pronta para a próxima etapa.

### Etapa 3: **Escala:** 
A saída do embedding (matriz $M$) é multiplicada por $\sqrt{d_{model}}$ (paper usa $\approx 22{,}6$). Como $E$ é inicializada com valores pequenos (norma $\approx 1$) e o positional encoding (explicado a seguir) tem norma $\sqrt{d_{model}/2} = 16$, a escala coloca as duas informações na mesma grandeza, evitando que a posição da palavra no texto domine o significado. Os motivos exatos do escalonamento não são apresentados no paper, mas essa é a resposta mais intuitiva.

### Etapa 4: **Positional Encoding (PE):** 
Como a self-attention sozinha não incorpora informações sobre a posição dos tokens, utiliza-se o positional encoding para introduzir explicitamente essa informação (somando a cada vetor um encoding senoidal). Isso acontece porque a posição de um token importa para o resultado.


$$PE_{(pos,\,2i)} = \sin\!\left(\tfrac{pos}{10000^{2i/d_{model}}}\right), \quad PE_{(pos,\,2i+1)} = \cos\!\left(\tfrac{pos}{10000^{2i/d_{model}}}\right)$$

Cada par de dimensões oscila numa frequência diferente, das rápidas às lentas, como os ponteiros de um relógio, o que dá a cada posição uma assinatura única. A matriz PE é calculada uma vez, não tem parâmetros treináveis e é somada elemento a elemento (a linha $pos$ ao token na posição $pos$), igual para todas as frases do batch. Propriedades: generaliza para sequências mais longas que as do treino, e a posição relativa vira uma transformação linear       
($PE_{pos+k}$ é uma rotação de $PE_{pos}$ que depende só de $k$). Isso pode ajudar o modelo a generalizar para tamanhos de frases maiores que o treinamento.

### Etapa 5: **Dropout:**
 Durante o treino, cada elemento da matriz tem probabilidade 0.1 de ser zerado (e os demais são divididos por 0.9, o que preserva o valor esperado). O dropout não altera os parâmetros de $E$, apenas a cópia que passa pela rede naquele forward. Isso funciona como regularização, forçando representações redundantes. Zerar elementos aleatoriamente força a "espalhar" a informação, e isso reduz o overfitting.

Resultado: Cada linha combina “qual token é” e “onde ele está”, e $X$ (A matriz resultante após os processos descritos) é a entrada do encoder.

## Etapa 6: **Decoder Layers:**
 A parte principal do encoder é uma pilha de $N$ camadas (paper usa $N = 6$) com pesos (parâmetros) independentes. Uma camada (layer) é um conjunto de operações matemáticas organizadas. Cada camada recebe uma matriz $n \times d_{model}$ (a matriz $X$) e devolve outra de mesma forma (a matriz $H$). Cada camada contém duas subcamadas, e logo após cada subcamada ocorre um processo intermediário. A ilustração abaixo representa o fluxo dentro de uma camada:

        ┌─────────────────────── layer N ───────────────────────┐
    ┌──►│  [attention] ──► [add&norm] ──► [FFN] ──► [add&norm]  │───┐
    │   └───────────────────────────────────────────────────────┘   │
    │                                                               │
    └───────────────────────────────────────────────────────────────┘   

### **PRIMEIRA SUBCAMADA: Multi-head self-attention**

- **Uma attention head é um mecanismo que aprende a identificar relações entre os tokens de uma sequência. Cada head pode aprender a prestar atenção em diferentes tipos de relações (muitas vezes a relação aprendida não é específica e bem definida);**
- **No Transformer, cada token possui sua própria Query, Key e Value em cada head de cada camada**

**O mecanismo de atenção permite que cada token considere informações dos outros tokens da sequência. Isso é importante porque o significado de uma palavra geralmente depende do contexto**. Para isso, cada head utiliza três matrizes de pesos aprendidas $(W^Q), (W^K) e (W^V)$, que transformam a matriz de entrada da camada (na primeira camada, $X$) em três representações: $Query (Q)$, $Key (K)$ e $Value (V)$, de tamanho $n \times 64$. Por enquanto, 1 matriz de entrada gera 3 matrizes intermediárias. A matriz $Q$ representa o que um token procura nos demais, a $K$ representa as características pelas quais um token pode ser identificado como relevante e $V$ contém a informação que será utilizada na combinação final.

O mecanismo calcula a compatibilidade entre as Queries e as Keys por meio de um produto escalar, dividido por $\sqrt{d_k}$ (no paper, $\sqrt{64} = 8$) para evitar que o Softmax sature: $(C = Q \cdot K^T / \sqrt{d_k})$. Em seguida, aplica a função Softmax em cada linha para transformar esses valores em pesos de atenção ($S = Softmax(C)$), que indicam o quanto cada token deve considerar cada um dos tokens da sequência (a softmax transforma números discretos em probabilidades, e cada linha soma 1). Esses pesos são utilizados para calcular uma média ponderada dos Values ($S \cdot V$), produzindo a saída de atenção para cada token (depois desse processo, as 3 matrizes intermediárias voltam a ser apenas umam, de tamanho $n \times 64$). 

No Multi-Head Attention, esse processo acontece em paralelo por meio de attention heads (no paper, $h=8$). Cada head possui suas próprias matrizes de pesos e pode aprender diferentes relações entre os tokens. 

Finalmente, as saídas das oito heads são concatenadas, formando novamente uma matriz de dimensão $n \times 512$ (no caso do paper, $64 \times 8 = 512$). Essa matriz passa por uma última transformação utilizando a matriz de pesos $W^O$, que combina as informações produzidas pelas diferentes heads.

No Encoder, utiliza-se uma máscara de padding para impedir que os tokens direcionem atenção às posições artificiais (`<pad>`) utilizadas para completar sequências de tamanhos diferentes.

### **PROCESSO INTERMEDIÁRIO: ADD & NORM:**
Antes da próxima subcamada ocorre um processo, que é realizado em 3 passos: Dropout, Residual Connection (ADD) e LayerNorm (NORM).
Assim que a matriz sai do mecanismo de attention (vou chamar de matriz $A$), é aplicado um dropout (já visto antes), com o mesmo motivo (a aplicação do dropout ajuda a reduzir a dependência excessiva de determinadas ativações e pode diminuir o overfitting), e mesma taxa (paper usa $P = 0.1$). Após esse dropout a matriz atual é somada com a matriz de entrada do mecanismo anterior (neste caso a matriz $X$ do mecanismo de atenção, mas quando esse processo ocorrer após FFN, será a matriz $Z$ de entrada do FFN); esse passo é a Residual Connection. Por último, são calculados a média e a variância de cada vetor (cada token) e aplica-se normalização, e logo após o resultado é multiplicado e somado por dois parâmetros aprendidos ($\gamma$) e 
($\beta$), respeitando a equação $$LayerNorm(x)=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$$
Esse último passo é a LayerNorm, e a ideia é manter as ativações em uma escala mais controlada, contribuindo para a estabilidade do treinamento. As fórmulas a seguir ilustram este processo intermediário:

Add&Norm após Attention: $\boxed{Z = LayerNorm(X + Dropout(A))}$

Add&Norm após FFN: $\boxed{H = LayerNorm(Z + Dropout(F))}$

Onde $X$ é a matriz de entrada do Encoder, $A$ é a matriz de saída de Attention e entrada do primeiro Add&Norm, $Z$ é a saída do primeiro Add&Norm e a de entrada de FFN, $F$ é a saída de FFN e entrada do segundo Add&Norm e H é a saída do segundo Add&Norm e entrada da próxima layer.


### **SEGUNDA SUBCAMADA: Feed-Forward Network (FFN):**
Se a self-attention permite que os tokens troquem informações entre si, **a FFN pega as informações que cada token já recebeu e as transforma individualmente**. No Transformer, a FFN é uma pequena rede neural com duas camadas lineares ($W_1$ e $W_2$) e uma função de ativação não linear (ReLU) entre elas.

$$\boxed{\max(0, zW_1 + b_1)W_2 + b_2}$$

Entendendo a função FFN por partes: $z$ é a entrada e representa o vetor de um token. O primeiro passo é passar pela primeira camada linear $W_1$ (uma matriz $\mathbb{R}^{512\times2048}$) e somar o resultado a um bias $b_1$; a ideia é aumentar o espaço para possibilitar combinações de características mais elaboradas. Logo após, passa pela função ReLU (transforma valores negativos em 0 e mantém os positivos); a ideia é introduzir não-linearidade (para permitir o aprendizado de relações mais complexas). Por último, o resultado passa pela segunda camada linear (uma matriz $\mathbb{R}^{2048\times512}$), que devolve à matriz seu tamanho original (necessário para servir de entrada para Add&Norm e a próxima layer). Em resumo, **a primeira camada expande a representação, a ReLU introduz não linearidade e a segunda camada transforma a representação intermediária de volta para $d_{model}$ dimensões**. Os parâmetros $W_1$ e $W_2$, e $b_1$ e $b_2$ são aprendidos.

Após essa segunda subcamada ocorre outro Add&Norm, e então a primeira layer termina e a saída deste Add&Norm serve de entrada para a próxima layer, onde o processo se repete. Na última layer (e.g., a sexta layer), a saída é uma matriz $n \times 512$, chamada matriz $H$, que será usada no decoder no mecanismo de **Cross-Attention**.

======================= CONTINUAR AQUI =======================================================

## Decoder
### Etapa 1: **Tokenização:**

Etapa D0: Entrada do decoder

O decoder recebe a sequência de saída (a tradução) deslocada uma posição para a direita: um token especial <bos> é inserido no início e o último token é removido. O alvo é a tradução seguida de <eos>. Assim, na posição $t$, o decoder recebe os tokens até $t$ e deve prever o token $t+1$ da tradução. Ex.: entrada <bos> die Katze trinkt Milch, alvo die Katze trinkt Milch <eos>.

Etapa D1: Pré-processamento

A entrada passa pelas mesmas etapas do encoder: tokenização (mesmo vocabulário), embedding (mesma matriz $E$), escala por $\sqrt{d_{model}}$, positional encoding (posições contadas a partir de 0 na tradução) e dropout. O resultado é a matriz $Y$, de tamanho $m \times d_{model}$, onde $m$ é o número de tokens da tradução (não precisa ser igual a $n$).

Camada do decoder

O decoder é uma pilha de $N$ camadas ($N = 6$ no paper) com pesos independentes. Cada camada recebe uma matriz $m \times d_{model}$ e devolve outra de mesma forma. Ela tem três subcamadas, cada uma seguida de Add & Norm (dropout, conexão residual e LayerNorm, como no encoder).

PRIMEIRA SUBCAMADA: Self-attention mascarada. Igual à self-attention do encoder (Q, K e V vêm da mesma matriz, com 8 heads e $W^O$), mas com uma máscara causal: os scores das posições futuras ($j > i$) recebem $-\infty$ antes do softmax, o que zera seus pesos. Assim, cada posição só considera os tokens anteriores e ela mesma. A máscara é necessária porque, no treino, a tradução inteira entra de uma vez: sem ela, o modelo poderia “colar” o próximo token, que está presente na entrada deslocada.

SEGUNDA SUBCAMADA: Cross-attention. É o ponto em que o decoder consulta a frase de origem. As queries vêm do decoder (saída da subcamada anterior, $m \times d_{model}$) e as keys e values vêm de $H$, a saída do encoder ($n \times d_{model}$). A matriz de scores tem tamanho $m \times n$: cada posição da tradução decide quais tokens da frase de origem são relevantes para gerar o próximo token (um alinhamento suave). Não há máscara causal, apenas a de padding da origem. A mesma $H$ é usada nas $N$ camadas do decoder, mas cada camada tem seus próprios pesos de cross-attention.

TERCEIRA SUBCAMADA: Feed-forward. Idêntica à do encoder: $\max(0, zW_1 + b_1)W_2 + b_2$, aplicada a cada posição de forma independente.

Etapa D2: Camada de saída (Linear + Softmax)

A saída da última camada, $D$ ($m \times d_{model}$), é multiplicada por $E^\top$ (a mesma matriz de embedding, transposta), gerando os logits ($m \times |V|$): cada vetor é comparado por produto escalar com o vetor de todos os tokens do vocabulário. O softmax, aplicado por linha, transforma os logits em uma distribuição de probabilidade sobre o vocabulário para cada posição.

Treino

No treino, a tradução correta é conhecida e entra inteira no decoder (teacher forcing). Graças à máscara causal, as $m$ previsões são feitas em paralelo numa única passada. A loss é a cross-entropy entre a distribuição prevista em cada posição e o token alvo, com média sobre as posições. O paper usa label smoothing (0,1), que evita que o modelo fique confiante demais.

Inferência

Na inferência, a tradução não existe ainda. O encoder roda uma única vez e gera $H$. O decoder começa com <bos>, prevê o próximo token a partir da última linha de $P$, acrescenta esse token à sua entrada e repete o processo, até gerar <eos> ou atingir um comprimento máximo. Em vez de escolher sempre o token mais provável (greedy), o paper usa beam search (largura 4), que mantém as sequências parciais mais prováveis a cada passo.

Atualização dos parâmetros

Ao fim de cada passo de treino (forward + backward), só os parâmetros são atualizados: $E$, os pesos de atenção ($W^Q, W^K, W^V, W^O$ de cada head e camada, incluindo a cross-attention), os $\gamma$ e $\beta$ das LayerNorms e os pesos das FFNs. As matrizes intermediárias ($X$, $Q$, $K$, $V$, $H$, $D$ etc.) são recalculadas a cada forward e servem apenas de caminho para o gradiente. O encoder não tem loss própria: aprende pelo gradiente que chega através de $H$ na cross-attention. Como $E$ é usada em três lugares (embedding do encoder, embedding do decoder e camada de saída), seu gradiente é a soma dessas três contribuições: pelas entradas, só as linhas dos tokens presentes no batch recebem gradiente; pela saída, todas as linhas recebem (a do token correto é aproximada do vetor do decoder e as demais são afastadas). $E$ é atualizada uma única vez, com o gradiente total.