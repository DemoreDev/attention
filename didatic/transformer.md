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

## Decoder
### Etapa 0: **Conhecimento prévio (pré-decoder):**
No transformer original, Encoder e Decoder compartilham a matriz de embeddings $E$, que associa os IDs dos tokens aos vetores. Nesse caso, utiliza-se um vocabulário conjunto (a matriz $E$ possui tokens inglês e alemão). Mantendo o exemplo tradutor, no Decoder a entrada é a **tradução esperada**. Por exemplo, caso a entrada do encoder fosse "The cat is sleeping", a entrada do decoder seria "Die Katze schläft". O Decoder é autoregressivo, ou seja, **seu objetivo é prever o próximo token da tradução considerando os tokens anteriores e as informações produzidas pelo Encoder (matriz $H$)**. Esse objetivo é ilustrado pela equação abaixo: 

$$P(y_t\mid y_{<t},H)$$

### Etapa 1: **Tokenização (pré-decoder):**
A tokenização acontece da mesma forma do encoder (texto quebrado em palavras ou "pedaços" de palavras). Seguindo a frase de exemplo, a tokenização resultaria em [`"Die", "Katze", "schläft"`]. A diferença é que, após a tokenização, são inseridos tokens especiais que marcam o início (`<bos>`) e o fim (`<eos>`) da frase. O resultado então passa a ser [`"<bos>", "Die", "Katze", "schläft", "<eos>"`]. Antes de fornecer essa entrada, é necessário entender os mecanismos de **Teacher Forcing e Deslocamento (Shifted Right):**

#### **Teacher Forcing:**
O Teacher Forcing consiste em fornecer ao Decoder os tokens **anteriores corretos** ao invés de usar os tokens que o próprio modelo previu. Isso permite que, caso o modelo erre alguma previsão (comum no começo pois os parâmetros não estão ajustados), esse erro não comprometa os contextos das previsões seguintes. Isso não impede que a loss e o Backpropagation atualizem os parâmetros.

#### **Deslocamento:**
Para implementar o Teacher Forcing, utiliza-se um procedimento chamado Shifted Right. A sequência completa da tradução é separada em duas sequências: a entrada do Decoder, que começa com `<bos>` e não inclui `<eos>`, e o target, que não inclui `<bos>` e termina com `<eos>`.

Seguindo o exemplo, a entrada [`"<bos>", "Die", "Katze", "schläft", "<eos>"`] seria separada na entrada real [`"<bos>", "Die", "Katze", "schläft"`] e no target [`"Die", "Katze", "schläft", "<eos>"`]. Cada posição da entrada é utilizada para prever o token correspondente do target. Assim, `<bos>` deve prever `Die`, `Die` deve prever `Katze`, e assim sucessivamente. A entrada real passa pelos processos seguintes normalmente, enquanto o target é usado apenas para calcular a loss (portanto não sofre embedding, escala, PE nem dropout).

**Um ponto de atenção:** durante o treinamento, é fornecida toda a sequência de entrada ao Decoder. Portanto, os tokens futuros já estão presentes na matriz. Para evitar que ocorra leakage, a primeira subcamada do Decoder aplica uma máscara causal (causal mask), que impede cada posição de prestar atenção em tokens posteriores. Assim, cada posição pode considerar apenas o token atual e os anteriores. Isso permite que todas as posições sejam processadas **em paralelo** durante o treinamento (grande vantagem computacional em relação aos modelos recorrentes tradicionais).

**Ideia principal:** o Decoder aprende a gerar uma sequência token por token, mas, graças ao Teacher Forcing e à máscara causal, consegue aprender todas essas previsões em paralelo durante o treinamento.

### Etapa 2: **Embedding, Escala, Positional Encoding e Dropout:**
Assim como no Encoder, todas essas etapas são aplicadas no Decoder também, antes de começar as layers. Apenas recapitulando:
- O embedding mapeia os tokens da entrada para seus vetores, gerando a matriz $X_{decoder}$; 
- A escala multiplica a saída do embedding (a matriz $X_{decoder}$) por $\sqrt{d_{model}}$, para manter as duas informações (matriz e Positional Encoding) na mesma ordem de grandeza;
- O Positional Encoding introduz explicitamente a posição de cada token no seu vetor (porque a posição de um token importa para o resultado).
- Por último, o Dropout, etapa onde cada elemento da matriz tem probabilidade 0.1 de ser zerado. Isso funciona como regularização (força representações redundantes). Costuma reduzir o overfitting.

### Etapa 3: **Decoder Layers:**
O Decoder é uma pilha de $N$ camadas ($N = 6$ no paper, assim como o Encoder). Cada camada recebe uma matriz $m \times d_{model}$ (onde $m$ é a quantidade de tokens da entrada real, após shifting) e devolve outra de mesma forma. Cada camada possui três subcamadas (uma a mais que o Encoder, a Cross Attention), e cada subcamada é seguida por Add&Norm (dropout, conexão residual e LayerNorm, como no encoder). A ilustração abaixo representa o fluxo de uma camada:

        ┌───────────────────────────────────────── layer N ────────────────────────────────────────────────┐
    ┌──►│  [masked attention] ──► [add&norm] ──► [cross attention] ──► [add&norm] ──►[FFN] ──► [add&norm]  │───┐
    │   └──────────────────────────────────────────────────────────────────────────────────────────────────┘   │
    │                                                                                                          │
    └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘

### PRIMEIRA SUBCAMADA: **Masked Self-attention:** 
**A máscara causal é o mecanismo que permite treinar o Decoder em paralelo enquanto preserva a mesma restrição de informação da fase de inferência.**
O mecanismo de atenção é praticamente idêntico ao Self-Attention do Encoder. A principal diferença é a utilização de uma máscara causal, que impede cada posição de consultar tokens futuros. Essa restrição é necessária porque o Decoder é autoregressivo, ou seja, aprende a prever o próximo token com base nos anteriores.

A entrada dessa subcamada é a matriz $X_{decoder}\in\mathbb{R}^{m \times 512}$, produzida após Embedding, Escala, Positional Encoding e Dropout. 

O mecanismo de atenção funciona da mesma forma que no Encoder: cada head utiliza três matrizes de pesos aprendidas ($W_Q$, $W_K$ e $W_V$) para transformar a entrada em Queries, Keys e Values. No paper, cada matriz possui dimensão $512 \times 64$, produzindo três representações de dimensão $m \times 64$. Os scores de atenção são calculados pelo produto entre Queries e Keys, dividido por $\sqrt{d_k}$: $C=\frac{QK^T}{\sqrt{d_k}}$. O resultado é uma matriz $C\in\mathbb{R}^{m\times m}$, que representa a compatibilidade entre cada Query e cada Key da sequência. A principal diferença aparece agora, antes da aplicação do Softmax: no Decoder, adicionamos uma matriz de máscara causal (M), que atribui $-\infty$ aos scores correspondentes às posições futuras ($j > i$) e mantém os demais inalterados, resultando em uma matriz diagonal. 

Como $e^{-\infty}=0$, os scores mascarados recebem peso zero após o Softmax. Dessa maneira, cada posição pode considerar apenas a si própria e as posições anteriores, sem acessar informações dos tokens futuros. Após o Softmax, a matriz de pesos de atenção é multiplicada pelos Values: $O = S \cdot V$. O resultado é uma matriz $ O\in\mathbb{R}^{m\times64}$, contendo uma nova representação contextualizada para cada posição. Diferentemente do Encoder, essa representação incorpora apenas informações do token atual e dos anteriores.

Esse processo acontece paralelamente em cada attention head (no paper, $h = 8$), cada uma com seus próprios parâmetros. A mesma máscara causal é aplicada em todas as heads. Cada head produz uma matriz $m \times 64$, e as oito matrizes são concatenadas, produzindo uma representação $m \times 512$. Finalmente, essa matriz passa por uma projeção aprendida $W^O\in\mathbb{R}^{512\times512}$, que combina as informações produzidas pelas diferentes heads. O resultado é a matriz $A\in\mathbb{R}^{m\times512}$, que será enviada ao primeiro Add&Norm e depois para a próxima subcamada.

**Ideia principal:** enquanto a Self-Attention do Encoder permite que cada token considere informações de toda a sequência, a Masked Self-Attention do Decoder restringe essa consulta aos tokens atuais e anteriores.

### Passo Intermediário: **Add & Norm:**
Antes da próxima subcamada ocorre Add&Norm, assim como no Encoder.
Assim que a matriz $A$ sai do mecanismo de masked self attention, é aplicado um Dropout (já visto antes). Após o Dropout é realizado Residual Connection: a matriz atual (pós Dropout) é somada com a matriz de entrada do mecanismo anterior (neste caso a matriz $X_{decoder}$). Por último, ocorre LayerNorm, onde são calculados a média e a variância de cada vetor e aplica-se normalização, e logo após o resultado é multiplicado e somado por dois parâmetros aprendidos ($\gamma$) e 
($\beta$), respeitando a equação $$LayerNorm(x)=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$$

### SEGUNDA SUBCAMADA: **Cross-attention:**
**A Cross-Attention é o mecanismo que permite ao Decoder consultar as representações da frase original (produzidas pelo Encoder)** para obter informações relevantes à geração dos próximos tokens da tradução.

Esse mecanismo é quase igual à Self-Attention, com uma diferença fundamental: as Queries são produzidas a partir do Decoder, enquanto as Keys e Values são produzidas a partir da matriz $H$ do Encoder. A entrada dessa subcamada é a matriz $Z\in\mathbb{R}^{m\times512}$, produzida após a Masked Self-Attention e seu Add&Norm. Além disso, a subcamada recebe a matriz $H\in\mathbb{R}^{n\times512}$, correspondente à saída final do Encoder. $m$ representa a quantidade de tokens da entrada do Decoder, enquanto $n$ representa a quantidade de tokens da frase original (entrada do encoder).

**Intuitivamente, o Decoder formula as consultas por meio das Queries, enquanto as Keys e Values do Encoder fornecem as características utilizadas para determinar a relevância dos tokens originais.** 

Assim como nos mecanismos anteriores, calculamos a compatibilidade entre Queries e Keys: $C=\frac{QK^T}{\sqrt{d_k}}$. Como Q possui $m$ linhas e K possui $n$ linhas, a matriz de scores possui dimensão $\boxed{C\in\mathbb{R}^{m\times n}}$. Cada linha representa uma posição do Decoder, enquanto cada coluna representa uma posição do Encoder. Dessa maneira, cada posição da tradução pode aprender quais posições da frase original são mais relevantes para produzir sua próxima previsão.

A Cross-Attention não utiliza máscara causal, pois a frase original já foi inteiramente processada pelo Encoder e está disponível para consulta. Entretanto, pode ser utilizada uma máscara de padding para impedir que o Decoder atribua atenção às posições artificiais (<pad>) da sequência original.

Após aplicar a máscara de padding, quando necessária, o Softmax é aplicado por linha, produzindo pesos que indicam a importância relativa de cada posição do Encoder para determinada Query do Decoder. Cada linha soma 1, considerando apenas as posições válidas.

Depois, a matriz de pesos é multiplicada pela matriz de Values: $\boxed{O = S \cdot V}$. Como $S\in\mathbb{R}^{m\times n}$ e $V\in\mathbb{R}^{n\times64}$, a saída possui dimensão $\boxed{O\in\mathbb{R}^{m\times64}}$. Essa operação produz, para cada posição do Decoder, uma representação construída a partir de uma combinação ponderada das informações do Encoder.

No paper, esse processo acontece em paralelo em oito attention heads ($h=8$), cada uma com suas próprias matrizes de projeção. Cada head produz uma matriz de dimensão (m\times64). As oito matrizes são concatenadas, formando uma representação $m\times512$, que passa pela projeção final $W_O\in\mathbb{R}^{512\times512}$. O resultado é $\boxed{A\in\mathbb{R}^{m\times512}}$

A mesma matriz $H$ é utilizada nas seis camadas do Decoder. Entretanto, cada camada possui suas próprias matrizes de pesos de Cross-Attention. Portanto, as diferentes camadas podem aprender maneiras distintas de consultar e combinar as informações da frase original. Além disso, as Queries mudam ao longo das camadas, pois são produzidas a partir das representações progressivamente transformadas pelo Decoder.

**A ideia principal é que a posição do Decoder consiga recuperar informações relevantes da frase original**, considerando o histórico da tradução que já foi processado pela Masked Self-Attention.

### Passo Intermediário: **Add & Norm:**
Antes da próxima subcamada ocorre Add&Norm, assim como no Encoder.

### TERCEIRA SUBCAMADA: **Feed-Forward Network (FFN):**
Essa subcamada é idêntica à do Encoder. Apenas revisando:
A FFN é uma pequena rede neural com duas camadas lineares ($W_1$ e $W_2$) e uma função de ativação não linear (ReLU) entre elas.

$$\boxed{\max(0, zW_1 + b_1)W_2 + b_2}$$

Revisando: o primeiro passo é a primeira camada linear $W_1$ (uma matriz $\mathbb{R}^{512\times2048}$) e a soma do resultado a um bias $b_1$; a ideia é aumentar o espaço para possibilitar combinações de características mais elaboradas. Logo após, passa pela função ReLU (transforma valores negativos em 0 e mantém os positivos); a ideia é introduzir não-linearidade (para permitir o aprendizado de relações mais complexas). Por último, o resultado passa pela segunda camada linear (uma matriz $\mathbb{R}^{2048\times512}$), que devolve à matriz seu tamanho original.

### Passo Intermediário: **Add & Norm:**
Antes da saída ocorre Add&Norm, assim como no Encoder.

### Etapa 4: **Saída (Linear + Softmax):**
Após passar pelas seis camadas do Decoder, obtemos uma matriz $D\in\mathbb{R}^{m\times d_{model}}$, onde $m$ é a quantidade de tokens da entrada do Decoder e $d_{model}=512$. Cada linha contém uma representação contextualizada de uma posição da tradução, construída a partir das informações dos tokens anteriores (Masked Self-Attention), da frase original (Cross-Attention) e das transformações não lineares realizadas pelas FFNs.

**O objetivo dessa etapa é transformar cada vetor de 512 dimensões em uma distribuição de probabilidades sobre todos os tokens do vocabulário, permitindo prever o próximo token da tradução.**

#### Primeira operação — Linear:
A matriz $D$ é multiplicada por uma matriz de projeção para o vocabulário (no paper, essa projeção é feita pela matriz de embeddings transposta $E^T$): $\boxed{L=D \cdot E^T}$. Como $D\in\mathbb{R}^{m\times512}$ e $E^T\in\mathbb{R}^{512\times|V|}$, obtemos $\boxed{L\in\mathbb{R}^{m\times|V|}}$. O resultado é uma matriz de logits, em que cada linha corresponde a uma posição do Decoder e cada coluna corresponde a um token do vocabulário. Esses logits são scores numéricos, não probabilidades. Cada logit é calculado por meio do produto escalar entre a representação da posição do Decoder e o embedding de um token candidato. Portanto, cada vetor produzido pelo Decoder é comparado com os embeddings de todos os tokens do vocabulário. Quanto maior o logit, maior será a probabilidade relativa atribuída ao token após o Softmax.

#### Segunda operação — Softmax:
O Softmax é aplicado individualmente a cada linha da matriz de logits:
$$\boxed{P_{ij}=\frac{e^{L_{ij}}}{\sum_{k=1}^{|V|}e^{L_{ik}}}}$$

O resultado é uma matriz $\boxed{P\in\mathbb{R}^{m\times|V|}}$, onde cada linha representa uma distribuição de probabilidades sobre o vocabulário, indicando a probabilidade atribuída a cada possível próximo token. As probabilidades são positivas e somam 1 em cada linha.

#### **Treinamento:**
Durante o treinamento, o Decoder recebe a tradução correta deslocada (Teacher Forcing), e a máscara causal impede cada posição de acessar os tokens futuros. Isso permite calcular as distribuições de probabilidades de todas as posições simultaneamente.

Essas distribuições são comparadas com os targets por meio da função de perda (Cross-Entropy, com label smoothing no paper original). A loss é utilizada pelo Backpropagation para calcular os gradientes e atualizar os parâmetros do Encoder e do Decoder, incluindo a matriz compartilhada de embeddings $E$.

#### **Inferência:**
Durante a inferência, não existe uma tradução esperada. O Decoder começa com `<bos>` e gera os tokens progressivamente, utilizando suas próprias previsões anteriores como entrada das próximas iterações. A cada iteração, a distribuição de probabilidades da última posição disponível é utilizada para selecionar o próximo token.

O Softmax não escolhe diretamente o token: ele produz as probabilidades. A escolha é realizada por uma estratégia de decodificação, como Greedy Decoding ou Beam Search (utilizada no paper original). Depois de selecionar um token, ele é adicionado à sequência, e o processo se repete até que o modelo produza `<eos>` ou alcance algum limite.