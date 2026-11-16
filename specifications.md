# Especificações do projeto

> Este arquivo é o espelho do projeto: toda decisão tomada (ou alterada) durante a implementação deve ser registrada aqui, com sua justificativa.

## Objetivo
Implementar a estrutura de um Transformer decoder-only, seu loop de treinamento e sua avaliação para converter datas em português para o padrão ISO `YYYY-MM-DD`. Fazer todo o modelo "à mão" e usar PyTorch pronto apenas quando a implementação for muito complicada / desnecessária / custosa (mal otimizada). A divisão exata está na seção [O que é implementado à mão](#o-que-é-implementado-à-mão).

Exemplo:
```text
Entrada: 25 de dezembro de 2023
Saída:   2023-12-25
```

Fluxo do trabalho:
1. Gerar um dataset supervisionado de datas, no formato adequado (com variações de formatação).
2. Separar os exemplos em treino, validação e teste **antes de formar tensores**.
3. Implementar um modelo decoder-only autoregressivo com as subcamadas **masked multi-head self-attention e FFN**, além do passo intermediário Add&Norm.
4. Implementar o treinamento com Cross-Entropy, backpropagation e AdamW (com warmup + decaimento cosseno do LR).
5. Treinar **três configurações de capacidade diferente**, usando as CPUs do servidor.
6. Avaliar os modelos em dados não vistos, principalmente com **exact-match accuracy**, e em um conjunto **OOD** (anos nunca vistos).
7. Registrar configuração, métricas e duração para comparação e reprodução.

**Escopo:** é um Transformer decoder-only, demonstrativo da arquitetura de modelos de linguagem autoregressivos. **Não há Encoder nem Cross-Attention.**

---

## Lista de Definições
| Item | Definição |
|---|---|
| Linguagem/framework | Python (3.12 local / 3.13 servidor) + PyTorch (CPU) |
| Arquitetura | Transformer decoder-only |
| Tarefa | `D(D) de mês de YYYY` → `YYYY-MM-DD` |
| Idioma da entrada | Português |
| Tokenizador | **Por caractere** (tokens especiais indivisíveis) |
| Formatação do dia | 2 dígitos (`01`, `13`); dias 1–9 aparecem com **1 dígito** em **20%** dos casos |
| Formatação do mês | Minúsculo (padrão do português); **inicial maiúscula** em **20%** dos casos |
| Formatação do ano | **4 dígitos** |
| Intervalo de datas | 1800–2049 (treino/val/teste) e 2050–2099 (teste OOD) |
| Vocabulário | Caracteres do split de treino + `<pad>`, `<bos>`, `<sep>`, `<eos>` |
| Entrada de treinamento | Prompt + resposta ISO, com shifting de 1 token |
| Loss | Cross-Entropy somente onde o **alvo** é um caractere ISO ou `<eos>`; demais posições `-100` |
| Attention | Máscara causal (nunca consultar posições futuras); sem máscara de padding (redundante, ver abaixo) |
| FFN | Duas lineares com ReLU; expansão $4 \cdot d_{model}$ (modelo do paper) |
| Add & Norm | Post-LayerNorm, como no paper original, em cada subcamada; LayerNorm implementada à mão |
| Posição | Encoding posicional senoidal (paper) + escala $\sqrt{d_{model}}$ dos embeddings |
| Saída | Projeção para o vocabulário **compartilhando pesos** com a matriz de embeddings $E$ (weight tying, como no paper) |
| Otimizador | AdamW, `betas=(0.9, 0.98)`, `eps=1e-9`, `weight_decay=0.1` (sem decay em bias, LayerNorm e embeddings) |
| LR | Pico `1e-3`, warmup linear de 300 passos, decaimento cosseno até `1e-4` |
| Gradient clipping | Norma global máxima `1.0` |
| Dropout | 0.1, nos mesmos lugares do paper |
| Label smoothing | Não usado |
| Divisão | 20.000 treino / 2.500 validação / 2.500 teste + 18.262 OOD |
| Principal métrica | Exact match da string ISO gerada |
| Ambiente | CPU, servidor 2 sockets × 10 núcleos físicos (40 CPUs lógicas, HT) |
| Experimentos | 3 capacidades, variando `d_model` e `n_layers`; 5 threads cada, em paralelo |

---

## Dataset: geração, unicidade e divisão
- Gerar **todas** as datas válidas de `1800-01-01` a `2049-12-31` (91.311 datas) e de `2050-01-01` a `2099-12-31` (18.262 datas).
- Selecionar **25.000 datas únicas, uniformemente, sem reposição**, do intervalo 1800–2049 (~100 por ano). Elas formam treino/validação/teste.
- Todas as 18.262 datas de 2050–2099 formam o **teste OOD** (anos nunca vistos no treino).
- Usar `datetime.date` e `timedelta` para garantir validade (inclusive anos bissextos).
- Alvo: `date.isoformat()`.
- Origem: `"{dia} de {mês} de {ano:04d}"`, onde:
  - `dia` é `{dia:02d}`; se o dia for 1–9, com probabilidade **0.2** é escrito com 1 dígito (`5` em vez de `05`).
  - `mês` é o nome em minúsculo (`janeiro, fevereiro, março, abril, maio, junho, julho, agosto, setembro, outubro, novembro, dezembro`); com probabilidade **0.2** recebe inicial maiúscula (`Dezembro`).
- **Cada data recebe um único formato**, sorteado uma vez e salvo no CSV. As mesmas taxas valem para treino, validação, teste e OOD.
- Variações raras (20%) foram escolhidas de propósito: há poucos exemplos (dia de 1 dígito ≈ 6% das datas), o que testa se o modelo aprende essas relações a partir de pouca evidência. A acurácia por variante é reportada separadamente.
- Reprodutibilidade: `seed=42`. A **seleção** das datas e o **sorteio de formatos** usam geradores separados (`seed` e `seed + 1`), de modo que mudar as taxas de formatação não altera quais datas caem em cada split.

Dias, meses e anos **vão se repetir**. O que deve ser único é a combinação completa de dia + mês + ano.

### Splits

| Split | Número | Intervalo | Uso |
|---|---:|---|---|
| Treino | 20.000 | 1800–2049 | Atualização dos pesos |
| Validação | 2.500 | 1800–2049 | Seleção de checkpoint e early stopping |
| Teste | 2.500 | 1800–2049 | **Avaliação final**, após congelar escolhas |
| Teste OOD | 18.262 | 2050–2099 | Generalização para anos nunca vistos |

As 25.000 datas sorteadas já saem em ordem aleatória (gerador determinístico) e são cortadas em treino/validação/teste. Os quatro splits são salvos em `data/train.csv`, `data/val.csv`, `data/test.csv` e `data/test_ood.csv` (colunas `source`, `target`, UTF-8) e **versionados no git**, para que **todos os experimentos usem exatamente os mesmos dados**. O script verifica que não há interseção entre os splits.

As variantes de formatação de cada exemplo (dia com 1 dígito, mês com maiúscula) não são colunas do CSV: elas são derivadas da própria string `source` na avaliação.

**Observação sobre o OOD:** os anos 2050–2099 têm, após o prefixo `20`, dezenas `5`–`9`, combinação que nunca aparece no treino (embora os dígitos `5`–`9` apareçam como dezena em 18xx/19xx). Um modelo que aprendeu a *copiar* o ano generaliza; um que memorizou padrões de ano, não.

---

## Tokenização e formatação autoregressiva

A tokenização ocorre por caractere. O vocabulário contém:
- Os tokens especiais `<pad>` (id 0), `<bos>`, `<sep>`, `<eos>`, que são indivisíveis.
- Os dígitos de 0 a 9, o espaço `" "` e o hífen `"-"`.
- As letras presentes nos nomes dos meses (minúsculas, `ç`, e as maiúsculas iniciais `J F M A S O N D`).

O vocabulário é construído a partir do **split de treino** (em ordem ordenada, para ser reproduzível) e salvo em `data/vocab.json`. Verifica-se que validação, teste e OOD não têm caracteres fora do vocabulário.

Exemplo:
```text
source: 25 de dezembro de 2023
target: 2023-12-25

full_tokens = [<bos>] + list(source) + [<sep>] + list(target) + [<eos>]
```

### Shift de um token (teacher forcing)

```python
input_ids = full_ids[:-1]
target_ids = full_ids[1:]
```

A entrada contém o contexto inteiro e a parte ISO deslocada. A máscara causal garante que, ao prever uma posição, o modelo não consulta a resposta futura.

Exemplo legível (tokens especiais representados como unidades únicas):

```text
full:    <bos>25 de dezembro de 2023<sep>2023-12-25<eos>
input:   <bos>25 de dezembro de 2023<sep>2023-12-25
target:       25 de dezembro de 2023<sep>2023-12-25<eos>
```

### Loss somente na resposta
O label de uma posição é `-100` (ignorado pela Cross-Entropy) sempre que o **alvo** dessa posição é um caractere da data por extenso ou o próprio `<sep>`. Só contribuem para a loss as posições cujo alvo é **um caractere da data ISO ou o `<eos>`** (11 posições por exemplo). Note que a previsão feita *na posição do `<sep>`* tem como alvo o primeiro caractere ISO, portanto **conta** na loss.

### Padding e limite de contexto
- Usar `collate_fn` para completar as sequências do batch até o maior comprimento do batch, com padding **à direita**.
- Usar `<pad>` na entrada e `-100` nos labels de posições preenchidas.
- **Sem máscara de padding na atenção.** Com padding à direita, todo `<pad>` fica *depois* dos tokens reais; a máscara causal já impede qualquer posição real de consultar posições posteriores, logo nunca consulta um `<pad>`. As posições de `<pad>` (queries) produzem saídas inúteis, mas seus labels são `-100`, então não afetam a loss. Isso é verificado por um teste.
- `max_seq_len=80` (o maior exemplo tem 36 tokens: `<bos>` + `"09 de fevereiro de 2023"` (23) + `<sep>` + 10 + `<eos>`; a entrada após o shift tem 35). Verificar programaticamente antes do treino.
- Não truncar exemplos de forma silenciosa; validar que todas as sequências cabem no contexto escolhido.

### Inferência (sem teacher forcing)

Fornecer somente o prompt:
```text
<bos>25 de dezembro de 2023<sep>
```

Gerar **um caractere por vez**, sempre usando os logits da última posição, acrescentando cada token gerado à sequência. Parar ao produzir `<eos>` ou ao atingir `max_new_tokens = 14` (10 caracteres ISO + `<eos>` + folga de 3 para diagnosticar saídas excedentes).

**Geração em batch:** os prompts têm tamanhos diferentes (20 a 25 tokens), então ficam com padding **à direita** (como no treino) e cada exemplo tem seu próprio cursor (comprimento atual). O token novo é escrito na posição do cursor, sobre um `<pad>`; pela máscara causal, nenhuma posição real enxerga os `<pad>` posteriores. Exemplos que já geraram `<eos>` param de avançar. Sem KV-cache: a sequência inteira é recalculada a cada passo (simples e suficiente para ≤ 39 tokens). Verificado: a geração em batch produz exatamente o mesmo resultado que a geração exemplo a exemplo.

Checkpoint (formato salvo pelo treino e lido por `generate.load_model`): dicionário com `model_config` (campos de `ModelConfig`) e `model_state` (`state_dict` do modelo).

Para as métricas, utilizar **greedy decoding** (`argmax`), determinístico. Comparar somente a string extraída após `<sep>` e antes de `<eos>` com o target exato. Qualquer resposta incompleta, excedente ou malformatada conta como incorreta.

---

## Arquitetura

### O que é implementado à mão

| Componente | Implementação |
|---|---|
| Multi-head masked self-attention (projeções Q/K/V, scores, máscara causal, softmax, concatenação, $W^O$) | **À mão** (sem `F.scaled_dot_product_attention` nem `nn.MultiheadAttention`) |
| Encoding posicional senoidal | **À mão** |
| FFN | **À mão** (composta de `nn.Linear` + ReLU) |
| LayerNorm | **À mão**, com operações de tensor do PyTorch (`mean`, `var`); parâmetros $\gamma$, $\beta$ aprendidos, `eps=1e-5`. Não pode usar numpy: sair do PyTorch quebraria o grafo de gradientes (autograd). Um teste compara com `nn.LayerNorm`. |
| Add & Norm, bloco decoder, modelo completo, geração greedy | **À mão** |
| Camadas lineares, lookup de embeddings, dropout | PyTorch (`nn.Linear`, `nn.Embedding`, `nn.Dropout`) |
| Cross-Entropy, AdamW | PyTorch (`F.cross_entropy`, `torch.optim.AdamW`) |

### Fluxo do modelo
```text
ids → E[ids] · √d_model → + PE → Dropout
    → N × [ masked MHA → Dropout → Add → LayerNorm → FFN → Dropout → Add → LayerNorm ]
    → logits = H · Eᵀ
```

- **Dropout (0.1), nos lugares do paper (seção 5.4):** na soma embeddings + PE e na saída de cada subcamada, antes da soma residual e da normalização. Não há dropout nos pesos de atenção nem dentro da FFN.
- **Weight tying:** a projeção final usa $E^T$ (sem bias), como no paper. Isso reduz parâmetros e faz cada logit ser o produto escalar entre a representação e o embedding do token candidato.
- **Inicialização:** $E \sim \mathcal{N}(0, d_{model}^{-1/2})$. O padrão do PyTorch ($\mathcal{N}(0,1)$), multiplicado por $\sqrt{d_{model}}$, faria os embeddings dominarem o PE (que tem valores em $[-1, 1]$); com essa inicialização, após a escala, cada elemento tem desvio ≈ 1, na mesma ordem do PE (é o que a implementação de referência do paper, tensor2tensor, faz). As camadas lineares usam a inicialização padrão do PyTorch.
- $d_k = d_v = d_{model} / h$.
- Projeções $W^Q, W^K, W^V, W^O$ **sem bias** (como nas fórmulas do paper); as $h$ matrizes de cada projeção ficam lado a lado em uma única `nn.Linear(d_model, d_model)`, matematicamente equivalente a $h$ matrizes separadas, porém calculando todas as heads de uma vez.
- **Observação (efeito do weight tying na inicialização):** como a saída é $H \cdot E^T$ e o $H$ ainda carrega o embedding do token de entrada (via conexões residuais), o modelo recém-inicializado tende a "prever o próprio token de entrada" (argmax igual à entrada em ~98–100% das posições). Por isso a loss inicial fica acima de $\ln|V| \approx 3{,}78$ (≈ 8 no Exp. 3). Não é um erro: é desaprendido nos primeiros passos do treino. Será verificado nas curvas de aprendizado.

### Número de parâmetros ($|V| = 44$)
| Exp. 1 | Exp. 2 | Exp. 3 |
|---:|---:|---:|
| 26.560 | 152.000 | 796.672 |

---

## Treinamento

- Loss: `F.cross_entropy(logits, labels, ignore_index=-100)` (média sobre os tokens válidos).
- AdamW: `betas=(0.9, 0.98)` e `eps=1e-9` (valores do paper), `weight_decay=0.1` aplicado só às matrizes das camadas lineares (não em bias, $\gamma$/$\beta$ da LayerNorm nem nos embeddings compartilhados).
- **Schedule de LR:** warmup linear de 0 a `1e-3` nos primeiros 300 passos (~1 época), depois decaimento cosseno até `1e-4` ao final de `max_epochs`. Justificativa: Post-LN é instável no início do treino sem warmup (o paper usa warmup pelo mesmo motivo; ver Xiong et al., 2020). O pico `1e-3` foi escolhido porque modelos pequenos toleram LR mais alto e `3e-4` tende a convergir devagar. O LR é atualizado a cada passo.
- Gradient clipping: norma global 1.0.
- Ordem dos batches de treino embaralhada a cada época com gerador seedado.
- `DataLoader` com `num_workers=0`: o carregamento roda no processo principal, pois workers extras disputariam os núcleos reservados ao cálculo.
- A `train_loss` é a média da época, medida com dropout ligado e com os pesos mudando ao longo da época; a `val_loss` usa os pesos do fim da época, sem dropout. Por isso a `val_loss` pode ficar abaixo da `train_loss`.
- Modo `--quick` (teste local): 2.000 exemplos de treino, 500 de validação e 3 épocas, só para verificar que o pipeline roda.
- Hiperparâmetros lidos de `configs/expN.json` (seções `model` e `train`); `vocab_size` vem do `vocab.json`.

### Critério de seleção
- Ao final de cada época, calcular `train_loss`, `val_loss` e o exact match de validação (geração greedy), também separado por variante de formatação: dia 1–9 com 1 dígito × 2 dígitos e mês maiúsculo × minúsculo. Objetivo: acompanhar como as variantes raras evoluem ao longo do treino. Ressalva: o grupo "dia com 1 dígito" tem só 150 exemplos na validação (cada exemplo vale 0,67 p.p.), então sua curva é ruidosa.
- Também por época: acurácia de validação de cada campo da data ISO (ano, mês, dia), para observar a **ordem em que o modelo aprende** cada parte da tarefa (copiar dia e ano × traduzir o nome do mês para o número).
- **Monitoramento OOD por época:** exact match e acurácia do ano numa **amostra fixa de 2.500 datas** do teste OOD (sorteada com `seed=42`, a mesma em todos os experimentos e épocas). É **apenas observação**: não participa do early stopping, da escolha do checkpoint nem de nenhum hiperparâmetro (o que seria vazamento de teste). A avaliação OOD final continua sendo feita nas 18.262 datas. Custo: cerca de dobra o tempo da validação de cada época.
- **Early stopping pela `val_loss`**, com paciência 5: parar após cinco épocas seguidas sem melhora estrita da menor `val_loss`. Justificativa: o exact match satura perto de 100% cedo e deixa de discriminar; a `val_loss` continua informativa.
- Guardar o checkpoint da época de menor `val_loss`; ao final, recarregá-lo e avaliar no teste e no OOD.

---

## Avaliação e métricas

Calculadas para validação (por época: só loss e exact match), teste e OOD (no final):

| Métrica | Definição |
|---|---|
| Loss | Cross-Entropy com teacher forcing (posições da resposta) |
| **Exact match** | Fração de exemplos cuja string gerada é idêntica ao target |
| Acurácia por caractere | Fração dos 10 caracteres ISO corretos na posição (saída curta conta como erro nas posições faltantes) |
| Acurácia por campo | Exact match separado de ano (`YYYY`), mês (`MM`) e dia (`DD`) |
| Exact match por variante | Dia em 3 grupos: "dia 1–9 com 1 dígito", "dia 1–9 com 2 dígitos" (comparação justa entre as duas formas dos mesmos dias) e "dia 10–31"; mês minúsculo × maiúsculo. Cada grupo traz o seu número de exemplos `n` |
| Taxa de saída válida | Fração das saídas no formato `YYYY-MM-DD` que são uma data existente (ex.: `2023-02-30` é inválida) |
| Exact match por mês | Um valor por mês (detectar meses problemáticos) |
| Exemplos de erro | Lista de previsões incorretas (`source`, `target`, `prediction`) |

---

## Configurações dos experimentos

Três modelos treinados **do zero**, com parâmetros independentes; os splits, seed, protocolo de treino e métricas são iguais. As configurações ficam em `configs/expN.json`.

| Configuração | Exp. 1 | Exp. 2 | Exp. 3 |
|---|---:|---:|---:|
| `d_model` | **32** | **64** | **128** |
| `n_layers` | **2** | **3** | **4** |
| `n_heads` | 4 | 4 | 4 |
| `d_ff` | 128 | 256 | 512 |
| `max_seq_len` | 80 | 80 | 80 |
| `dropout` | 0.1 | 0.1 | 0.1 |
| `batch_size` | 64 | 64 | 64 |
| `lr` (pico) | 1e-3 | 1e-3 | 1e-3 |
| `warmup_steps` | 300 | 300 | 300 |
| `min_lr` | 1e-4 | 1e-4 | 1e-4 |
| `weight_decay` | 0.1 | 0.1 | 0.1 |
| `grad_clip` | 1.0 | 1.0 | 1.0 |
| `max_epochs` | 50 | 50 | 50 |
| `early_stopping_patience` | 5 | 5 | 5 |
| `seed` | 42 | 42 | 42 |
| `device` | cpu | cpu | cpu |
| `threads` | **5** | **5** | **5** |
| Núcleos físicos (CPU affinity) | **0–4** | **5–9** | **10–14** |
| Nó NUMA (memória) | 0 | 0 | 1 |

**Interpretação:** busco comparar a capacidade, variando simultaneamente largura (`d_model`) e profundidade (`n_layers`), não isolando o efeito de cada fator. Registrar o número efetivo de parâmetros, o tempo de treinamento e o throughput (tokens/s) para uma comparação completa.

### Alocação de CPU
Topologia do servidor (`lscpu -e`): 2 sockets/nós NUMA, 10 núcleos físicos cada. CPUs lógicas 0–9 são os núcleos físicos do socket 0, 10–19 os do socket 1, e 20–39 são os irmãos de hyperthreading (20–29 → socket 0, 30–39 → socket 1).

- Os três experimentos rodam **em paralelo**, com **o mesmo número de threads (5)**, para que o tempo de treino seja comparável entre capacidades.
- Cada experimento fica inteiro num único socket, fixado com `taskset -c <núcleos>` (o `numactl` não está instalado no servidor). A memória fica local pela política *first-touch* do Linux (a página é alocada no nó NUMA da CPU que a usa primeiro), evitando acesso à memória do outro socket, que deixaria um experimento mais lento por motivo alheio à capacidade. O `config.json` de cada run registra a afinidade efetiva (`cpu_affinity`), confirmando a fixação.
- Os irmãos de hyperthreading (20–39) ficam ociosos, para que cada thread tenha um núcleo físico exclusivo.
- `torch.set_num_threads(5)` e `torch.set_num_interop_threads(1)`.
- Limitação conhecida: Exp. 1 e Exp. 2 dividem o cache L3 e a banda de memória do socket 0, enquanto o Exp. 3 está sozinho no socket 1. Com modelos deste tamanho, o efeito esperado é pequeno.
- O servidor é compartilhado, mas é considerado ocioso durante os experimentos.

---

## Fluxo de execução (servidor × máquina local)

- **Servidor:** apenas executa os scripts (`git pull` → `scripts/run_all.sh` → `git push`). Treino, avaliação de teste e avaliação OOD rodam lá.
  - `scripts/run_expN.sh`: treino + avaliação de um experimento, fixado nos seus núcleos com `taskset`.
  - `scripts/run_all.sh`: roda o `pytest` (se falhar, nada é treinado), lança os 3 experimentos em paralelo, espera todos e informa qual falhou. A saída de cada um vai para `runs/expN/run.log`.
  - Execução, dentro de uma sessão `tmux` (mantém o processo vivo se a conexão SSH cair): `scripts/run_all.sh`. A saída geral vai para o terminal e para `runs/run_all.log` (o script cria `runs/`, que não existe após o clone, pois o git não versiona pastas vazias).
- **Máquina local:** implementação, testes rápidos (minutos), análise dos resultados, tabela comparativa e gráficos.
- **Versionado no git:** código, `data/*.csv`, `data/vocab.json`, métricas, logs e previsões de `runs/` (JSON/CSV) e o checkpoint `best.pt` de cada experimento (pequeno, entre ~0,1 e 3 MB, para inspeção local do modelo).
- **Fora do git (`.gitignore`):** `.venv/` e os demais checkpoints (`*.pt`, exceto `best.pt`).
- Ambiente: um venv (`.venv/`) por máquina, criado a partir do `requirements.txt` (PyTorch CPU). O código deve ser compatível com Python 3.12 e 3.13.

### Saídas de cada experimento (`runs/expN/`)
| Arquivo | Conteúdo |
|---|---|
| `config.json` | Configuração usada + número de parâmetros + ambiente (versões, threads) |
| `history.csv` | Por época: `train_loss`, `val_loss` (notação científica, para não perder precisão quando ficam muito pequenas), `val_exact_match`, `val_em_day_1digit`, `val_em_day_2digits`, `val_em_month_upper`, `val_em_month_lower`, `val_year_accuracy`, `val_month_accuracy`, `val_day_accuracy`, `ood_monitor_exact_match`, `ood_monitor_year_accuracy`, `lr`, tempo da época (inclui validação e monitoramento OOD), throughput de treino (tokens/s); gravado a cada época |
| `train_summary.json` | Melhor época, melhor `val_loss`, épocas rodadas, se parou por early stopping, tempo total de treino (inclui a validação de cada época; o throughput de treino puro fica no `history.csv`) |
| `best.pt` | Checkpoint da época de menor `val_loss` |
| `metrics_test.json`, `metrics_ood.json` | Métricas finais (tabela da seção de avaliação) |
| `predictions_test.csv`, `predictions_ood.csv` | `source`, `target`, `prediction`, `correct` |
| `train.log` | Log textual do treino |

---

## Testes

Testes com `pytest` (arquivo `tests/test_essentials.py`; rodar `pytest` na raiz do projeto), **apenas os essenciais**, focados em bugs silenciosos (que não geram erro mas estragam o treino). Cada teste foi validado introduzindo o bug correspondente e conferindo que ele falha:
1. Tokenizer: encode → decode devolve a string original; tokens especiais são indivisíveis.
2. Dataset: shift correto e labels `-100` exatamente nas posições esperadas (só ISO + `<eos>` contam).
3. Máscara causal: alterar um token futuro não altera a saída das posições anteriores (cobre também a dispensa da máscara de padding).
4. LayerNorm à mão: mesmo resultado que `nn.LayerNorm`.

---

## Estrutura do projeto

```text
attention/
├── specifications.md
├── requirements.txt
├── .gitignore
├── didatic/                    # material de estudo (paper, resumo, enunciado)
│   ├── transformer.md
│   ├── paper_attention_en.pdf
│   ├── paper_attention_ptbr.pdf
│   └── Desafio Neospace.pdf
├── configs/
│   ├── exp1.json
│   ├── exp2.json
│   └── exp3.json
├── data/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   ├── test_ood.csv
│   └── vocab.json
├── src/
│   ├── generate_data.py        # geração e divisão determinísticas
│   ├── tokenizer.py            # char-level + tokens especiais
│   ├── dataset.py              # Dataset, shift, mask de loss e collate/padding
│   ├── model.py                # LayerNorm, MHA, FFN, DecoderBlock, Transformer decoder-only
│   ├── generate.py             # geração autoregressiva greedy
│   ├── train.py                # treino/validação/checkpoints/early stopping
│   ├── metrics.py              # funções de métrica (usadas por train.py e evaluate.py)
│   ├── evaluate.py             # métricas de teste e OOD
│   └── report.py               # análise local: tabela comparativa e curvas de aprendizado
├── tests/
│   └── test_essentials.py      # os 4 testes essenciais (pytest)
├── pytest.ini                  # pytest: testes em tests/, módulos importados de src/
├── scripts/
│   ├── run_exp1.sh
│   ├── run_exp2.sh
│   ├── run_exp3.sh
│   └── run_all.sh
└── runs/
    ├── exp1/
    ├── exp2/
    └── exp3/
```

Tabela comparativa (pode sofrer alterações):

| Experimento | Params | Melhor época | Train loss | Val loss | Test loss | Val exact match | Test exact match | OOD exact match | Tempo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Exp. 1 | — | — | — | — | — | — | — | — | — |
| Exp. 2 | — | — | — | — | — | — | — | — | — |
| Exp. 3 | — | — | — | — | — | — | — | — | — |

Analisar trade-off entre **capacidade, acurácia e custo**.

### Relatório (`src/report.py`, roda na máquina local)
Lê os resultados de `runs/expN/` (experimentos incompletos são pulados) e gera, versionados no git:
- `runs/report.md`: tabela comparativa acima (métricas da melhor época), tabelas de diagnóstico de teste e OOD (acurácia por caractere, por campo, saídas válidas, variantes de formatação), exact match por mês e exemplos de erro.
- `runs/figures/learning_curves.png`: curvas de aprendizado — train loss, val loss (escala log) e val exact match por época, um painel por métrica (um único eixo y cada), com a melhor época marcada.
- `runs/figures/variant_curves.png`: evolução por época do exact match de validação nas variantes raras (linha cheia) contra a forma comum dos mesmos casos (tracejada), e do OOD na amostra de monitoramento.
- `runs/figures/field_curves.png`: acurácia de validação do ano, do mês e do dia por época (um painel por campo).
- `runs/figures/exact_match_by_case.png`: exact match de cada experimento no teste, nas variantes raras (dia com 1 dígito, mês maiúsculo) e no OOD.

Cada experimento tem uma cor fixa em todos os gráficos (azul, laranja, verde-água).

### Fora de escopo (decidido)
- **Experimento de quantidade de dados** (treinar com 1k/5k/20k exemplos para medir eficiência de dados): responderia uma pergunta diferente das curvas de aprendizado (quantos exemplos *distintos* cada capacidade precisa), mas não será executado por falta de tempo. Fica como próximo passo sugerido.
- **Múltiplas seeds por configuração**: também fica como próximo passo.

---

## Ordem de implementação (pode sofrer alteração)

1. `generate_data.py` → dados válidos e quatro CSVs sem interseções.
2. `tokenizer.py` → encode/decode, IDs especiais, `vocab.json` reproduzível.
3. `dataset.py` → full sequence, shift, labels `-100`, padding.
4. `model.py` → embedding + posição + blocos causais + logits.
5. `generate.py` → greedy decoding a partir de `<bos>source<sep>`.
6. `train.py` → forward/loss/backward/AdamW/schedule/validação/checkpoint/early stopping.
7. `evaluate.py` → test loss, exact match e demais métricas (teste e OOD).
8. Testes essenciais (`tests/`).
9. Configs + shell scripts → um script por experimento + script organizador.
10. Produzir tabela comparativa, curvas de aprendizado (loss e exact match por época) de cada modelo, exemplos gerados e discussão dos resultados.
