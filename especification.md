# Especificações do projeto

## Objetivo 
Implementar a estrutura de um Transformer decoder-only, seu loop de treinamento e sua avaliação para converter datas em português para o padrão ISO `YYYY-MM-DD`. Fazer todo o modelo "à mão" e usar pytorch apenas quando a implementação for muito complicada / desnecessária / custosa (mal otimizada).

Exemplo:
```text
Entrada: 25 de Dezembro de 2023
Saída:   2023-12-25
```

Fluxo do trabalho:
1. Gerar um dataset supervisionado de datas, no formato adequado;
2. Separar os exemplos em treino, validação e teste **antes de formar tensores**.
3. Implementar um modelo Decoder-only autoregressivo com as subcamadas **masked multi-head self-attention e FFN.**, além do passo intermediário Add&Norm.
4. Implementar o treinamento com Cross-Entropy, backpropagation e AdamW.
5. Treinar **três configurações de capacidade diferente**, usando as CPUs do servidor.
6. Avaliar os modelos em dados não vistos, principalmente com **exact-match accuracy**.
7. Registrar configuração, métricas e duração para comparação e reprodução.

**Escopo:** é um Transformer decoder-only, demonstrativo da arquitetura de modelos de linguagem autoregressivos. **Não há Encoder nem Cross-Attention.**

---

## Lista de Definições
| Item | Definição |
|---|---|
| Linguagem/framework | Python + PyTorch |
| Arquitetura | Transformer decoder-only |
| Tarefa | `DD de Mês de YYYY` → `YYYY-MM-DD` |
| Idioma da entrada | Português |
| Tokenizador | **Por caractere** (tokens especiais indivisíveis) |
| Formatação do dia | **2 dígitos** (`01`, `13`, `25`, etc.) |
| Formatação do ano | **4 dígitos**, intervalo 1800–2099 |
| Vocabulário | Caracteres que aparecem no corpus + `<bos>`, `<sep>`, `<eos>`, `<pad>` |
| Entrada de treinamento | Prompt + resposta ISO, com shifting de 1 token |
| Loss | Cross-Entropy somente nos tokens ISO e `<eos>`; demais posições `-100` |
| Attention | Máscara Causal (nunca consultar posições futuras) |
| FFN | Duas lineares com ReLU; expansão $4 * d_{model}$ (modelo do paper)|
| Add & Norm | Post-LayerNorm, como no paper original, em cada subcamada |
| Posição | Embeddings posicionais senoidais (padrão do paper) + escalonamento |
| Otimizador | AdamW |
| Dropout | 0.1 |
| Divisão | 20.000 treino / 2.500 validação / 2.500 teste / Aumentar se necessário |
| Principal métrica | Exact match da string ISO gerada |
| Ambiente | CPU, servidor de 20 núcleos físicos / 40 CPUs lógicas |
| Experimentos | 3 capacidades, variando `d_model` e `n_layers` |

---

## Dataset: geração, unicidade e divisão
- Gerar datas válidas do intervalo `1800-01-01` a `2099-12-31`.
- Selecionar **25.000 datas únicas, sem reposição**, do intervalo `1800-01-01` a `2049-12-31` para o treino. O restante do intervalo será usado para verificar a capacidade de generalização do modelo no teste.
- Formatar a origem como `"{dia:02d} de {mes} de {ano:04d}"`.
- Formatar o alvo com `date.isoformat()`.
- Meses: calendário completo `Janeiro, Fevereiro, Março, Abril, Maio, Junho, Julho, Agosto, Setembro, Outubro, Novembro, Dezembro`.
- Usar `datetime.date` e `timedelta` para garantir validade (inclusive anos bissextos).
- Fixar `seed=42` na geração e divisão para reprodutibilidade.

Dias, Meses e Anos **vão se repetir**. O que deve ser único é a combinação completa de dia + mês + ano.

### Splits

| Split | Número | Uso |
|---|---:|---|
| Treino | 20.000 | Atualização dos pesos |
| Validação | 2.500 | Seleção de checkpoint, early stopping e escolha de hiperparâmetros |
| Teste | 2.500 | **Avaliação final**, após congelar escolhas |

Gerar/sortear as datas únicas, embaralhar com gerador determinístico e dividir. Salvar os três splits em arquivos persistentes (`data/train.csv`, `data/val.csv`, `data/test.csv`) para que **todos os experimentos usem exatamente os mesmos dados**. Colunas: `source`, `target`.

---

## Tokenização e formatação autoregressiva

A tokenização ocorre por caractere. Assim, o vocabulário contém apenas:
- Os dígitos de 0 a 9.
- As letras presentes nos nomes dos meses.
- O espaço " ".
- O hífen "-".
- Os tokens especiais (`<bos>, <sep>, <eos>, <pad>`), que devem ser indivisíveis.

Exemplo:
```text
source: 25 de Dezembro de 2023
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
full:    <bos>25 de Dezembro de 2023<sep>2023-12-25<eos>
input:   <bos>25 de Dezembro de 2023<sep>2023-12-25
target:       25 de Dezembro de 2023<sep>2023-12-25<eos>
```

### Loss somente na resposta
As previsões correspondentes à data original e ao marcador `<sep>` **não contribuem** para a Cross-Entropy, apenas as previsões correspondentes a **cada caractere da data ISO + `<eos>`**.

### Padding e limite de contexto
- Usar `collate_fn` para completar as sequências do batch até o maior comprimento do batch.
- Usar `<pad>` na entrada e `-100` nos labels de posições preenchidas.
- Para a atenção, bloquear **keys** de padding (`key_padding_mask`) e continuar aplicando máscara causal; padding de query não afeta a loss se seus labels forem ignorados.
- Preferir `max_seq_len=80` (suficiente para os formatos escolhidos, verificar programaticamente antes do treino).
- Não truncar exemplos de forma silenciosa; validar que todas as sequências cabem no contexto escolhido.

### Inferência (sem teacher forcing)

Fornecer somente o prompt:
```text
<bos>25 de Dezembro de 2023<sep>
```

Gerar **um caractere por vez**, sempre usando os logits da última posição, acrescentando cada token gerado à sequência. Parar ao produzir `<eos>` ou ao atingir `max_new_tokens` (ex.: 11 tokens: 10 caracteres ISO + `<eos>`; pode permitir pequena folga para diagnosticar saídas inválidas).

Para a métrica final, utilizar `greedy decoding` (`argmax`) para obter resultados determinísticos. Comparar somente a string extraída após `<sep>` e antes de `<eos>` com o target exato. Qualquer resposta incompleta, excedente ou malformatada conta como incorreta.

---

## Configurações dos experimentos

Três modelos treinados **do zero**, com parâmetros independentes; os splits, seed, protocolo de treino e métricas serão iguais.

| Configuração | Exp. 1 | Exp. 2 | Exp. 3 |
|---|---:|---:|---:|
| `d_model` | **32** | **64** | **128** |
| `n_layers` | **2** | **3** | **4** |
| `n_heads` | 4 | 4 | 4 |
| `d_ff` | 128 | 256 | 512 |
| `max_seq_len` | 80 | 80 | 80 |
| `dropout` | 0.1 | 0.1 | 0.1 |
| `batch_size` | 64 | 64 | 64 |
| `learning_rate` | 3e-4 | 3e-4 | 3e-4 |
| `optimizer` | AdamW | AdamW | AdamW |
| `max_epochs` | 50 | 50 | 50 |
| `early_stopping_patience` | 5 | 5 | 5 |
| `seed` | 42 | 42 | 42 |
| `device` | cpu | cpu | cpu |
| `threads` inicial | **4** | **6** | **8** |
| CPU affinity inicial | **0–3** | **4–9** | **10–17** |

**Interpretação:** Busco comparar a capacidade, variando simultaneamente largura (`d_model`) e profundidade (`n_layers`), não isolando o efeito de cada fator. Registrar o número efetivo de parâmetros e o tempo de treinamento para uma comparação completa.

### Critério de seleção
- Ao final de cada época, calcular `train_loss`, `val_loss` e exact match de validação.
- Early stopping com paciência 5 (parar após cinco épocas seguidas sem melhora).
- Recarregar a melhor época e avaliar no teste.

---

## Estrutura do projeto

```text
date-transformer/
├── instructions.md
├── requirements.txt
├── data/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── vocab.json
├── src/
│   ├── generate_data.py       # geração e divisão determinísticas
│   ├── tokenizer.py           # char-level + tokens especiais
│   ├── dataset.py             # Dataset, shift, mask de loss e collate/padding
│   ├── model.py               # DecoderBlock + Transformer Decoder-Only
│   ├── generate.py            # geração autoregressiva greedy
│   ├── train.py               # treino/validação/checkpoints
│   └── evaluate.py            # geração e exact match de teste
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

| Experimento | Params | Melhor época | Train loss | Val loss | Test loss | Val exact match | Test exact match | Tempo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Exp. 1 | — | — | — | — | — | — | — | — |
| Exp. 2 | — | — | — | — | — | — | — | — |
| Exp. 3 | — | — | — | — | — | — | — | — |

Analisar trade-off entre **capacidade, acurácia e custo**;

### Experimento de generalização
Além da divisão aleatória principal, avaliar generalização para **anos não vistos**: treino até 2049 e teste adicional em 2050–2099, com splits apropriados.

---

## Ordem de implementação (pode sofrer alteração)

**Implementação sugerida, nesta ordem:**
1. `generate_data.py` → dados válidos e três CSVs sem interseções.
2. `tokenizer.py` → encode/decode, IDs especiais, `vocab.json` reproduzível.
3. `dataset.py` → full sequence, shift, labels `-100`, padding, testes de índice.
4. `model.py` → embedding + posição + blocos causais + logits.
5. `generate.py` → greedy decoding a partir de `<bos>source<sep>`.
6. `train.py` → forward/loss/backward/AdamW/validação/checkpoint/early stopping.
7. `evaluate.py` → test loss e exact match gerado.
8. Shell scripts → um script por experimento + script organizador
10. Produzir tabela comparativa, gráfico de learning rate de cada modelo, exemplos gerados e discussão dos resultados.
