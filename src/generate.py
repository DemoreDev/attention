"""
Geração autoregressiva greedy (inferência, sem teacher forcing).

O modelo recebe só o prompt "<bos>data por extenso<sep>" e gera a data ISO um
caractere por vez: a cada passo usa os logits da ÚLTIMA posição real, escolhe o
token mais provável (argmax), acrescenta à sequência e repete, até gerar <eos>
ou atingir max_new_tokens.

Geração em batch: os prompts têm tamanhos diferentes (ex.: "5 de maio" × "15 de
fevereiro"), então ficam com padding à DIREITA, como no treino, e cada exemplo tem
o seu próprio "cursor" (comprimento atual). O token novo de cada exemplo é escrito
na posição do seu cursor, por cima de um <pad>. Pela máscara causal, nenhuma
posição real enxerga os <pad> que estão depois dela, então eles não interferem.

Uso (gera para datas digitadas; sem --checkpoint usa um modelo NÃO treinado):
    python src/generate.py --source "25 de dezembro de 2023" "5 de Março de 1935"
    python src/generate.py --checkpoint runs/exp1/best.pt --source "25 de dezembro de 2023"
"""

import argparse
from pathlib import Path

import torch

from dataset import build_prompt
from model import DecoderOnlyTransformer, ModelConfig
from tokenizer import CharTokenizer

# 10 caracteres ISO + <eos> + folga de 3 tokens, para diagnosticar saídas que não terminam.
MAX_NEW_TOKENS = 14


@torch.no_grad()  # inferência: não precisa calcular gradientes (mais rápido, menos memória)
def greedy_generate(
    model: DecoderOnlyTransformer,
    prompts: list[list[int]],
    pad_id: int,
    eos_id: int,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> list[list[int]]:
    """Gera a continuação de cada prompt. Devolve os IDs gerados, sem o <eos>.

    Se um exemplo não gerar <eos> em max_new_tokens passos, devolve tudo o que
    gerou (a saída será, portanto, maior que uma data ISO e contará como erro).
    """
    batch_size = len(prompts)
    lengths = torch.tensor([len(p) for p in prompts])  # cursor de cada exemplo
    prompt_lengths = lengths.clone()
    total_len = int(lengths.max()) + max_new_tokens
    assert total_len <= model.config.max_seq_len, "prompt + max_new_tokens excede max_seq_len"

    # Buffer (B, total_len) preenchido com <pad>; cada prompt é copiado para o início da sua linha.
    sequences = torch.full((batch_size, total_len), pad_id, dtype=torch.long)
    for row, prompt in enumerate(prompts):
        sequences[row, :len(prompt)] = torch.tensor(prompt)

    finished = torch.zeros(batch_size, dtype=torch.bool)  # quem já gerou <eos>
    rows = torch.arange(batch_size)

    for _ in range(max_new_tokens):
        # Forward na sequência inteira (sem cache: recalcula tudo a cada passo, simples
        # e suficiente para sequências tão curtas).
        logits = model(sequences)  # (B, total_len, |V|)

        # Logits da última posição real de cada exemplo (cursor - 1) e escolha greedy.
        next_tokens = logits[rows, lengths - 1].argmax(dim=-1)  # (B,)

        # Só avança quem ainda não terminou; quem terminou fica parado.
        active = ~finished
        sequences[rows[active], lengths[active]] = next_tokens[active]
        lengths[active] += 1
        finished |= next_tokens == eos_id

        if finished.all():
            break

    # Extrai o que foi gerado (do fim do prompt até o cursor), removendo o <eos> final.
    outputs = []
    for row in range(batch_size):
        generated = sequences[row, prompt_lengths[row]:lengths[row]].tolist()
        if generated and generated[-1] == eos_id:
            generated = generated[:-1]
        outputs.append(generated)
    return outputs


def generate_dates(
    model: DecoderOnlyTransformer,
    sources: list[str],
    tokenizer: CharTokenizer,
    batch_size: int = 512,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> list[str]:
    """Converte uma lista de datas por extenso nas datas ISO geradas pelo modelo (texto)."""
    # Modo de avaliação: desliga o dropout. O modo anterior é restaurado no final,
    # pois esta função também é chamada no meio do treino (validação por época).
    was_training = model.training
    model.eval()

    predictions = []
    for start in range(0, len(sources), batch_size):
        batch_sources = sources[start:start + batch_size]
        prompts = [build_prompt(src, tokenizer) for src in batch_sources]
        generated = greedy_generate(model, prompts, tokenizer.pad_id, tokenizer.eos_id, max_new_tokens)
        # skip_special=False: se o modelo gerar um token especial no meio (ex.: <sep>),
        # ele aparece no texto e a previsão é contada como errada.
        predictions += [tokenizer.ids_to_text(ids) for ids in generated]

    model.train(was_training)
    return predictions


def load_model(checkpoint_path: Path) -> DecoderOnlyTransformer:
    """Carrega um modelo salvo pelo treino (checkpoint com "model_config" e "model_state")."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = DecoderOnlyTransformer(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", nargs="+", required=True, help="datas por extenso")
    parser.add_argument("--checkpoint", type=Path, help="checkpoint treinado (ex.: runs/exp1/best.pt)")
    parser.add_argument("--vocab", type=Path, default=Path("data/vocab.json"))
    args = parser.parse_args()

    tokenizer = CharTokenizer.load(args.vocab)
    if args.checkpoint:
        model = load_model(args.checkpoint)
    else:
        print("(sem --checkpoint: modelo NÃO treinado, as saídas serão aleatórias)\n")
        torch.manual_seed(42)
        model = DecoderOnlyTransformer(ModelConfig(tokenizer.vocab_size, d_model=32, n_layers=2, n_heads=4, d_ff=128))

    for source, prediction in zip(args.source, generate_dates(model, args.source, tokenizer)):
        print(f"{source!r} → {prediction!r}")


if __name__ == "__main__":
    main()
