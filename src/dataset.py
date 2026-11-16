"""
Dataset para o treino autoregressivo (teacher forcing) do Transformer decoder-only.

Cada exemplo (source, target) vira UMA sequência, com entrada e saída juntas:

    full:   <bos>25 de dezembro de 2023<sep>2023-12-25<eos>

Shift de um token (teacher forcing): a posição t da entrada deve prever o token t+1.

    input:  <bos>25 de dezembro de 2023<sep>2023-12-25
    target:      25 de dezembro de 2023<sep>2023-12-25<eos>

A loss só considera as posições cujo alvo é um caractere ISO ou o <eos>; as demais
(alvo = caractere da data por extenso ou <sep>) recebem label -100, que a
Cross-Entropy do PyTorch ignora.

Uso (mostra um exemplo e verifica os comprimentos de todos os splits):
    python src/dataset.py [--data-dir data] [--max-seq-len 80]
"""

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset

from tokenizer import CharTokenizer

# Valor de label ignorado por F.cross_entropy (é o padrão de ignore_index do PyTorch).
IGNORE_INDEX = -100


def read_examples(csv_path: Path) -> list[tuple[str, str]]:
    """Lê um split e devolve a lista de pares (source, target)."""
    with csv_path.open(encoding="utf-8", newline="") as f:
        return [(row["source"], row["target"]) for row in csv.DictReader(f)]


def build_sequence(source: str, target: str, tokenizer: CharTokenizer) -> tuple[list[int], list[int]]:
    """Monta (input_ids, labels) de um exemplo, já com o shift de um token.

    labels[t] é o token que a posição t deve prever, ou IGNORE_INDEX quando essa
    previsão não entra na loss.
    """
    source_ids = tokenizer.text_to_ids(source)
    target_ids = tokenizer.text_to_ids(target)

    # Sequência completa: <bos> + data por extenso + <sep> + data ISO + <eos>
    full_ids = [tokenizer.bos_id] + source_ids + [tokenizer.sep_id] + target_ids + [tokenizer.eos_id]

    # Shift: a entrada perde o último token (<eos>) e o alvo perde o primeiro (<bos>).
    input_ids = full_ids[:-1]
    labels = full_ids[1:]

    # Os primeiros len(source) + 1 alvos são os caracteres da data por extenso e o
    # <sep>: o modelo recebe isso pronto, não precisa aprender a prever. Ignora na loss.
    # O que sobra são exatamente os 10 caracteres ISO + <eos>.
    n_ignored = len(source_ids) + 1
    labels[:n_ignored] = [IGNORE_INDEX] * n_ignored

    return input_ids, labels


def build_prompt(source: str, tokenizer: CharTokenizer) -> list[int]:
    """Prompt usado na inferência (sem a resposta): <bos> + data por extenso + <sep>."""
    return [tokenizer.bos_id] + tokenizer.text_to_ids(source) + [tokenizer.sep_id]


class DateDataset(Dataset):
    """Guarda os exemplos de um split já convertidos em (input_ids, labels).

    As sequências são montadas uma única vez, no construtor; __getitem__ só as devolve.
    """

    def __init__(self, examples: list[tuple[str, str]], tokenizer: CharTokenizer, max_seq_len: int):
        # Os pares originais ficam guardados para a avaliação (comparar texto gerado com o target).
        self.examples = examples
        self.sequences = [build_sequence(src, tgt, tokenizer) for src, tgt in examples]

        # Nenhum exemplo pode ser truncado em silêncio: se algum não couber no contexto, falha.
        longest = max(len(input_ids) for input_ids, _ in self.sequences)
        if longest > max_seq_len:
            raise ValueError(f"sequência de {longest} tokens excede max_seq_len={max_seq_len}")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids, labels = self.sequences[index]
        return torch.tensor(input_ids), torch.tensor(labels)


def collate_batch(batch: list[tuple[torch.Tensor, torch.Tensor]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Junta exemplos de tamanhos diferentes em dois tensores (batch, seq_len).

    Padding à DIREITA até o maior comprimento do batch: <pad> na entrada e
    IGNORE_INDEX nos labels. Como o <pad> fica sempre depois dos tokens reais, a
    máscara causal já impede que posições reais o consultem (não precisa de
    máscara de padding), e seus labels -100 não entram na loss.

    No DataLoader: collate_fn=functools.partial(collate_batch, pad_id=tokenizer.pad_id)
    """
    max_len = max(len(input_ids) for input_ids, _ in batch)

    # Começa tudo preenchido com padding e copia cada sequência para o início da sua linha.
    inputs = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
    for row, (input_ids, label_ids) in enumerate(batch):
        inputs[row, :len(input_ids)] = input_ids
        labels[row, :len(label_ids)] = label_ids

    return inputs, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--max-seq-len", type=int, default=80)
    args = parser.parse_args()

    tokenizer = CharTokenizer.load(args.data_dir / "vocab.json")

    # Exemplo legível: mostra entrada e label posição a posição ("·" = ignorado na loss).
    source, target = read_examples(args.data_dir / "train.csv")[0]
    input_ids, labels = build_sequence(source, target, tokenizer)
    print(f"source: {source!r}  target: {target!r}\n")
    print(f"{'pos':>3}  {'entrada':>7}  {'label':>7}")
    for pos, (inp, lab) in enumerate(zip(input_ids, labels)):
        lab_text = "·" if lab == IGNORE_INDEX else tokenizer.ids_to_text([lab])
        print(f"{pos:>3}  {tokenizer.ids_to_text([inp]):>7}  {lab_text:>7}")

    # Verificação programática do contexto: constrói todos os splits (falha se algum exceder).
    print()
    for split in ["train", "val", "test", "test_ood"]:
        dataset = DateDataset(read_examples(args.data_dir / f"{split}.csv"), tokenizer, args.max_seq_len)
        longest = max(len(input_ids) for input_ids, _ in dataset.sequences)
        print(f"{split:>9}: {len(dataset):>6} exemplos | maior entrada: {longest} tokens (limite {args.max_seq_len})")


if __name__ == "__main__":
    main()
