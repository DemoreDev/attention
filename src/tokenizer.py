"""
Tokenizador por caractere, com tokens especiais indivisíveis.

Cada caractere do texto vira um token (um ID inteiro). Os tokens especiais
(<pad>, <bos>, <sep>, <eos>) nunca são produzidos a partir do texto: eles são
inseridos diretamente pelo ID (no dataset e na geração), por isso não são
divisíveis.

O vocabulário é construído a partir do split de treino e salvo em data/vocab.json.

Uso (constrói e salva o vocabulário):
    python src/tokenizer.py [--data-dir data]
"""

import argparse
import csv
import json
from pathlib import Path

# Tokens especiais. A ordem define os IDs: <pad> = 0, <bos> = 1, <sep> = 2, <eos> = 3.
PAD, BOS, SEP, EOS = "<pad>", "<bos>", "<sep>", "<eos>"
SPECIAL_TOKENS = [PAD, BOS, SEP, EOS]


class CharTokenizer:
    """Converte texto <-> lista de IDs, um caractere por token."""

    def __init__(self, chars: list[str]):
        # Lista completa de tokens: primeiro os especiais, depois os caracteres.
        # O ID de um token é a sua posição nesta lista.
        self.tokens = SPECIAL_TOKENS + list(chars)

        # Dicionários de ida e volta (token -> ID e ID -> token).
        self.token_to_id = {tok: i for i, tok in enumerate(self.tokens)}
        self.id_to_token = dict(enumerate(self.tokens))

        # Atalhos para os IDs especiais, usados no dataset e na geração.
        self.pad_id = self.token_to_id[PAD]
        self.bos_id = self.token_to_id[BOS]
        self.sep_id = self.token_to_id[SEP]
        self.eos_id = self.token_to_id[EOS]

    @property
    def vocab_size(self) -> int:
        """Número total de tokens (|V|), que define o tamanho da matriz de embeddings E."""
        return len(self.tokens)

    @classmethod
    def from_texts(cls, texts: list[str]) -> "CharTokenizer":
        """Constrói o vocabulário com todos os caracteres que aparecem nos textos.

        Os caracteres são ordenados para que o vocabulário (e portanto os IDs)
        seja sempre o mesmo, independentemente da ordem dos textos.
        """
        chars = sorted(set("".join(texts)))
        return cls(chars)

    def text_to_ids(self, text: str) -> list[int]:
        """Texto -> lista de IDs. Falha se houver caractere fora do vocabulário."""
        unknown = set(text) - self.token_to_id.keys()
        if unknown:
            raise ValueError(f"caracteres fora do vocabulário: {sorted(unknown)} em {text!r}")
        return [self.token_to_id[ch] for ch in text]

    def ids_to_text(self, ids: list[int], skip_special: bool = False) -> str:
        """Lista de IDs -> texto.

        Com skip_special=True, os tokens especiais são omitidos; caso contrário,
        aparecem escritos como "<bos>", "<sep>" etc. (útil para depuração).
        """
        special_ids = {self.token_to_id[tok] for tok in SPECIAL_TOKENS}
        return "".join(
            self.id_to_token[i] for i in ids
            if not (skip_special and i in special_ids)
        )

    def save(self, path: Path) -> None:
        """Salva o vocabulário em JSON (a lista de tokens, na ordem dos IDs)."""
        with path.open("w", encoding="utf-8") as f:
            json.dump({"tokens": self.tokens}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        """Carrega um vocabulário salvo por save()."""
        with path.open(encoding="utf-8") as f:
            tokens = json.load(f)["tokens"]
        # Garante que o arquivo foi gerado com os mesmos tokens especiais, na mesma ordem.
        assert tokens[:len(SPECIAL_TOKENS)] == SPECIAL_TOKENS, "tokens especiais inesperados no vocab"
        return cls(tokens[len(SPECIAL_TOKENS):])


def read_texts(csv_path: Path) -> list[str]:
    """Lê um split e devolve todos os textos (source e target) de todos os exemplos."""
    with csv_path.open(encoding="utf-8", newline="") as f:
        return [text for row in csv.DictReader(f) for text in (row["source"], row["target"])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    # O vocabulário vem apenas do treino (o modelo só "conhece" o que viu no treino).
    tokenizer = CharTokenizer.from_texts(read_texts(args.data_dir / "train.csv"))

    # Os demais splits não podem ter caracteres que o treino não tem
    # (text_to_ids() levanta erro se encontrar algum).
    for split in ["val", "test", "test_ood"]:
        for text in read_texts(args.data_dir / f"{split}.csv"):
            tokenizer.text_to_ids(text)

    tokenizer.save(args.data_dir / "vocab.json")
    print(f"vocab_size = {tokenizer.vocab_size}")
    print("tokens:", tokenizer.tokens)


if __name__ == "__main__":
    main()
