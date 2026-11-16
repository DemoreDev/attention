"""
Geração e divisão do dataset de datas.

Gera os splits de treino/validação/teste (datas de 1800–2049) e o teste OOD
(todas as datas de 2050–2099), cada exemplo com o formato:

    source: "25 de dezembro de 2023" (com variações de formatação para 
    avaliar capacidade de generalização do modelo)
    target: "2023-12-25"

Uso:
    python src/generate_data.py [--out-dir data] [--seed 42]
"""

import argparse
import csv
import random
from datetime import date, timedelta
from pathlib import Path

MONTHS = [
    "janeiro", "fevereiro", "março", "abril", "maio", "junho",
    "julho", "agosto", "setembro", "outubro", "novembro", "dezembro",
]

# Intervalos de datas (inclusivos).
IN_DIST_RANGE = (date(1800, 1, 1), date(2049, 12, 31))
OOD_RANGE = (date(2050, 1, 1), date(2099, 12, 31))

# Tamanho dos splits sorteados do intervalo 1800–2049.
N_TRAIN, N_VAL, N_TEST = 20_000, 2_500, 2_500

# Probabilidades das variações de formatação. O número 0.2 foi escolhido porque
# isso gera poucos exemplos de treinamento / avaliação, servindo como "desafio" ao modelo
P_SINGLE_DIGIT_DAY = 0.2  # dias 1–9 escritos com 1 dígito ("5" em vez de "05")
P_CAPITALIZED_MONTH = 0.2  # mês com inicial maiúscula ("Dezembro" em vez de "dezembro")


def all_dates(start: date, end: date) -> list[date]:
    """Todas as datas válidas de start a end (inclusive), em ordem."""
    n_days = (end - start).days + 1
    return [start + timedelta(days=i) for i in range(n_days)]


def format_source(d: date, rng: random.Random) -> str:
    """Escreve a data por extenso, sorteando as variações de formatação."""
    day = f"{d.day:02d}"
    if d.day <= 9 and rng.random() < P_SINGLE_DIGIT_DAY:
        day = str(d.day)

    month = MONTHS[d.month - 1]
    if rng.random() < P_CAPITALIZED_MONTH:
        month = month.capitalize()

    return f"{day} de {month} de {d.year:04d}"


def to_examples(dates: list[date], rng: random.Random) -> list[tuple[str, str]]:
    return [(format_source(d, rng), d.isoformat()) for d in dates]


def save_csv(path: Path, examples: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target"])
        writer.writerows(examples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Geradores separados: mudar as taxas de formatação não altera
    # quais datas são sorteadas para cada split.
    selection_rng = random.Random(args.seed)
    format_rng = random.Random(args.seed + 1)

    # Sorteio uniforme sem reposição; random.sample já devolve em ordem aleatória.
    pool = all_dates(*IN_DIST_RANGE)
    selected = selection_rng.sample(pool, N_TRAIN + N_VAL + N_TEST)
    splits = {
        "train": selected[:N_TRAIN],
        "val": selected[N_TRAIN:N_TRAIN + N_VAL],
        "test": selected[N_TRAIN + N_VAL:],
        "test_ood": all_dates(*OOD_RANGE),
    }

    # Verificações: tamanhos e ausência de interseção entre splits.
    names = list(splits)
    for i, a in enumerate(names):
        assert len(set(splits[a])) == len(splits[a]), f"datas repetidas em {a}"
        for b in names[i + 1:]:
            assert not set(splits[a]) & set(splits[b]), f"interseção entre {a} e {b}"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, dates in splits.items():
        examples = to_examples(dates, format_rng)
        save_csv(args.out_dir / f"{name}.csv", examples)

        n_single = sum(src.split(" ")[0].isdigit() and len(src.split(" ")[0]) == 1 for src, _ in examples)
        n_cap = sum(src.split(" ")[2][0].isupper() for src, _ in examples)
        print(f"{name:>9}: {len(examples):>6} exemplos | "
              f"dia com 1 dígito: {n_single:>5} | mês maiúsculo: {n_cap:>5}")


if __name__ == "__main__":
    main()
