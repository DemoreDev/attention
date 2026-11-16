"""
Métricas sobre as previsões geradas (texto), usadas no treino (validação por época)
e na avaliação final (teste e OOD).

Ficam em um arquivo próprio porque train.py e evaluate.py usam as mesmas funções:
se uma ficasse dentro do outro, os dois arquivos importariam um ao outro (import
circular, que o Python não resolve).
"""

import re
from datetime import date

ISO_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")
ISO_LENGTH = 10

# Nomes dos grupos de formatação (usados como chaves nos JSONs e no relatório).
DAY_1_DIGIT = "dia 1-9 com 1 dígito"
DAY_2_DIGITS = "dia 1-9 com 2 dígitos"
DAY_10_31 = "dia 10-31"
MONTH_UPPER = "mês maiúsculo"
MONTH_LOWER = "mês minúsculo"


def exact_match(predictions: list[str], targets: list[str]) -> float:
    """Fração de previsões idênticas ao target."""
    return sum(p == t for p, t in zip(predictions, targets)) / len(targets)


def is_valid_iso_date(text: str) -> bool:
    """True se o texto é uma data ISO existente (formato YYYY-MM-DD e data válida, ex.: sem 30 de fevereiro)."""
    if not ISO_PATTERN.fullmatch(text):
        return False
    try:
        date.fromisoformat(text)
        return True
    except ValueError:
        return False


def char_accuracy(predictions: list[str], targets: list[str]) -> float:
    """Fração dos caracteres ISO corretos na posição certa.

    Compara as 10 posições do target; se a previsão for mais curta, as posições
    que faltam contam como erro (e caracteres excedentes são ignorados aqui; eles
    já derrubam o exact match).
    """
    correct = sum(
        sum(i < len(pred) and pred[i] == tgt[i] for i in range(ISO_LENGTH))
        for pred, tgt in zip(predictions, targets)
    )
    return correct / (ISO_LENGTH * len(targets))


def field_accuracy(predictions: list[str], targets: list[str]) -> dict[str, float]:
    """Exact match separado de cada campo: ano (YYYY), mês (MM) e dia (DD).

    Os campos são lidos pelas posições fixas do formato ISO (0–3, 5–6, 8–9).
    Útil para ver ONDE o modelo erra (ex.: acerta dia e mês, mas erra o ano).
    """
    fields = {"year": slice(0, 4), "month": slice(5, 7), "day": slice(8, 10)}
    return {
        name: sum(pred[s] == tgt[s] for pred, tgt in zip(predictions, targets)) / len(targets)
        for name, s in fields.items()
    }


def day_variant(source: str) -> str:
    """Formatação do dia no source. Separa os dias 1–9 (os únicos que podem variar)."""
    day = source.split(" ")[0]
    if len(day) == 1:
        return DAY_1_DIGIT
    if day.startswith("0"):
        return DAY_2_DIGITS
    return DAY_10_31


def month_variant(source: str) -> str:
    """Formatação do mês no source: minúsculo (padrão) ou com inicial maiúscula."""
    month = source.split(" ")[2]
    return MONTH_UPPER if month[0].isupper() else MONTH_LOWER


def exact_match_by_group(correct: list[bool], groups: list[str]) -> dict[str, dict]:
    """Exact match calculado separadamente para cada grupo (com o número de exemplos)."""
    result = {}
    for group in sorted(set(groups)):
        hits = [c for c, g in zip(correct, groups) if g == group]
        result[group] = {"n": len(hits), "exact_match": sum(hits) / len(hits)}
    return result
