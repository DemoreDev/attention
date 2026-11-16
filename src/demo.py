"""
Para cada modelo mostra:
    - a data ISO gerada e se ela está correta (✓/✗), quando a entrada segue o formato
      "D de mês de AAAA" (a resposta esperada é calculada por regra, sem o modelo);
    - a geração caractere por caractere, com a CONFIANÇA de cada escolha (a probabilidade
      que a softmax deu ao token escolhido). Confiança baixa indica onde o modelo "hesitou".

Uso (na raiz do projeto):
    python src/demo.py                                   # modo interativo (digite "sair" para encerrar)
    python src/demo.py "25 de dezembro de 2023" "1 de janeiro de 2100"
    python src/demo.py --models exp2                     # só um modelo (ou --models exp1,exp3)
"""

import argparse
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from dataset import build_prompt
from generate import MAX_NEW_TOKENS, load_model
from model import DecoderOnlyTransformer
from tokenizer import CharTokenizer

MONTHS = [
    "janeiro", "fevereiro", "março", "abril", "maio", "junho",
    "julho", "agosto", "setembro", "outubro", "novembro", "dezembro",
]
# Formato esperado da entrada: dia (1 ou 2 dígitos) "de" mês "de" ano (4 dígitos).
SOURCE_PATTERN = re.compile(r"(\d{1,2}) de (\w+) de (\d{4})")
LOW_CONFIDENCE = 0.90  # abaixo disso, o caractere é marcado com "!"


def expected_answer(source: str) -> str | None:
    """Resposta correta calculada por regra (sem o modelo), ou None se a entrada fugir do formato.

    Não verifica se a data existe (ex.: 31 de fevereiro): só converte o formato, como o modelo.
    """
    match = SOURCE_PATTERN.fullmatch(source.strip())
    if not match or match.group(2).lower() not in MONTHS:
        return None
    day, month_name, year = match.groups()
    month = MONTHS.index(month_name.lower()) + 1
    return f"{year}-{month:02d}-{int(day):02d}"


@torch.no_grad()
def generate_with_confidence(model: DecoderOnlyTransformer, source: str,
                             tokenizer: CharTokenizer) -> list[tuple[str, float]]:
    """Geração greedy de UM exemplo, guardando a probabilidade de cada token escolhido.

    É o mesmo procedimento de generate.greedy_generate, escrito para um exemplo só
    (mais simples de ler): a cada passo, softmax nos logits da última posição,
    escolhe o token mais provável e o acrescenta à sequência.
    """
    ids = build_prompt(source, tokenizer)
    steps = []
    for _ in range(MAX_NEW_TOKENS):
        logits = model(torch.tensor([ids]))         # (1, T, |V|)
        probs = F.softmax(logits[0, -1], dim=-1)    # distribuição do próximo token
        token = int(probs.argmax())
        steps.append((tokenizer.ids_to_text([token]), float(probs[token])))
        if token == tokenizer.eos_id:
            break
        ids.append(token)
    return steps


def show(source: str, models: dict[str, DecoderOnlyTransformer], tokenizer: CharTokenizer) -> None:
    """Roda todos os modelos para uma entrada e imprime o resultado."""
    # Entrada com caractere fora do vocabulário: o modelo não tem como representá-la.
    unknown = sorted(set(source) - set(tokenizer.token_to_id))
    if unknown:
        print(f"  caracteres fora do vocabulário: {unknown} — o modelo só conhece "
              f"os caracteres vistos no treino (ex.: o mês em minúsculo ou com inicial maiúscula).\n")
        return

    # Entrada longa demais: prompt + resposta não caberiam no contexto do modelo (max_seq_len).
    max_seq_len = min(model.config.max_seq_len for model in models.values())
    if len(build_prompt(source, tokenizer)) + MAX_NEW_TOKENS > max_seq_len:
        print(f"  entrada longa demais: o contexto do modelo é de {max_seq_len} tokens.\n")
        return

    expected = expected_answer(source)
    print(f"  esperado: {expected if expected else '(entrada fora do formato; sem resposta de referência)'}")

    for name, model in models.items():
        steps = generate_with_confidence(model, source, tokenizer)
        # Resposta = tudo antes do <eos> (sem <eos>, a resposta não terminou e conta como errada).
        finished = steps[-1][0] == "<eos>"
        answer = "".join(tok for tok, _ in (steps[:-1] if finished else steps))
        mark = "" if expected is None else (" ✓" if answer == expected and finished else " ✗")
        lowest = min(prob for _, prob in steps)

        cfg = model.config
        print(f"  {name} (d={cfg.d_model}, L={cfg.n_layers}): {answer}{mark}   (menor confiança: {100 * lowest:.2f}%)")
        # Detalhe: cada caractere gerado com sua confiança; "!" marca as escolhas incertas.
        detail = "  ".join(
            f"{tok}{'!' if prob < LOW_CONFIDENCE else ''}:{100 * prob:.1f}" for tok, prob in steps
        )
        print(f"      {detail}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="*", help="datas por extenso (sem nenhuma: modo interativo)")
    # Lista separada por vírgula (ex.: --models exp1,exp2). 
    parser.add_argument("--models", default="exp1,exp2,exp3", help="ex.: exp2 ou exp1,exp3")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--vocab", type=Path, default=Path("data/vocab.json"))
    args = parser.parse_args()

    tokenizer = CharTokenizer.load(args.vocab)
    models = {}
    for name in args.models.split(","):
        checkpoint = args.runs_dir / name / "best.pt"
        if checkpoint.exists():
            models[name] = load_model(checkpoint).eval()  # eval: sem dropout
        else:
            print(f"{name}: {checkpoint} não encontrado, pulando")
    if not models:
        raise SystemExit("nenhum modelo encontrado")

    # Modo direto: processa as datas passadas na linha de comando e termina.
    if args.sources:
        for source in args.sources:
            print(f"> {source}")
            show(source, models, tokenizer)
        return

    # Modo interativo.
    print("Digite uma data por extenso (ex.: 25 de dezembro de 2023). \"exit\" para encerrar.\n")
    while True:
        try:
            source = input("> ").strip()
        except (EOFError, KeyboardInterrupt):  # Ctrl+D / Ctrl+C
            print()
            break
        if source.lower() in {"sair", "exit", "quit"}:
            break
        if source:
            show(source, models, tokenizer)


if __name__ == "__main__":
    main()
