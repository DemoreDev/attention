"""
Avaliação final de um experimento: carrega o best.pt e mede teste e teste OOD.

Para cada split:
    - loss (teacher forcing) e exact match (geração greedy, sem teacher forcing);
    - métricas de diagnóstico: acurácia por caractere, por campo (ano/mês/dia),
      taxa de saída válida, exact match por variante de formatação e por mês;
    - exemplos de erro.

Saídas no diretório do experimento:
    metrics_test.json, metrics_test_ood.json           métricas
    predictions_test.csv, predictions_test_ood.csv     source, target, prediction, correct

Uso:
    python src/evaluate.py --run-dir runs/exp1
"""

import argparse
import csv
import functools
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import DateDataset, collate_batch, read_examples
from generate import generate_dates, load_model
from tokenizer import CharTokenizer
from metrics import (
    char_accuracy, day_variant, exact_match, exact_match_by_group,
    field_accuracy, is_valid_iso_date, month_variant,
)
from train import evaluate_loss

N_ERROR_EXAMPLES = 20  # quantos erros listar no JSON (todos ficam no CSV de previsões)


def evaluate_split(model, tokenizer: CharTokenizer, examples: list[tuple[str, str]], max_seq_len: int) -> tuple[dict, list[str]]:
    """Calcula todas as métricas de um split. Devolve (métricas, previsões)."""
    sources = [src for src, _ in examples]
    targets = [tgt for _, tgt in examples]
    start = time.perf_counter()

    # Loss com teacher forcing (mesma medida da val_loss do treino).
    dataset = DateDataset(examples, tokenizer, max_seq_len)
    loader = DataLoader(dataset, batch_size=512, shuffle=False,
                        collate_fn=functools.partial(collate_batch, pad_id=tokenizer.pad_id))
    loss = evaluate_loss(model, loader)

    # Geração greedy, sem teacher forcing (o que importa na prática).
    predictions = generate_dates(model, sources, tokenizer)
    correct = [p == t for p, t in zip(predictions, targets)]
    errors = [{"source": s, "target": t, "prediction": p}
              for s, t, p, c in zip(sources, targets, predictions, correct) if not c]

    metrics = {
        "n_examples": len(examples),
        "loss": loss,
        "exact_match": exact_match(predictions, targets),
        "char_accuracy": char_accuracy(predictions, targets),
        "field_accuracy": field_accuracy(predictions, targets),
        "valid_output_rate": sum(map(is_valid_iso_date, predictions)) / len(predictions),
        "exact_match_by_day_format": exact_match_by_group(correct, [day_variant(s) for s in sources]),
        "exact_match_by_month_format": exact_match_by_group(correct, [month_variant(s) for s in sources]),
        "exact_match_by_month": exact_match_by_group(correct, [t[5:7] for t in targets]),
        "n_errors": len(errors),
        "error_examples": errors[:N_ERROR_EXAMPLES],
        "eval_time_s": round(time.perf_counter() - start, 1),
    }
    return metrics, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="diretório do experimento (com best.pt e config.json)")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    # Mesmas threads usadas no treino (lidas do config.json gravado pelo train.py).
    with (args.run_dir / "config.json").open(encoding="utf-8") as f:
        run_info = json.load(f)
    torch.set_num_threads(run_info["config"]["train"]["threads"])
    torch.set_num_interop_threads(1)

    tokenizer = CharTokenizer.load(args.data_dir / "vocab.json")
    model = load_model(args.run_dir / "best.pt")
    model.eval()

    for split in ["test", "test_ood"]:
        examples = read_examples(args.data_dir / f"{split}.csv")
        metrics, predictions = evaluate_split(model, tokenizer, examples, model.config.max_seq_len)

        with (args.run_dir / f"metrics_{split}.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        with (args.run_dir / f"predictions_{split}.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["source", "target", "prediction", "correct"])
            for (src, tgt), pred in zip(examples, predictions):
                writer.writerow([src, tgt, pred, int(pred == tgt)])

        # Resumo no terminal.
        fa = metrics["field_accuracy"]
        print(f"{split:>9}: loss {metrics['loss']:.3e} | exact match {metrics['exact_match']:.4f} | "
              f"char acc {metrics['char_accuracy']:.4f} | ano/mês/dia {fa['year']:.4f}/{fa['month']:.4f}/{fa['day']:.4f} | "
              f"saídas válidas {metrics['valid_output_rate']:.4f} | erros {metrics['n_errors']}")


if __name__ == "__main__":
    main()
