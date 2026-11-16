"""
Treinamento do Transformer decoder-only (um experimento por execução).

Por época:
    1. Treino: forward → Cross-Entropy (só nas posições da resposta) → backward →
       clipping do gradiente → passo do AdamW → atualização do LR (warmup + cosseno).
    2. Validação: val_loss (teacher forcing) e exact match (geração greedy), também
       separado por variante de formatação (dia com 1 dígito, mês maiúsculo...).
       Monitoramento OOD: exact match numa amostra fixa de 2.500 datas de 2050–2099
       (só observação: NÃO influencia o early stopping nem nenhuma outra escolha).
    3. Early stopping pela val_loss: guarda o checkpoint da melhor época (best.pt)
       e para após `early_stopping_patience` épocas seguidas sem melhora.

Saídas em --out-dir (ex.: runs/exp1/):
    config.json          configuração usada + nº de parâmetros + ambiente
    history.csv          métricas por época (gravado a cada época)
    train_summary.json   melhor época, tempo total, se parou por early stopping
    best.pt              checkpoint da melhor época (menor val_loss)
    train.log            log textual

Uso:
    python src/train.py --config configs/exp1.json --out-dir runs/exp1
    python src/train.py --config configs/exp1.json --out-dir /tmp/teste --quick   # teste local rápido
"""

import argparse
import csv
import functools
import json
import logging
import math
import os
import platform
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import IGNORE_INDEX, DateDataset, collate_batch, read_examples
from generate import generate_dates
from metrics import (
    DAY_1_DIGIT, DAY_2_DIGITS, MONTH_LOWER, MONTH_UPPER,
    day_variant, exact_match, exact_match_by_group, field_accuracy, month_variant,
)
from model import DecoderOnlyTransformer, ModelConfig
from tokenizer import CharTokenizer

# Modo --quick: subconjunto pequeno e poucas épocas, só para verificar que tudo roda (minutos).
QUICK_TRAIN_SIZE, QUICK_VAL_SIZE, QUICK_EPOCHS = 2_000, 500, 3

# Tamanho da amostra OOD acompanhada a cada época (a avaliação final usa as 18.262 datas).
OOD_MONITOR_SIZE = 2_500


def set_seed(seed: int) -> None:
    """Fixa as fontes de aleatoriedade (inicialização dos pesos, dropout, ordem dos batches)."""
    random.seed(seed)
    torch.manual_seed(seed)


def setup_logger(log_path: Path) -> logging.Logger:
    """Logger que escreve ao mesmo tempo no terminal e em train.log."""
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    for handler in (logging.StreamHandler(), logging.FileHandler(log_path, mode="w", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def build_optimizer(model: DecoderOnlyTransformer, cfg: dict) -> torch.optim.AdamW:
    """AdamW com dois grupos de parâmetros: com e sem weight decay.

    Weight decay só nas matrizes das camadas lineares. Ficam sem decay:
      - bias e γ/β da LayerNorm (vetores: puxá-los para 0 não regulariza nada útil
        e atrapalha a normalização);
      - a matriz de embeddings E (compartilhada com a saída): o decay encolheria
        os embeddings, que já têm escala controlada (× √d_model).
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if param.dim() >= 2 and "embedding" not in name:
            decay.append(param)
        else:
            no_decay.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg["lr"],
        betas=(0.9, 0.98),  # valores do paper
        eps=1e-9,           # valor do paper
    )


def lr_at_step(step: int, cfg: dict, total_steps: int) -> float:
    """LR no passo `step` (começando em 1): warmup linear, depois decaimento cosseno.

        passos 1..warmup:     sobe linearmente de ~0 até lr
        passos warmup..total: desce em forma de cosseno de lr até min_lr
    """
    peak, min_lr, warmup = cfg["lr"], cfg["min_lr"], cfg["warmup_steps"]
    if step <= warmup:
        return peak * step / warmup

    # progress vai de 0 (fim do warmup) a 1 (último passo possível, em max_epochs).
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    # cos vai de 1 a -1, logo (1 + cos) / 2 vai de 1 a 0.
    return min_lr + (peak - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def compute_loss(model: DecoderOnlyTransformer, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-Entropy média sobre as posições com label válido (as -100 são ignoradas)."""
    logits = model(inputs)  # (B, T, |V|)
    # cross_entropy espera (N, |V|) e (N,): achata batch e posições em uma dimensão só.
    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=IGNORE_INDEX)


@torch.no_grad()
def evaluate_loss(model: DecoderOnlyTransformer, loader: DataLoader) -> float:
    """Loss média por token (teacher forcing, sem dropout) em um split inteiro."""
    was_training = model.training
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for inputs, labels in loader:
        n_tokens = int((labels != IGNORE_INDEX).sum())
        # compute_loss devolve a média do batch; multiplica de volta para somar por token.
        total_loss += compute_loss(model, inputs, labels).item() * n_tokens
        total_tokens += n_tokens
    model.train(was_training)
    return total_loss / total_tokens


def generation_metrics(model: DecoderOnlyTransformer, examples: list[tuple[str, str]],
                       tokenizer: CharTokenizer) -> dict[str, float]:
    """Gera as datas (greedy) e calcula o exact match geral, por variante e por campo (ano/mês/dia).

    Usada a cada época na validação e na amostra OOD.
    """
    sources = [src for src, _ in examples]
    targets = [tgt for _, tgt in examples]
    predictions = generate_dates(model, sources, tokenizer)
    correct = [p == t for p, t in zip(predictions, targets)]

    by_day = exact_match_by_group(correct, [day_variant(s) for s in sources])
    by_month = exact_match_by_group(correct, [month_variant(s) for s in sources])
    # Acurácia de cada campo: mostra a ORDEM em que o modelo aprende cada parte da tarefa
    # (copiar dia e ano × traduzir o nome do mês para o número).
    fields = field_accuracy(predictions, targets)

    def group_em(groups: dict, name: str) -> float:
        # nan se o grupo não aparece no conjunto (só acontece em subconjuntos pequenos, ex.: --quick).
        return groups[name]["exact_match"] if name in groups else float("nan")

    return {
        "exact_match": exact_match(predictions, targets),
        "em_day_1digit": group_em(by_day, DAY_1_DIGIT),
        "em_day_2digits": group_em(by_day, DAY_2_DIGITS),
        "em_month_upper": group_em(by_month, MONTH_UPPER),
        "em_month_lower": group_em(by_month, MONTH_LOWER),
        "year_accuracy": fields["year"],
        "month_accuracy": fields["month"],
        "day_accuracy": fields["day"],
    }


def save_checkpoint(path: Path, model: DecoderOnlyTransformer, epoch: int, val_loss: float) -> None:
    """Salva o checkpoint no formato lido por generate.load_model."""
    torch.save(
        {
            "model_config": asdict(model.config),
            "model_state": model.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--quick", action="store_true", help="teste local: poucos dados e poucas épocas")
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as f:
        config = json.load(f)
    train_cfg = config["train"]
    if args.quick:
        train_cfg["max_epochs"] = QUICK_EPOCHS

    # ---------------- Ambiente ----------------
    # Threads de cálculo do PyTorch (paralelismo dentro de cada operação, ex.: multiplicação
    # de matrizes). interop = 1: não roda várias operações em paralelo, evitando que o
    # processo use mais núcleos do que os reservados para ele.
    torch.set_num_threads(train_cfg["threads"])
    torch.set_num_interop_threads(1)
    set_seed(train_cfg["seed"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.out_dir / "train.log")

    # ---------------- Dados ----------------
    tokenizer = CharTokenizer.load(args.data_dir / "vocab.json")
    train_examples = read_examples(args.data_dir / "train.csv")
    val_examples = read_examples(args.data_dir / "val.csv")
    if args.quick:
        train_examples, val_examples = train_examples[:QUICK_TRAIN_SIZE], val_examples[:QUICK_VAL_SIZE]

    # Amostra OOD fixa (mesma seed => mesmas 2.500 datas em todos os experimentos e épocas).
    # Gerador próprio, para não alterar a sequência aleatória do treino.
    ood_examples = read_examples(args.data_dir / "test_ood.csv")
    ood_monitor = random.Random(train_cfg["seed"]).sample(
        ood_examples, QUICK_VAL_SIZE if args.quick else OOD_MONITOR_SIZE
    )

    max_seq_len = config["model"]["max_seq_len"]
    train_set = DateDataset(train_examples, tokenizer, max_seq_len)
    val_set = DateDataset(val_examples, tokenizer, max_seq_len)

    collate = functools.partial(collate_batch, pad_id=tokenizer.pad_id)
    # Ordem dos batches embaralhada a cada época, com gerador seedado (reprodutível).
    # num_workers=0: o carregamento roda no processo principal; workers extras
    # disputariam os núcleos reservados para o cálculo.
    train_loader = DataLoader(
        train_set, batch_size=train_cfg["batch_size"], shuffle=True, collate_fn=collate,
        generator=torch.Generator().manual_seed(train_cfg["seed"]),
    )
    val_loader = DataLoader(val_set, batch_size=512, shuffle=False, collate_fn=collate)

    # ---------------- Modelo e otimizador ----------------
    model = DecoderOnlyTransformer(ModelConfig(vocab_size=tokenizer.vocab_size, **config["model"]))
    optimizer = build_optimizer(model, train_cfg)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * train_cfg["max_epochs"]

    # Registro da configuração e do ambiente (para reprodução e comparação).
    run_info = {
        "config": config,
        "quick": args.quick,
        "n_parameters": model.count_parameters(),
        "vocab_size": tokenizer.vocab_size,
        "n_train": len(train_set),
        "n_val": len(val_set),
        "n_ood_monitor": len(ood_monitor),
        "steps_per_epoch": steps_per_epoch,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "num_threads": torch.get_num_threads(),
            # CPUs em que o processo pode rodar (confirma a afinidade definida pelo taskset).
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
        },
    }
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)
    logger.info(f"{config['name']}: {run_info['n_parameters']:,} parâmetros | "
                f"{len(train_set)} treino / {len(val_set)} validação | {steps_per_epoch} passos por época | "
                f"threads={torch.get_num_threads()} | CPUs={run_info['environment']['cpu_affinity']}")

    # ---------------- Loop de treino ----------------
    history_path = args.out_dir / "history.csv"
    history_fields = [
        "epoch", "train_loss", "val_loss", "val_exact_match",
        # exact match de validação por variante de formatação
        "val_em_day_1digit", "val_em_day_2digits", "val_em_month_upper", "val_em_month_lower",
        # acurácia de validação por campo da data ISO
        "val_year_accuracy", "val_month_accuracy", "val_day_accuracy",
        # monitoramento OOD (amostra fixa; não usado em nenhuma decisão)
        "ood_monitor_exact_match", "ood_monitor_year_accuracy",
        "lr", "epoch_time_s", "train_tokens_per_s",
    ]
    with history_path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(history_fields)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    step = 0
    stopped_early = False
    train_start = time.perf_counter()

    for epoch in range(1, train_cfg["max_epochs"] + 1):
        epoch_start = time.perf_counter()
        model.train()  # liga o dropout
        loss_sum, loss_tokens, input_tokens = 0.0, 0, 0

        for inputs, labels in train_loader:
            step += 1
            # LR do passo atual (warmup + cosseno), aplicado aos dois grupos do otimizador.
            lr = lr_at_step(step, train_cfg, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            loss = compute_loss(model, inputs, labels)

            optimizer.zero_grad()  # zera os gradientes do passo anterior
            loss.backward()        # backpropagation: calcula os gradientes
            # Clipping: se a norma global dos gradientes passar de grad_clip, reduz todos
            # proporcionalmente (evita passos gigantes, comuns no início com Post-LN).
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            optimizer.step()       # atualiza os pesos

            # Acumuladores para a média da época e o throughput.
            n_tokens = int((labels != IGNORE_INDEX).sum())
            loss_sum += loss.item() * n_tokens
            loss_tokens += n_tokens
            input_tokens += int((inputs != tokenizer.pad_id).sum())

        train_time = time.perf_counter() - epoch_start
        # Obs.: train_loss é medida com dropout ligado e com os pesos mudando ao longo da
        # época; val_loss usa os pesos do fim da época, sem dropout. Por isso a val_loss
        # pode ficar abaixo da train_loss, principalmente no início.
        train_loss = loss_sum / loss_tokens

        # Validação: loss (teacher forcing) e métricas de geração (greedy, sem teacher forcing).
        val_loss = evaluate_loss(model, val_loader)
        val = generation_metrics(model, val_examples, tokenizer)
        # Monitoramento OOD: só registrado, nunca usado para decidir nada.
        ood = generation_metrics(model, ood_monitor, tokenizer)
        epoch_time = time.perf_counter() - epoch_start

        # Losses em notação científica (ex.: 3.123457e-05): perto do fim do treino elas ficam
        # muito pequenas, e casas decimais fixas virariam 0.000000 (quebrando o gráfico em escala log).
        with history_path.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{train_loss:.6e}", f"{val_loss:.6e}", f"{val['exact_match']:.4f}",
                f"{val['em_day_1digit']:.4f}", f"{val['em_day_2digits']:.4f}",
                f"{val['em_month_upper']:.4f}", f"{val['em_month_lower']:.4f}",
                f"{val['year_accuracy']:.4f}", f"{val['month_accuracy']:.4f}", f"{val['day_accuracy']:.4f}",
                f"{ood['exact_match']:.4f}", f"{ood['year_accuracy']:.4f}",
                f"{lr:.2e}", f"{epoch_time:.1f}", f"{input_tokens / train_time:.0f}",
            ])

        # Early stopping: melhora estrita da menor val_loss vista até agora.
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss, best_epoch = val_loss, epoch
            epochs_without_improvement = 0
            save_checkpoint(args.out_dir / "best.pt", model, epoch, val_loss)
        else:
            epochs_without_improvement += 1

        logger.info(f"época {epoch:>2} | train_loss {train_loss:.3e} | val_loss {val_loss:.3e} | "
                    f"val_EM {val['exact_match']:.4f} (dia 1 díg. {val['em_day_1digit']:.3f}, "
                    f"mês maiúsc. {val['em_month_upper']:.3f}) | OOD_EM {ood['exact_match']:.4f} | "
                    f"lr {lr:.2e} | {epoch_time:.1f}s" + (" | *melhor*" if improved else ""))

        if epochs_without_improvement >= train_cfg["early_stopping_patience"]:
            stopped_early = True
            logger.info(f"early stopping: {epochs_without_improvement} épocas sem melhora da val_loss")
            break

    total_time = time.perf_counter() - train_start
    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epochs_run": epoch,
        "stopped_early": stopped_early,
        "total_steps": step,
        "train_time_s": round(total_time, 1),
    }
    with (args.out_dir / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"fim: melhor época {best_epoch} (val_loss {best_val_loss:.3e}) | {total_time:.0f}s no total")


if __name__ == "__main__":
    main()
