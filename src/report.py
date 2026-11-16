"""
Análise local dos experimentos: tabela comparativa e gráficos (roda na máquina local,
depois de dar pull dos resultados do servidor).

Lê de cada runs/expN/: config.json, history.csv, train_summary.json,
metrics_test.json e metrics_test_ood.json. Experimentos sem resultados são pulados.

Saídas:
    runs/report.md                          tabelas comparativas + exemplos de erro
    runs/figures/learning_curves.png        train loss, val loss e val exact match por época
    runs/figures/variant_curves.png         evolução por época: variantes raras e OOD (amostra)
    runs/figures/field_curves.png           evolução por época: acurácia de ano, mês e dia (validação)
    runs/figures/exact_match_by_case.png    exact match por experimento em cada situação

Uso:
    python src/report.py [--runs-dir runs]
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # gera PNG sem precisar de janela/interface gráfica
import matplotlib.pyplot as plt

EXPERIMENTS = ["exp1", "exp2", "exp3"]

# Uma cor fixa por experimento (sempre a mesma em todos os gráficos).
COLORS = {"exp1": "#2a78d6", "exp2": "#eb6834", "exp3": "#1baf7a"}
TEXT_COLOR, MUTED_COLOR, GRID_COLOR, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"


def load_run(run_dir: Path) -> dict | None:
    """Carrega todos os resultados de um experimento (ou None se estiver incompleto)."""
    files = ["config.json", "history.csv", "train_summary.json", "metrics_test.json", "metrics_test_ood.json"]
    if not all((run_dir / f).exists() for f in files):
        return None

    def read_json(name):
        with (run_dir / name).open(encoding="utf-8") as f:
            return json.load(f)

    with (run_dir / "history.csv").open(encoding="utf-8", newline="") as f:
        history = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]

    return {
        "config": read_json("config.json"),
        "history": history,
        "summary": read_json("train_summary.json"),
        "test": read_json("metrics_test.json"),
        "ood": read_json("metrics_test_ood.json"),
    }


def label(name: str, run: dict) -> str:
    """Rótulo legível: 'exp1 (d=32, L=2)'."""
    model = run["config"]["config"]["model"]
    return f"{name} (d={model['d_model']}, L={model['n_layers']})"


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


# ---------------------------------------------------------------- tabelas (markdown)

def build_report(runs: dict[str, dict]) -> str:
    lines = ["# Resultados dos experimentos", ""]

    # Tabela principal (a mesma da especificação).
    lines += [
        "## Tabela comparativa", "",
        "| Experimento | Params | Melhor época | Train loss | Val loss | Test loss | Val EM | Test EM | OOD EM | Tempo de treino |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, run in runs.items():
        best = run["summary"]["best_epoch"]
        row = next(r for r in run["history"] if int(r["epoch"]) == best)  # métricas da melhor época
        lines.append(
            f"| {label(name, run)} | {run['config']['n_parameters']:,} | {best} / {run['summary']['epochs_run']} "
            f"| {row['train_loss']:.2e} | {row['val_loss']:.2e} | {run['test']['loss']:.2e} "
            f"| {pct(row['val_exact_match'])} | {pct(run['test']['exact_match'])} | {pct(run['ood']['exact_match'])} "
            f"| {run['summary']['train_time_s'] / 60:.1f} min |"
        )
    lines += ["", "Melhor época = época de menor val loss / épocas rodadas. EM = exact match. "
              "Tempo de treino = soma das épocas, incluindo a validação de cada época (loss + geração).", ""]

    # Diagnóstico: onde cada modelo erra (teste e OOD).
    for split, title in [("test", "Teste (1800–2049)"), ("ood", "Teste OOD (2050–2099)")]:
        lines += [
            f"## Diagnóstico: {title}", "",
            "| Experimento | EM | Acurácia por caractere | Ano | Mês | Dia | Saídas válidas "
            "| Dia 1–9 com 1 dígito | Dia 1–9 com 2 dígitos | Mês maiúsculo | Mês minúsculo |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name, run in runs.items():
            m = run[split]
            day, month, fields = m["exact_match_by_day_format"], m["exact_match_by_month_format"], m["field_accuracy"]
            lines.append(
                f"| {name} | {pct(m['exact_match'])} | {pct(m['char_accuracy'])} | {pct(fields['year'])} "
                f"| {pct(fields['month'])} | {pct(fields['day'])} | {pct(m['valid_output_rate'])} "
                f"| {pct(day['dia 1-9 com 1 dígito']['exact_match'])} | {pct(day['dia 1-9 com 2 dígitos']['exact_match'])} "
                f"| {pct(month['mês maiúsculo']['exact_match'])} | {pct(month['mês minúsculo']['exact_match'])} |"
            )
        lines.append("")

    # Exact match por mês (teste), para achar meses problemáticos.
    months = sorted(next(iter(runs.values()))["test"]["exact_match_by_month"])
    lines += ["## Exact match por mês (teste)", "",
              "| Experimento | " + " | ".join(months) + " |",
              "|---|" + "---:|" * len(months)]
    for name, run in runs.items():
        by_month = run["test"]["exact_match_by_month"]
        lines.append(f"| {name} | " + " | ".join(pct(by_month[m]["exact_match"]) for m in months) + " |")
    lines.append("")

    # Exemplos de erro.
    lines += ["## Exemplos de erro", ""]
    for name, run in runs.items():
        for split in ["test", "ood"]:
            errors = run[split]["error_examples"][:5]
            lines.append(f"**{name}, {'teste' if split == 'test' else 'OOD'}** ({run[split]['n_errors']} erros no total)")
            lines.append("")
            lines += [f"- `{e['source']}` → `{e['prediction']}` (correto: `{e['target']}`)" for e in errors] or ["- nenhum erro"]
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------- gráficos (PNG)

def style_axis(ax, title: str, ylabel: str, grid_axis: str = "both") -> None:
    """Estilo discreto: grade e eixos claros, para os dados se destacarem."""
    ax.set_title(title, loc="left", fontsize=11, color=TEXT_COLOR)
    ax.set_ylabel(ylabel, color=MUTED_COLOR)
    ax.set_xlabel("época", color=MUTED_COLOR)
    ax.grid(True, axis=grid_axis, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=MUTED_COLOR)
    ax.set_facecolor(SURFACE)


def plot_learning_curves(runs: dict[str, dict], path: Path) -> None:
    """Três painéis lado a lado (cada um com um único eixo y): train loss, val loss, val EM."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor=SURFACE)
    panels = [
        ("train_loss", "Train loss (escala log)", "cross-entropy", True),
        ("val_loss", "Val loss (escala log)", "cross-entropy", True),
        ("val_exact_match", "Val exact match", "exact match", False),
    ]
    for ax, (key, title, ylabel, log_scale) in zip(axes, panels):
        for name, run in runs.items():
            epochs = [r["epoch"] for r in run["history"]]
            values = [r[key] for r in run["history"]]
            ax.plot(epochs, values, color=COLORS[name], linewidth=2, label=label(name, run))
            # Marca a melhor época (menor val loss), onde o checkpoint foi salvo.
            best = run["summary"]["best_epoch"]
            ax.plot(best, values[best - 1], "o", color=COLORS[name], markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=2)
        if log_scale:
            ax.set_yscale("log")
        style_axis(ax, title, ylabel)

    axes[0].legend(frameon=False, labelcolor=TEXT_COLOR)
    fig.text(0.01, 0.01, "● = melhor época (menor val loss, checkpoint salvo)", color=MUTED_COLOR, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_variant_curves(runs: dict[str, dict], path: Path) -> None:
    """Evolução, por época, do exact match nas variantes raras e no OOD (amostra).

    Nos dois primeiros painéis: linha cheia = variante rara; tracejada = forma comum
    dos mesmos casos (referência), com a mesma cor do experimento.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor=SURFACE)
    panels = [
        ("Val: dia 1–9 com 1 dígito (cheia) × 2 dígitos (tracejada)", "val_em_day_1digit", "val_em_day_2digits"),
        ("Val: mês maiúsculo (cheia) × minúsculo (tracejada)", "val_em_month_upper", "val_em_month_lower"),
        ("OOD (amostra fixa de 2.500 datas de 2050–2099)", "ood_monitor_exact_match", None),
    ]
    for ax, (title, key, reference_key) in zip(axes, panels):
        for name, run in runs.items():
            epochs = [r["epoch"] for r in run["history"]]
            ax.plot(epochs, [r[key] for r in run["history"]], color=COLORS[name], linewidth=2, label=label(name, run))
            if reference_key:
                ax.plot(epochs, [r[reference_key] for r in run["history"]], color=COLORS[name],
                        linewidth=1.5, linestyle="--", alpha=0.8)
        ax.set_ylim(-0.02, 1.02)
        style_axis(ax, title, "exact match")
        ax.title.set_fontsize(10)

    axes[0].legend(frameon=False, labelcolor=TEXT_COLOR, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_field_curves(runs: dict[str, dict], path: Path) -> None:
    """Acurácia de validação de cada campo (ano, mês, dia) por época: em que ordem o modelo aprende cada parte."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor=SURFACE)
    panels = [("Val: acurácia do ano", "val_year_accuracy"),
              ("Val: acurácia do mês", "val_month_accuracy"),
              ("Val: acurácia do dia", "val_day_accuracy")]
    for ax, (title, key) in zip(axes, panels):
        for name, run in runs.items():
            epochs = [r["epoch"] for r in run["history"]]
            ax.plot(epochs, [r[key] for r in run["history"]], color=COLORS[name], linewidth=2, label=label(name, run))
        ax.set_ylim(-0.02, 1.02)
        style_axis(ax, title, "acurácia")

    axes[0].legend(frameon=False, labelcolor=TEXT_COLOR, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_exact_match_by_case(runs: dict[str, dict], path: Path) -> None:
    """Barras agrupadas: exact match de cada experimento em cada situação de avaliação."""
    cases = [
        ("Teste", lambda r: r["test"]["exact_match"]),
        ("Teste:\ndia com 1 dígito", lambda r: r["test"]["exact_match_by_day_format"]["dia 1-9 com 1 dígito"]["exact_match"]),
        ("Teste:\nmês maiúsculo", lambda r: r["test"]["exact_match_by_month_format"]["mês maiúsculo"]["exact_match"]),
        ("OOD\n(2050–2099)", lambda r: r["ood"]["exact_match"]),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=SURFACE)
    width = 0.8 / len(runs)
    for i, (name, run) in enumerate(runs.items()):
        xs = [c + (i - (len(runs) - 1) / 2) * width for c in range(len(cases))]
        values = [100 * get(run) for _, get in cases]
        # Borda da cor do fundo: cria um pequeno espaço entre barras vizinhas.
        bars = ax.bar(xs, values, width, color=COLORS[name], edgecolor=SURFACE, linewidth=2, label=label(name, run))
        ax.bar_label(bars, fmt="%.1f", fontsize=8, color=TEXT_COLOR, padding=2)

    ax.set_xticks(range(len(cases)), [c for c, _ in cases])
    ax.set_ylim(0, 108)
    style_axis(ax, "Exact match (%) por situação de avaliação", "exact match (%)", grid_axis="y")
    ax.set_xlabel("")
    # Legenda abaixo do gráfico, em uma linha, para não cobrir as barras.
    ax.legend(frameon=False, labelcolor=TEXT_COLOR, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=len(runs))
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")  # "tight" inclui a legenda externa na imagem
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()

    runs = {}
    for name in EXPERIMENTS:
        run = load_run(args.runs_dir / name)
        if run is None:
            print(f"{name}: resultados incompletos ou ausentes, pulando")
        else:
            runs[name] = run
    if not runs:
        raise SystemExit("nenhum experimento com resultados completos")

    report = build_report(runs)
    (args.runs_dir / "report.md").write_text(report, encoding="utf-8")

    figures_dir = args.runs_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    plot_learning_curves(runs, figures_dir / "learning_curves.png")
    plot_variant_curves(runs, figures_dir / "variant_curves.png")
    plot_field_curves(runs, figures_dir / "field_curves.png")
    plot_exact_match_by_case(runs, figures_dir / "exact_match_by_case.png")

    print(f"relatório: {args.runs_dir / 'report.md'}")
    print(f"gráficos:  {figures_dir}/")


if __name__ == "__main__":
    main()
