#!/usr/bin/env bash
# Exp. 2 (d_model=64, n_layers=3): treino + avaliação final (teste e OOD).
# Núcleos físicos 5-9 do socket 0 (CPUs lógicas, ver "Alocação de CPU" na especificação).
#
# Uso (na raiz do projeto): scripts/run_exp2.sh
set -euo pipefail                # interrompe no primeiro erro
cd "$(dirname "$0")/.."          # roda sempre a partir da raiz do projeto

NAME=exp2
CPUS=5-9

mkdir -p "runs/$NAME"
# taskset -c: o processo (e todas as suas threads) só pode rodar nesses núcleos.
# A memória fica no nó NUMA desses núcleos pela política "first-touch" do Linux.
taskset -c "$CPUS" .venv/bin/python src/train.py --config "configs/$NAME.json" --out-dir "runs/$NAME"
taskset -c "$CPUS" .venv/bin/python src/evaluate.py --run-dir "runs/$NAME"
