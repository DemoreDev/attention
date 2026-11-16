#!/usr/bin/env bash
# Exp. 3 (d_model=128, n_layers=4): treino + avaliação final (teste e OOD).
# Núcleos físicos 10-14 do socket 1 (CPUs lógicas, ver "Alocação de CPU" na especificação).
#
# Uso (na raiz do projeto): scripts/run_exp3.sh
set -euo pipefail                # interrompe no primeiro erro
cd "$(dirname "$0")/.."          # roda sempre a partir da raiz do projeto

NAME=exp3
CPUS=10-14

mkdir -p "runs/$NAME"
# taskset -c: o processo (e todas as suas threads) só pode rodar nesses núcleos.
# A memória fica no nó NUMA desses núcleos pela política "first-touch" do Linux.
taskset -c "$CPUS" .venv/bin/python src/train.py --config "configs/$NAME.json" --out-dir "runs/$NAME"
taskset -c "$CPUS" .venv/bin/python src/evaluate.py --run-dir "runs/$NAME"
