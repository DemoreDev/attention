#!/usr/bin/env bash
# Roda os 3 experimentos EM PARALELO (cada um nos seus 5 núcleos físicos) e espera todos terminarem.
#
# Antes de treinar, roda os testes (pytest): se algum falhar, nada é treinado.
# A saída de cada experimento vai para runs/expN/run.log.
#
# A saída geral aparece no terminal e também fica salva em runs/run_all.log.
#
# Uso no servidor (na raiz do projeto), dentro de uma sessão tmux (continua rodando se a
# conexão SSH cair; reconectar com "tmux attach"):
#     scripts/run_all.sh
# Acompanhar um experimento específico: tail -f runs/exp1/run.log
set -euo pipefail
cd "$(dirname "$0")/.."

# O git não versiona pastas vazias: runs/ pode não existir logo após o clone/pull.
mkdir -p runs
# A partir daqui, tudo o que o script imprime vai para o terminal E para runs/run_all.log.
exec > >(tee runs/run_all.log) 2>&1

echo "[$(date '+%F %T')] rodando os testes"
.venv/bin/pytest -q

# Lança cada experimento em segundo plano (&) e guarda o PID (identificador do processo).
declare -A PIDS
for NAME in exp1 exp2 exp3; do
    mkdir -p "runs/$NAME"
    "scripts/run_$NAME.sh" > "runs/$NAME/run.log" 2>&1 &
    PIDS[$NAME]=$!
    echo "[$(date '+%F %T')] $NAME iniciado (PID ${PIDS[$NAME]}, log em runs/$NAME/run.log)"
done

# Espera cada um e registra se terminou com sucesso, sem interromper os outros em caso de falha.
FAILED=0
for NAME in exp1 exp2 exp3; do
    if wait "${PIDS[$NAME]}"; then
        echo "[$(date '+%F %T')] $NAME concluído"
    else
        echo "[$(date '+%F %T')] $NAME FALHOU (ver runs/$NAME/run.log)"
        FAILED=1
    fi
done

exit $FAILED
