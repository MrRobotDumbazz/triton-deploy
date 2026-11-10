#!/usr/bin/env bash
# Управление стеком triton-deploy (зона deployment/, W-2).
# Стек: triton (CPU, бэкенд onnxruntime — GPU/TensorRT не применяются, I54)
#       + prometheus + grafana. См. deployment/docker-compose.yml.
#
# Использование:
#   bash deployment/run_triton.sh            # build + up -d + ожидание готовности + список моделей
#   bash deployment/run_triton.sh --smoke    # smoke-тест: инференс resnet50 (внутри контейнера)
#   bash deployment/run_triton.sh --bench    # бенчмарк before/after (внутри контейнера)
#   bash deployment/run_triton.sh --logs     # логи triton (follow, Ctrl-C для выхода)
#   bash deployment/run_triton.sh --down     # остановить и убрать стек
set -euo pipefail

# --- Переход в корень репо (compose-пути и bind-тома отсчитываются от него/от deployment/) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"; then
    cd "$REPO_ROOT"
else
    # Фолбэк (громко, I54): вне git-клона предполагаем штатную раскладку ../
    echo "ВНИМАНИЕ: git-репозиторий не обнаружен — перехожу в '${SCRIPT_DIR}/..' как в корень проекта." >&2
    cd "${SCRIPT_DIR}/.."
fi

COMPOSE_FILE="deployment/docker-compose.yml"
TRITON_READY_URL="http://localhost:8000/v2/health/ready"

die() { echo "ОШИБКА: $*" >&2; exit 1; }

require_docker() {
    command -v docker >/dev/null 2>&1 \
        || die "docker не найден в PATH. Установите Docker: https://docs.docker.com/get-docker/"
    docker info >/dev/null 2>&1 \
        || die "демон Docker недоступен ('docker info' не прошёл). Запустите сервис (sudo systemctl start docker) или Docker Desktop."
}

compose() {
    docker compose -f "$COMPOSE_FILE" "$@"
}

wait_triton_ready() {
    command -v curl >/dev/null 2>&1 \
        || die "curl не найден — не могу дождаться готовности Triton (${TRITON_READY_URL})."
    echo "Жду готовности Triton (${TRITON_READY_URL}); таймаут 120s (60 попыток x 2s)..."
    local attempt
    for attempt in $(seq 1 60); do
        if curl -sf "$TRITON_READY_URL" >/dev/null; then
            echo "Triton готов (попытка №${attempt})."
            return 0
        fi
        sleep 2
    done
    # Громкий фолбэк (I54): не молчим — показываем хвост логов и что делать.
    echo "ОШИБКА (I54): Triton не стал готов за 120s." >&2
    compose logs --tail=50 triton >&2 || true
    die "Посмотрите логи целиком: bash deployment/run_triton.sh --logs"
}

cmd_up() {
    require_docker
    echo "[1/3] Собираю образ triton (CPU, onnxruntime)..."
    compose build
    echo "[2/3] Запускаю стек (triton + prometheus + grafana)..."
    compose up -d
    echo "[3/3] Ожидаю готовности..."
    wait_triton_ready
    echo
    echo "=== Модели в репозитории Triton (/v2/repository/index) ==="
    if curl -s -X POST http://localhost:8000/v2/repository/index | python3 -m json.tool; then
        :
    else
        echo "ВНИМАНИЕ (I54): список моделей получить не удалось — сервер не ответил на /v2/repository/index." >&2
    fi
    echo
    cat <<'HINTS'
=== Стек поднят. Эндпоинты ===
  Triton HTTP    : http://localhost:8000   (готовность: /v2/health/ready)
  Triton gRPC    : localhost:8001
  Triton metrics : http://localhost:8002/metrics
  Prometheus     : http://localhost:9090
  Grafana        : http://localhost:3000   (admin / ${GRAFANA_ADMIN_PASSWORD:-admin} из .env)

=== Дальнейшие шаги ===
  bash deployment/run_triton.sh --smoke   # smoke-тест: инференс resnet50
  bash deployment/run_triton.sh --bench   # бенчмарк before/after (v1 fp32 vs v2 int8)
  bash deployment/run_triton.sh --logs    # логи triton
  bash deployment/run_triton.sh --down    # остановить стек
HINTS
}

cmd_down() {
    require_docker
    compose down
    echo "Стек остановлен."
}

cmd_logs() {
    require_docker
    exec compose logs -f triton
}

cmd_bench() {
    require_docker
    # Бенчмарк выполняется внутри контейнера triton (клиент скопирован в /workspace/client).
    compose exec -T triton python /workspace/client/benchmark.py --url localhost:8000
}

cmd_smoke() {
    require_docker
    # Smoke-тест выполняется внутри контейнера triton.
    compose exec -T triton python /workspace/client/inference_client.py --model resnet50 --url localhost:8000
}

usage() {
    sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

case "${1:---up}" in
    --up|up)       cmd_up ;;
    --down|down)   cmd_down ;;
    --logs|logs)   cmd_logs ;;
    --bench|bench) cmd_bench ;;
    --smoke|smoke) cmd_smoke ;;
    -h|--help|help) usage ;;
    *) die "неизвестная подкоманда '$1'. Допустимо: (пусто) | --up | --down | --logs | --bench | --smoke | --help" ;;
esac
