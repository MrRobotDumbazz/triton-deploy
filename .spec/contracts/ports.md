# Контракт портов и переменных (deployment ↔ monitoring ↔ client)
# Версия: 1.0 (2026-11-09). Меняет только S-1.

## Порты (все стеки)
| Сервис | Порт | Назначение |
|---|---|---|
| Triton HTTP | 8000 | inference REST (`/v2/...`) |
| Triton gRPC | 8001 | inference gRPC |
| Triton metrics | 8002 | Prometheus-экспортёр `/metrics` |
| Prometheus | 9090 | UI + rule eval |
| Grafana | 3000 | дашборд (admin/admin, смена через `.env`) |

## Переменные окружения (см. `.env.example`)
- `TRITON_URL` (default `localhost:8000`) — клиент.
- `GRAFANA_ADMIN_PASSWORD` (default `admin`) — только деплой.
- Секретов в репо нет (I52); `.env` в `.gitignore`.

## Имена контейнеров/сервисов (docker-compose)
`triton`, `prometheus`, `grafana`; сеть `triton-net`; том модели — build-time COPY (не volume: воспроизводимость).

## Ограничение среды (громко, I54)
Хост разработки — **CPU-only, без GPU**: TensorRT неприменим (требует NVIDIA GPU) — задание прямо разрешает ONNX-бэкенд на CPU с CPU-метриками. GPU-метрики (DCGM) в мониторинге опциональны и помечены.
