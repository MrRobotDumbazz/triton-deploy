# triton-deploy

Учебный проект: продакшн-подобный деплой CV-моделей на **NVIDIA Triton Inference Server** в Docker — с оптимизацией инференса (ONNX), мониторингом (Prometheus + Grafana), логированием запросов и версионированием моделей.

Роль-плей из задания: DevOps-команда производственной компании поднимает сервис классификации (ResNet50, ImageNet-1000) и детекции (YOLOv5s, COCO-80) с контролем латентности и throughput.

## Возможности

- **Serving**: Triton (23.12) в Docker, две pre-trained CV-модели, REST и gRPC эндпоинты.
- **Оптимизация**: ONNX-бэкенд; ResNet50 в двух версиях — v1 fp32 (baseline) и v2 int8 (динамическая квантизация onnxruntime); dynamic batching на обеих моделях; бенчмарк до/после.
- **DevOps**: один `Dockerfile` + `docker-compose` (triton + prometheus + grafana), воспроизводимо на любой машине.
- **Мониторинг**: метрики Triton в Prometheus, готовый дашборд Grafana, алерты на высокую задержку и ошибочные запросы.
- **Логирование**: CSV-журнал запросов/ответов; история инференса.
- **Версионирование**: две версии ResNet50 в репозитории моделей, переключение без даунтайма.

## Быстрый старт

Требования: Docker (CUDA не обязателен), Python 3.10+.

```bash
pip install -r requirements.txt          # клиентские зависимости (хост)
bash deployment/run_triton.sh            # build + up + ожидание готовности
```

Скрипт поднимет три сервиса: **triton** (8000 REST / 8001 gRPC / 8002 метрики), **prometheus** (9090), **grafana** (3000, admin / `$GRAFANA_ADMIN_PASSWORD` из `.env`, дефолт `admin`).

Проверка, что модели загрузились:

```bash
curl -s http://localhost:8000/v2/repository/index | python3 -m json.tool
```

## Запуск инференса

Клиент сам делает препроцессинг по контракту моделей (resize/нормализация для ResNet50, letterbox для YOLOv5):

```bash
# классификация: топ-5 ImageNet по всем картинкам из client/samples/
python client/inference_client.py --model resnet50

# детекция: боксы в координатах оригинала
python client/inference_client.py --model yolov5

# конкретная версия модели и свои картинки
python client/inference_client.py --model resnet50 --version 1 \
    --images client/samples/img_01.jpg --top-k 5

# батч-режим (один запрос на N картинок, N ≤ 32 для resnet50)
python client/inference_client.py --model resnet50 --batch 4
```

Каждый запрос логируется в `client/inference_log.csv` (бонус-требование):

```
timestamp_utc,model,version,image,batch,latency_ms,ok,prediction_summary
2026-11-09T10:00:00Z,resnet50,1,img_01.jpg,1,18.4,1,"Labrador retriever 41.2%; golden retriever 12.0%; ..."
```

Отключить лог: `--no-log`. Адрес сервера — `--url` или переменная `TRITON_URL`.

## Бенчмарк до/после оптимизации

```bash
python client/benchmark.py --model resnet50            # v1 vs v2, batch 1..16
python client/benchmark.py --model yolov5              # v1, batch 1..8
```

«До» = resnet50 **v1 fp32**, «после» = **v2 int8** (динамическая квантизация весов) + эффект dynamic batching на растущих партиях. Результаты — в `client/benchmark_results.md` (обновляется прогоном; числа только из реальных запусков). Интерпретация — в разделе «Результаты оптимизации» ниже.

## Версионирование моделей

Triton хранит версии в нумерованных подкаталогах `models/<model>/<version>/model.onnx`:

```
models/resnet50/
├── config.pbtxt          # version_policy { latest { num_versions: 2 } }
├── 1/model.onnx          # v1 — fp32 baseline
└── 2/model.onnx          # v2 — int8 (динамическая квантизация)
```

- **Кто обслуживается**: `version_policy latest {num_versions: 2}` — обе версии загружены, запросы без версии идут в старшую (v2).
- **Запрос к конкретной версии**: `python client/inference_client.py --model resnet50 --version 1` (в REST это `POST /v2/models/resnet50/versions/1/infer`).
- **Откат/переключение** — правкой `version_policy` (например, `specific { versions: 1 }`) + горячая перезагрузка без рестарта контейнера:

```bash
curl -s -X POST http://localhost:8000/v2/repository/models/resnet50/load
curl -s -X POST http://localhost:8000/v2/repository/models/resnet50/unload   # выгрузить ненужное
```

- **Добавление версии**: положить `models/resnet50/3/model.onnx` и вызвать `load` — Triton подхватит новую версию на лету (`--model-control=poll` также доступен).

## Dynamic batching

Включён в `config.pbtxt` обеих моделей: resnet50 — `preferred_batch_size [4, 8]`, `max_batch_size 32`, очередь 2 мс; yolov5 — `[2, 4]`/`8`, очередь 5 мс. Отдельные запросы склеиваются в батч: latency чуть растёт (ожидание в очереди), throughput — значительно. Эффект виден в бенчмарке: сравните req/s при batch=1 и batch=8.

## Мониторинг: Prometheus + Grafana

- Triton сам экспортирует метрики в формате Prometheus на `:8002/metrics` (`nv_inference_*`: успехи/ошибки запросов, длительности, очереди, батчи).
- `monitoring/prometheus.yml` — scrape каждые 5 с.
- `monitoring/alerts.yml` (бонус) — алерты: `HighInferenceLatency` (средняя задержка запроса > 100 мс на 5-минутном окне) и `FailedRequests` (любой поток ошибок > 2 мин). Видны на вкладке Alerts в Prometheus (`http://localhost:9090/alerts`).
- Grafana (`http://localhost:3000`): дашборд **«Triton Deploy — Inference Monitoring»** провижинится автоматически — throughput по моделям, средняя задержка, compute vs queue, ошибочные запросы, счётчик активных моделей.

Нагрузка для живых графиков:

```bash
for i in $(seq 1 200); do python client/inference_client.py --model resnet50 --no-log >/dev/null; done
```

## Структура репозитория

```
triton-deploy/
├── models/
│   ├── resnet50/            # v1 fp32 + v2 int8, dynamic batching
│   └── yolov5/              # yolov5s v1
├── deployment/
│   ├── docker/Dockerfile    # образ Triton + модели + клиент
│   ├── docker-compose.yml   # triton + prometheus + grafana
│   ├── grafana/             # provisioning (datasource, dashboard)
│   └── run_triton.sh        # build/up/ready/smoke/bench/down
├── monitoring/
│   ├── prometheus.yml
│   ├── alerts.yml           # бонус: правила алертов
│   └── grafana-dashboard.json
├── client/
│   ├── inference_client.py  # REST-клиент обеих моделей, CSV-лог
│   ├── benchmark.py         # замер latency/throughput, v1 vs v2
│   ├── imagenet_classes.txt
│   └── samples/             # 10 тестовых изображений
├── requirements.txt
├── .env.example
└── README.md
```

## CPU/GPU

Стенд этого проекта — **CPU** (GPU в среде разработки нет): TensorRT требует NVIDIA GPU, поэтому оптимизация — ONNX-runtime бэкенд + int8-квантизация и dynamic batching, метрики — CPU. На GPU-машине тот же стек работает без изменений; для TensorRT достаточно положить `model.plan` вместо `model.onnx` (бэкенд tensorrt) — остальная обвязка не меняется.

## Результаты оптимизации

См. `client/benchmark_results.md` (генерируется `client/benchmark.py`, числа — из реального прогона стенда; методология и интерпретация — там же).
