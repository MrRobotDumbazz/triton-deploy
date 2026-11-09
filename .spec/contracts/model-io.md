# Контракт model-io (B1 — единая точка межзонного доверия: models ↔ deployment ↔ client)
# Версия: 1.0 (2026-11-09). Меняет только S-1. Потребители работают от этого файла (моки/тесты — от схем ниже).

## Инстанс Triton
- Образ: `nvcr.io/nvidia/tritonserver:23.12-py3` (CPU-совместим с onnxruntime-бэкендом, GPU-опционален).
- Backend: `onnxruntime` для всех моделей (задание: без GPU — ONNX; TensorRT требует NVIDIA GPU).

## Модель 1: `resnet50` (классификация, ImageNet-1000)
- Репозиторий: `models/resnet50/`, версии `1/` (baseline fp32) и `2/` (оптимизированная int8) — динамическая смена через `version_policy` (README).
- Файл модели: `<version>/model.onnx`, opset ≥ 13, dynamic batch axis.
- **Вход:** `input`: float32, dims `[-1, 3, 224, 224]` (NCHW), препроцессинг клиента: RGB → resize (224×224, bilinear) → /255 → normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]).
- **Выходы:**
  - `probs`: float32 `[-1, 1000]` — softmax-вероятности;
  - `preds`: int64 `[-1, 1]` — argmax-индекс.
- Лейблы: `client/imagenet_classes.txt` (1000 строк; строка i = класс i).
- Dynamic batching: max_batch_size 32, preferred 8, max_queue_delay_microseconds 2000 (CPU-баланс latency/throughput — обоснование в benchmark).

## Модель 2: `yolov5` (детекция, COCO-80; веса yolov5s)
- Репозиторий: `models/yolov5/`, версия `1/`, файл `1/model.onnx` (экспорт ultralytics, opset ≥ 12).
- **Вход:** `images`: float32, dims `[-1, 3, 640, 640]`, диапазон 0..1; препроцессинг клиента: letterbox 640 (pad 114/255).
- **Выход:** `output0`: float32 `[-1, 25200, 85]` — строки [cx, cy, w, h, obj, cls0..cls79], сигмоиды уже в графе.
- Постпроцессинг клиента: conf = obj·max(cls) ≥ 0.25 → NMS(IoU 0.45); классы COCO-80 встроены в клиент.
- Dynamic batching: max_batch_size 8, preferred 4, max_queue_delay_microseconds 5000.
- Имена входов/выходов (`images`, `output0`) — как в экспорте ultralytics v7.0, не переименовывать.

## API-протокол (REST)
- `GET /v2/health/ready` — готовность сервера.
- `POST /v2/models/<model>/versions/<v>/infer` — инференс (JSON, binary_data_optional); без сегмента version — по `version_policy`.
- `POST /v2/repository/models/<model>/load` | `/unload` — горячая перезагрузка (демо versioning).
- Метрики: `GET :8002/metrics` (Prometheus-формат).

## CSV-лог инференса (бонус) — схема
`timestamp_utc,model,version,image,batch,latency_ms,ok,prediction_summary` — append, путь `client/inference_log.csv` (runtime, в git не входит).

## Протокол бенчмарка before/after (Performance Optimization)
- `before` = resnet50 **v1** (fp32), `after` = resnet50 **v2** (квантованная int8) + эффект dynamic batching.
- Партии: [1, 2, 4, 8, 16]; warmup 5; итераций 50 (resnet50) / 20 (yolov5); метрики: avg/p50/p95 latency (мс) и throughput (req/s).
- Инструменты: `client/benchmark.py` (REST, реальный сервер). Числа в отчёте — только из реальных прогонов (E8); невозможность прогона — громкий фолбэк (I54).
