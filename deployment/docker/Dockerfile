# Triton Inference Server — образ для triton-deploy.
# ВАЖНО (громко): среда разработки CPU-only. Бэкенд моделей — onnxruntime;
# TensorRT НЕ применяется (требует NVIDIA GPU) и здесь не обещается.
FROM nvcr.io/nvidia/tritonserver:23.12-py3

# Клиентские зависимости для скриптов внутри контейнера: smoke-тест и бенчмарк
# запускаются через `docker compose exec` из /workspace/client (см. run_triton.sh).
RUN pip install --no-cache-dir "tritonclient[http]" pillow numpy

# Репозиторий моделей: build-time COPY, не volume (контракт ports.md — воспроизводимость).
COPY models /models

# Клиент (inference_client.py / benchmark.py) — внутрь контейнера для --smoke/--bench.
COPY client /workspace/client

# 8000 — HTTP (REST /v2/...), 8001 — gRPC, 8002 — метрики Prometheus (/metrics).
EXPOSE 8000 8001 8002

# Готовность: GET /v2/health/ready. Щедрые интервалы: при старте Triton загружает
# обе модели (resnet50 fp32+int8, yolov5), на CPU это может занять минуты.
HEALTHCHECK --interval=60s --timeout=30s --retries=3 --start-period=120s \
    CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/v2/health/ready', timeout=3).status==200 else 1)"

# ENTRYPOINT намеренно НЕ задаём: команду определяет docker compose (command сервиса).
# Дефолт ниже — только для `docker run` без аргументов.
CMD ["tritonserver", "--model-repository=/models"]
