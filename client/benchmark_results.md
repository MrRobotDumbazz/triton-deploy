# Benchmark: triton-deploy — resnet50, yolov5 (before/after)

- Дата (UTC): 2026-11-10 04:05:34
- CPU: AMD Ryzen 7 7730U with Radeon Graphics (16 ядер)
- Triton: nvcr.io/nvidia/tritonserver:23.12-py3 (образ из контракта B1); server 2.41.0; localhost:8000
- Клиент: tritonclient[http], бинарная передача; латентность — клиентские часы per request
- Примечание: before = v1 (fp32), after = v2 (int8) для resnet50; yolov5 — v1 fp32. warmup=5, итераций (последний прогон, yolov5)=20

| модель | версия | batch | avg, мс | p50, мс | p95, мс | req/s |
|---|---|---|---|---|---|---|
| resnet50 | v1 (before, fp32) | 1 | 63.31 | 56.65 | 96.09 | 15.74 |
| resnet50 | v1 (before, fp32) | 2 | 105.50 | 102.99 | 147.41 | 9.45 |
| resnet50 | v1 (before, fp32) | 4 | 147.56 | 142.38 | 177.81 | 6.76 |
| resnet50 | v1 (before, fp32) | 8 | 286.06 | 281.89 | 328.00 | 3.49 |
| resnet50 | v1 (before, fp32) | 16 | 565.20 | 562.77 | 595.76 | 1.76 |
| resnet50 | v2 (after, int8) | 1 | 32.59 | 28.83 | 53.96 | 30.53 |
| resnet50 | v2 (after, int8) | 2 | 61.88 | 56.92 | 84.89 | 16.09 |
| resnet50 | v2 (after, int8) | 4 | 102.38 | 99.96 | 128.62 | 9.72 |
| resnet50 | v2 (after, int8) | 8 | 193.26 | 188.54 | 233.35 | 5.14 |
| resnet50 | v2 (after, int8) | 16 | 350.29 | 349.77 | 375.85 | 2.84 |
| yolov5 | v1 | 1 | 304.25 | 298.35 | 337.26 | 3.27 |
| yolov5 | v1 | 2 | 584.80 | 583.82 | 611.90 | 1.71 |
| yolov5 | v1 | 4 | 1231.36 | 1232.00 | 1267.79 | 0.81 |
| yolov5 | v1 | 8 | 2613.44 | 2619.89 | 2656.13 | 0.38 |
