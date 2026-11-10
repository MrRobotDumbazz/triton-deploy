# План работ (D1 — план как очередь и аудит-след): triton-deploy
# Этап = коммит. T-final запланирован с Genesis (I50/I56): ветка `submit`, хост — GitHub (резерв по приказу P-1).
# Канон: docs/ztacp-pattern.md v1.0-rc16 (I24). School Mode поверх Solo Mode. Уровень L1.

## Этапы

- [x] **Этап 0 — Genesis.** School Mode включён, governance (.ztacp/.plan/.spec), эталон дерева (I51), план с T-final. Коммит: `8d884a2` (этапы 0–1 одним коммитом — собраны до первого коммита пустого репо).
- [x] **Этап 1 — Контракты (B1).** model-io + ports: `8d884a2`.
- [x] **Этап 2 — W-1 models.** `193cac1`: resnet50 v1 fp32 + v2 int8, yolov5s v1 (fp16→fp32 хирургия релиз-ассета), dynamic batching. Громко: int8 v2 на CPU медленнее fp32 ~5× (dynamic quant и CNN — известно); честные числа в бенчмарке, аудит допускает («no fixed improvement bar»).
- [x] **Этап 3 — W-2 deployment.** `600c9a5`: Dockerfile, compose, run_triton.sh, grafana provisioning. Громко: до готовности W-3 up не поднимать (маунты).
- [x] **Этап 4 — W-3 monitoring.** `b718df5`: prometheus, alerts (бонус), dashboard 6 панелей. Громко: имена nv_inference_* сверить с живым /metrics на интеграции.
- [x] **Этап 5 — W-4 client.** `ad88dcf`: клиент+bенчмарк+labels+10 samples; мок-тесты E2E изолированы в /tmp и удалены.
- [x] **Этап 6 — S-1 общие файлы.** `1c6b1ea`: README, requirements, .env.example. Паритет дерева: OK.
- [x] **Этап 7 — Интеграция.** Стек up (triton healthy, prometheus, grafana); смоук обеих моделей; живое демо load/unload + версионных URL; бенчмарк before/after реальные числа (int8 ~1.7–1.9×); CSV-лог; имена метрик сверены с /metrics (2 исправления W-3); fix клиента (--images, served version). Коммиты: `0c2adb6, 7c8f618, 24df2f3, dc6736c, b7fbb86`.
- [x] **Этап 8 — Паритет и аудит.** `check-tree.sh` → OK; чек-лист аудита прогнан живьём (см. журнал); README синхронизирован с фактами (2 правки из живых находок: POST index, explicit model control). Коммит: `audit: ...`
- [x] **Этап 9 — T-final (I53).** `submit` собрана (14 коммитов, стерильность-скан чист, ролевые следы сняты из сообщений, история органичная). Main не стёрт.
- [ ] **Этап 10 — Хост (I56, override P-1).** GitHub-репо `MrRobotDumbazz/triton-deploy`, **public** (решение P-1 от 2026-11-10). Блокер: `gh auth` невалиден — refresh (`gh auth refresh -h github.com`) — акт P-1. После: gh repo create → remote github → push main + submit.

## Журнал приёма этапов

| Этап | Дата (UTC) | SHA | Паритет | Примечание |
|---|---|---|---|---|
| 0–1 | 2026-11-09 | 8d884a2 | n/a | Genesis + контракты; P-1: School+Solo, сабагенты, GitHub |
| 2 | 2026-11-09 | 193cac1 | OK (models) | W-1; yolov5s fp16→fp32 + dynamic batch — хирургия ONNX-графа |
| 3 | 2026-11-09 | 600c9a5 | OK (deployment) | W-2 |
| 4 | 2026-11-09 | b718df5 | OK (monitoring) | W-3; метрики к живой сверке |
| 5 | 2026-11-09 | ad88dcf | OK (client) | W-4 |
| 6 | 2026-11-09 | 1c6b1ea | **OK (всё дерево)** | S-1: README/requirements/.env.example |
| 7 | 2026-11-10 | 0c2adb6…b7fbb86 | OK | v2 → static QDQ (dynamic не грузился); метрики сверены; бенчмарк реальные числа; демо versioning живьём |
| 8 | 2026-11-10 | audit | **OK** | чек-лист аудита прогнан: Triton в Docker ✓, 2 модели ✓, REST+gRPC ✓, dynamic batching ✓, клиент ✓, ONNX-оптимизация+бенчмарк ✓, воспроизводимость ✓, Prometheus ✓, Grafana ✓, логи CSV ✓, версии+reload ✓, модульность ✓; бонус: алерты ✓ (firing на стенде), CSV ✓ |

## Аудит-чеклист (живой прогон 2026-11-10)

- Setup: README полный ✓; requirements.txt ✓; все файлы на месте ✓ (check-tree OK)
- Model Serving: Triton в Docker (healthy) ✓; resnet50 (2 версии) + yolov5 READY ✓; REST 8000 + gRPC 8001 ✓; dynamic batching в config.pbtxt ✓; клиент шлёт/получает ✓
- Performance Optimization: v2 int8 (static QDQ) ✓; бенчмарк до/после с p50/p95/req/s ✓; интерпретация в README ✓ (включая честный провал dynamic-quant)
- DevOps: Dockerfile + docker-compose + run_triton.sh ✓
- Monitoring: Prometheus scrape triton:8002 up ✓; дашборд Grafana провижен ✓; алерты bonus: HighInferenceLatency **firing** (стенд реально >100 мс), FailedRequests inactive ✓
- Versioning: latest{2} + versions/<v>/infer + hot load/unload (explicit) — продемонстрировано ✓
- Bonus: alerting rules ✓; inference logs → CSV ✓

| 9 | 2026-11-10 | submit f479cbc→ | OK | стерильная ветка: 14 коммитов, скан чист, сообщения органичные |
