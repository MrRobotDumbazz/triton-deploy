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
- [ ] **Этап 7 — Интеграция.** Docker stack up, смоук-инференс обеих моделей, benchmark before/after (реальные числа), CSV-лог. Громкий фолбэк (I54), если среда не тянет. Коммит `bench: ...`
- [ ] **Этап 8 — Паритет и аудит.** `bash .ztacp/check-tree.sh`; прогон чек-пунктов аудита из `.spec/task.md`; README синхронизирован. Коммит `audit: ...`
- [ ] **Этап 9 — T-final (I53).** `bash .agents/bin/sterile-branch.sh submit main` → стерильность-скан. Main не стирается.
- [ ] **Этап 10 — Хост (I56, override P-1).** GitHub-репо создано по прямому приказу P-1 (D2, фикс в `.ztacp/overrides.log`), пуш main + submit.

## Журнал приёма этапов

| Этап | Дата (UTC) | SHA | Паритет | Примечание |
|---|---|---|---|---|
| 0–1 | 2026-11-09 | 8d884a2 | n/a | Genesis + контракты; P-1: School+Solo, сабагенты, GitHub |
| 2 | 2026-11-09 | 193cac1 | OK (models) | W-1; yolov5s fp16→fp32 + dynamic batch — хирургия ONNX-графа |
| 3 | 2026-11-09 | 600c9a5 | OK (deployment) | W-2 |
| 4 | 2026-11-09 | b718df5 | OK (monitoring) | W-3; метрики к живой сверке |
| 5 | 2026-11-09 | ad88dcf | OK (client) | W-4 |
| 6 | 2026-11-09 | 1c6b1ea | **OK (всё дерево)** | S-1: README/requirements/.env.example |
