# План работ (D1 — план как очередь и аудит-след): triton-deploy
# Этап = коммит. T-final запланирован с Genesis (I50/I56): ветка `submit`, хост — GitHub (резерв по приказу P-1).
# Канон: docs/ztacp-pattern.md v1.0-rc16 (I24). School Mode поверх Solo Mode. Уровень L1.

## Этапы

- [x] **Этап 0 — Genesis.** School Mode включён, governance (.ztacp/.plan/.spec), эталон дерева (I51), план с T-final. Коммит: `genesis: ...`
- [ ] **Этап 1 — Контракты (B1).** `.spec/contracts/model-io.md` + `ports.md`: имена входов/выходов, препроцессинг, порты, схема CSV. Коммит.
- [ ] **Этап 2 — W-1 models.** resnet50 v1 (fp32) + v2 (оптимизированная), yolov5s v1, config.pbtxt с dynamic batching, верификация onnxruntime. Коммит `w1(models): ...`
- [ ] **Этап 3 — W-2 deployment.** deployment/docker/Dockerfile, deployment/docker-compose.yml, deployment/run_triton.sh, grafana provisioning. Коммит `w2(deployment): ...`
- [ ] **Этап 4 — W-3 monitoring.** monitoring/prometheus.yml, alerts.yml (бонус), grafana-dashboard.json. Коммит `w3(monitoring): ...`
- [ ] **Этап 5 — W-4 client.** client/inference_client.py (REST, CSV-лог — бонус), benchmark.py, 10 samples, imagenet_classes.txt. Коммит `w4(client): ...`
- [ ] **Этап 6 — S-1 общие файлы.** README.md (setup, inference, monitoring, структура, versioning), requirements.txt, .env.example. Коммит `docs: ...`
- [ ] **Этап 7 — Интеграция.** Docker stack up, смоук-инференс обеих моделей, benchmark before/after (реальные числа), CSV-лог. Громкий фолбэк (I54), если среда не тянет. Коммит `bench: ...`
- [ ] **Этап 8 — Паритет и аудит.** `bash .ztacp/check-tree.sh`; прогон чек-пунктов аудита из `.spec/task.md`; README синхронизирован. Коммит `audit: ...`
- [ ] **Этап 9 — T-final (I53).** `bash .agents/bin/sterile-branch.sh submit main` → стерильность-скан. Main не стирается.
- [ ] **Этап 10 — Хост (I56, override P-1).** GitHub-репо создано по прямому приказу P-1 (D2, фикс в `.ztacp/overrides.log`), пуш main + submit.

## Журнал приёма этапов

| Этап | Дата (UTC) | SHA | Паритет | Примечание |
|---|---|---|---|---|
| 0 | 2026-11-09 | — | n/a | Genesis, P-1 подтвердил режим: School+Solo, сабагенты, GitHub |
