# AI-42: новая архитектура обучаемого ИИ режима «Штурм»

## 1. Цель и ограничения

AI-42 должна стать обучаемой гибридной системой для командного режима 5v5: отдельная низкочастотная командная policy назначает роли и цели, а общая для всех аватаров micro-policy исполняет эти назначения с учётом тумана войны, памяти, состава героев и доступных действий.

Целевая машина: RTX 5090, Ryzen 9 9950X3D, 64 GB DDR5. Модель проектируется для одной GPU и параллельных CPU headless-сред. Покупка предметов, прокачка и server-authoritative safety остаются скриптовыми до отдельного решения.

В рамках этого пакета обучение не запускается. Разрешены только unit/integration tests, короткие protocol/inference smoke-проверки без optimizer updates и статическая проверка моделей.

## 2. Архитектурные принципы

- Actor получает только доступное конкретному герою состояние с учётом тумана войны.
- Centralized critic используется только при обучении и может получать полное авторитетное состояние матча; в production inference он отсутствует.
- Observation, action, teacher trajectory, checkpoint и inference manifest имеют независимые версии и hashes.
- Одна shared micro-policy обслуживает все аватары; hero/ability embeddings задают специализацию.
- Командные решения отделены от микроконтроля по частоте, интерфейсу и тестам.
- Невозможные действия исключаются server-authoritative masks до sampling.
- Обновление архитектуры не меняет AI-0/10/20/30 и не запускается как live-профиль без явной настройки.

## 3. Компоненты AI-42

### 3.1 Observation encoder

Вход представляется типизированными токенами:

- controlled hero;
- четыре способности;
- видимые/раскрытые союзные и вражеские герои;
- крипы, осадные крипы и волны;
- пушки, казармы, источники и алтари;
- navigation anchors и macro assignment;
- последние действия, last-seen и краткая event memory.

Каждый токен содержит type/team/visibility/validity embeddings, относительную позицию, состояние и стабильный slot identity. Наборы сущностей кодируются permutation-aware entity attention вместо `amax` pooling.

### 3.2 Micro actor

- 2–4 entity-attention блока, model width 256–384;
- temporal core: gated recurrent memory либо GTrXL с ограниченной памятью;
- shared weights и отдельное recurrent state каждого героя;
- recurrent control head `ISSUE/WAIT/HOLD/CANCEL`, затем для `ISSUE`
  autoregressive heads `action kind -> target -> point/anchor -> timing`;
- отдельные контексты для навыков, телепортации и расходников;
- macro assignment входит в actor observation как условие, а не как безусловная команда: retreat, stun, death и safety имеют приоритет.

Micro-policy работает с текущей частотой 5 Гц. Целевой начальный размер модели — 8–25 млн параметров.

### 3.3 Team macro policy

Macro-policy работает с частотой 0.5–1 Гц и видит доступное команде агрегированное состояние:

- пять allied hero tokens;
- состояние линий, волн и построек;
- видимые угрозы и last-seen summaries;
- readiness: HP, mana, death/respawn, teleport и ultimate cooldown;
- предыдущий план и время его удержания.

MAT-style autoregressive assignment формирует один согласованный план команды:

- режим `farm/defend/push/gank/group/recover`;
- objective pointer;
- роль и lane/objective assignment каждого героя;
- readiness/commitment flag и bounded plan horizon.

Macro-policy не выдаёт координат движения и не управляет способностями. До отдельного этапа она может работать в shadow mode рядом с AI-20/AI-30 orchestrator.

### 3.4 Centralized multi-head critic

Critic получает полное состояние только в learner:

- все десять героев и скрытые server-authoritative сущности;
- состояния линий, построек, волн и cooldown;
- совместные macro assignments и действия.

Отдельные value heads предсказывают:

- вероятность/return победы;
- преимущество по постройкам;
- XP/economy tempo;
- survival/teamfight outcome;
- общий scalar return для policy optimization.

Нужны value normalization, death masking и раздельная telemetry ошибок каждого head.

### 3.5 Teacher trajectories

Teacher protocol записывает каждый policy tick, включая `wait`, удержание приказа, retreat, recover и отмены. Для каждого действия сохраняются observation/action/masks и recurrent boundary, validity/rejection reason, исходное AI-30 intent, спроецированное neural action, executed action, outcome и все schema hashes.

Dataset pipeline обязан публиковать class balance и accuracy/loss отдельно для kind, target, point, anchor, timing и каждого навыка.

## 4. Обучение после завершения пакета

Обучение будет отдельной явно разрешённой операцией:

1. Полный behavior cloning на complete trajectories.
2. Проверка воспроизведения AI-30 по action-head metrics и матчам.
3. Entity-MAPPO с централизованным critic против scripted opponents.
4. Curriculum от lane/farm и боевых эпизодов к полному матчу.
5. League self-play: main policy, historical snapshots, AI-30 и bounded exploiters.
6. Опционально — world-model auxiliary heads; imagined rollouts не входят в первую production-версию.

## 5. Пакеты реализации

### AI42-P0 — source reconciliation

Вернуть существующую AI-40/41 реализацию с учебного ПК в Git source of truth, сохранив checkpoints вне репозитория. Зафиксировать baseline tests и manifest текущих схем.

### AI42-P1 — protocol foundation

Разделить версии observation/action/teacher/critic contracts, добавить frame length/offset table, строгую проверку полного body и Go/Python golden fixtures. Любой лишний или отсутствующий байт обязан давать диагностическую ошибку с полем и offset.

### AI42-P2 — entity actor

Заменить max pooling на entity attention, добавить typed tokens, hero/ability specialization, temporal core и autoregressive action heads. Сохранить action masks и детерминированный inference API.

### AI42-P3 — centralized critic

Добавить training-only full-state observation, multi-head critic, normalization contracts и tests, подтверждающие отсутствие privileged tensors в actor/export.

### AI42-P4 — macro policy

Добавить низкочастотный team state, joint assignments и shadow-mode adapter к текущему orchestrator. Проверить permutation handling, стабильность assignment и приоритет local safety.

### AI42-P5 — trajectory and metrics

Сделать полные teacher trajectories и stratified sampling, per-head metrics, dataset audit и deterministic replay. Исправление sampling не должно скрывать реальные редкие действия искусственным дублированием validation.

### AI42-P6 — migration/export/inference

Добавить manifest AI-42, миграцию только совместимых embeddings/heads, ONNX parity, recurrent-state lifecycle, latency/fallback telemetry и явный opt-in profile.

### AI42-P7 — verification gate

Запустить тесты и smoke-инференс без обучения. Production rollout и optimizer остаются запрещены до отдельной команды пользователя.

## 6. Acceptance gates

- Не менее 1 000 000 сериализованных test steps без protocol drift, trailing bytes или offset mismatch.
- Go/Python golden fixtures совпадают byte-for-byte для observation, action, critic state и teacher trajectory.
- Actor export не содержит hidden-enemy/full-state inputs; critic export в production отсутствует.
- Entity encoder различает один объект, группу объектов и перестановку равнозначных slots согласно контракту.
- Все action heads obey masks; masked action никогда не исполняется.
- Macro assignment покрывает ровно пять allied slots без дублей и не отменяет death/stun/retreat safety.
- Complete trajectory содержит ровно одну запись на каждый policy tick и явно кодирует wait/hold/cancel.
- До production rollout: PyTorch/ONNX outputs и recurrent-state transitions
  совпадают в заданной погрешности. ONNX не является входным условием для
  PyTorch-обучения на CUDA.
- Model/config/checkpoint schema mismatch приводит к детерминированному fallback, а не к частичной загрузке.
- Все unit/integration/smoke gates проходят без запуска обучения.

## 7. Риски и решения

- **Недостоверная симуляция:** обучение блокируется до прохождения protocol и gameplay invariants.
- **Слишком большая модель:** начинать с width 256 и измерять inference/learner memory до расширения.
- **Transformer instability:** gated/pre-norm blocks, gradient clipping и recurrent baseline остаются обязательным ablation.
- **Macro/micro конфликт:** versioned assignment contract и server safety priority.
- **Reward hacking:** component returns, win-rate evaluation и scripted/historical opponents анализируются раздельно.
- **Catastrophic migration:** старые AI-41 checkpoints никогда не загружаются частично без migration report.

## 8. Запрещённые действия текущего пакета

- запуск PPO, BC, self-play или campaign training;
- изменение live AI-профиля по умолчанию;
- удаление checkpoints или существующих AI версий;
- изменение reward weights без отдельной измерительной задачи;
- выдача actor скрытой информации из тумана войны.

## 9. Статус реализации на 2026-08-25

Реализованы P0-P7 и предобучающий actor-BC контур: immutable AI-30 dataset,
match-disjoint split, recurrent batching, explicit `ISSUE/WAIT/HOLD/CANCEL`
loss, exact atomic checkpoints, native control-first runtime и отдельные
no-training preflight/smoke команды. Skill-навигация v13 закреплена как
`offset81_only`; фиктивные skill anchors отклоняются.

Cross-language hash v13 закреплён golden-значением
`915e2e4547ccf727567304839f4780c60d48521f3dd1f0dbef7c4a5cc9131274`.
Два одинаковых headless-прогона seed 4242 дали побайтово одинаковые manifest и
NPZ shard. На RTX 5090 полный AI-30 dataset → recurrent batch → CUDA
forward/backward/clip/checkpoint preflight прошёл без изменения параметров;
полный AI-42 Python-набор дал 82 успешных теста и один необязательный ONNX skip.

Teacher row имеет строгую временную семантику: observation снимается сразу до
решения конкретного AI-30 героя, projection — сразу после его решения, а
`ACTION/HOLD/CANCEL/WAIT` финализируется ровно один раз после определения
terminal state. Публикация dataset всегда выполняет strict replay; lineage
`HOLD/CANCEL` принадлежит `original_ai30_intent`, а не external executed action.

Обучение и optimizer updates не запускались. `--execute` намеренно остаётся
закрыт до отдельной команды пользователя. Для первого реального BC нужны
репрезентативный многоматчевый dataset и зафиксированный одинаковый Git SHA на
локальной и учебной машинах. Real ONNX parity остаётся gate только для
production inference/rollout, а не для CUDA-обучения.

### 9.1 Behavior-cloning dataset milestone

Production collector hot path реализован на Go рядом с simulator. Формат v2
имеет magic `AI42GS2\0` и schema `AI42-go-shard-v2`; cap одного матча — 15
минут, или 4500 ticks. Финальный dataset опубликован на удалённом пути
`E:\code\Tanat Online\tanat\server\ai42_datasets\clone-v13-dataset01-v2`.
Его manifest SHA256 —
`98709a6c0606c4a4b64b59370236cab28f2d22ad0fbf7570567e61c32178ccae`, а
runtime/schedule hash —
`7a12f25a078df47ad3f8c43732aded771e2e08c9bcaf03cdbda3c242e0f38afc`.

Dataset содержит 320 matches и 685387 ticks, split — train 280 / validation
40, общий размер — 6,808,347,634 bytes; зафиксированы 3 draws/timeouts.
Native 8-match pilot записал 17,513 ticks за 8.58 s collection time и 10.705
s wall time. V2 был скомпактирован без decompression и rerun: устранено около
1.012 GB metadata overhead при сохранении compressed payload и hashes.

Удалённые тесты: 13 passed. CUDA BC preflight прошёл с 12,118,772 params,
finite loss `13.3901386261`, grad norm `11.6460485458`, checkpoint roundtrip
`true` и `parameters_unchanged true`; `training_implemented false`. Optimizer
step и training не выполнялись. Live deployment не выполнялся и не
разрешён.
