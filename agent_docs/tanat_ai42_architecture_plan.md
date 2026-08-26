# AI-42: новая архитектура обучаемого ИИ режима «Штурм»

## 1. Цель и ограничения

AI-42 должна стать обучаемой гибридной системой для командного режима 5v5: отдельная низкочастотная командная policy назначает роли и цели, а общая для всех аватаров micro-policy исполняет эти назначения с учётом тумана войны, памяти, состава героев и доступных действий.

Целевая машина: RTX 5090, Ryzen 9 9950X3D, 64 GB DDR5. Модель проектируется для одной GPU и параллельных CPU headless-сред. Покупка предметов, прокачка и server-authoritative safety остаются скриптовыми до отдельного решения.

Основной пакет не запускает PPO, self-play или campaign training. Ограниченный
BC-запуск разрешён только отдельной явной командой через `--execute` и
пятиминутный лимит optimizer; остальные проверки остаются unit/integration
tests, короткими protocol/inference smoke-проверками и статической проверкой
моделей.

## 2. Архитектурные принципы

- Actor получает только доступное конкретному герою состояние с учётом тумана войны.
- На этапе behavior cloning используется только actor. Centralized critic добавляется отдельным модулем при переходе к PPO и может получать полное авторитетное состояние матча; в production inference он отсутствует.
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

- текущая BC-baseline: 2 entity-attention блока, model width 192;
- temporal core: gated recurrent memory либо GTrXL с ограниченной памятью;
- shared weights и отдельное recurrent state каждого героя;
- recurrent control head `ISSUE/WAIT/HOLD/CANCEL`, затем для `ISSUE`
  heads `action kind -> target -> point/anchor`;
- отдельные контексты для навыков, телепортации и расходников;
- macro assignment входит в actor observation как условие, а не как безусловная команда: retreat, stun, death и safety имеют приоритет.

Micro-policy работает с текущей частотой 5 Гц. Текущий structured compact actor
содержит 2,169,643 параметра. Размер увеличивается только при подтверждённом
underfit на train и validation; timing head возвращается только после появления
исполняемого сервером duration/repeat-контракта.

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

Critic не входит в текущий BC-контур. На этапе PPO он получает полное состояние
только в learner:

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
- **Слишком большая модель:** начинать с width 192 и измерять underfit, inference и learner throughput до расширения.
- **Transformer instability:** gated/pre-norm blocks, gradient clipping и recurrent baseline остаются обязательным ablation.
- **Macro/micro конфликт:** versioned assignment contract и server safety priority.
- **Reward hacking:** component returns, win-rate evaluation и scripted/historical opponents анализируются раздельно.
- **Catastrophic migration:** старые AI-41 checkpoints никогда не загружаются частично без migration report.

## 8. Запрещённые действия текущего пакета

- запуск PPO, self-play или campaign training без отдельного разрешения;
- запуск BC без явного `--execute` и лимита optimizer не более 300 секунд;
- изменение live AI-профиля по умолчанию;
- удаление checkpoints или существующих AI версий;
- изменение reward weights без отдельной измерительной задачи;
- выдача actor скрытой информации из тумана войны.

## 9. Статус реализации на 2026-08-25

Реализованы protocol/actor/dataset/export/preflight части P0-P2 и P5-P7, а
также предобучающий actor-BC контур: immutable AI-30 dataset,
match-disjoint split, recurrent batching, explicit `ISSUE/WAIT/HOLD/CANCEL`
loss, exact atomic checkpoints, native control-first runtime и отдельные
no-training preflight/smoke команды. Skill-навигация v13 закреплена как
`offset81_only`; фиктивные skill anchors отклоняются.

P3 centralized critic и отдельная learned P4 macro-policy отложены до PPO и
появления соответствующих returns/teacher labels. Скриптовый orchestrator и
macro assignment в observation при этом остаются частью среды.

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

До отдельного разрешения обучение оставалось закрытым: preflight не вызывает
`optimizer.step`, а `--execute` требуется для BC. Для разрешённого запуска
зафиксированы одинаковый Git SHA на локальной и учебной машинах и
репрезентативный многоматчевый dataset. Real ONNX parity остаётся gate только
для production inference/rollout, а не для CUDA-обучения.

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

Историческая проверка actor-v1: 13 удалённых тестов passed. CUDA BC preflight
прошёл с 12,118,772 params,
finite loss `13.3901386261`, grad norm `11.6460485458`, checkpoint roundtrip
`true` и `parameters_unchanged true`; `training_implemented false`. Optimizer
step и training в этом preflight не выполнялись. Live deployment не выполнялся
и не разрешён.

### 9.2 Пяти минут behavior cloning и первый accepted run

Executable workflow запускается из `tanat/server` на checkout с двумя
зафиксированными revisions: `e0056a2` поверх `80c3600`. Его реализация —
[`train_ai42_bc.py`](../server/ai40/src/tanat_ai40/train_ai42_bc.py), параметры —
в [`ai42_bc_training.json`](../server/ai40/config/ai42_bc_training.json).
Dataset должен быть
валидированным и неизменяемым; для первого запуска использован
`E:\code\Tanat Online\tanat\server\ai42_datasets\clone-v13-dataset01-v2`.
Из `server` команда имеет вид:

```powershell
$env:PYTHONPATH = "$PWD\ai40\src"
$dataset = 'E:\code\Tanat Online\tanat\server\ai42_datasets\clone-v13-dataset01-v2'
$run = 'E:\code\Tanat Online\ai42_runs\bc5m-e0056a2-01'
python -m tanat_ai40.train_ai42_bc `
  --config .\ai40\config\ai42_bc_training.json `
  --dataset $dataset --output $run --device cuda `
  --batch-size 8 --max-optimizer-seconds 300 --execute
```

План batch-ов строится детерминированно: match IDs ранжируются SHA-256 и
стратифицируются по `scenario`; в optimizer попадают только batch-ы с
эффективной supervised-маской. Checkpoint сохраняет hash плана и точный
`batch_cursor`, поэтому `--resume <run>\latest.pt` продолжает тот же поток;
периодический и финальный candidate публикуются как `latest.pt`. Accepted
promotion создаёт новую immutable generation, валидирует полный digest pointer,
checkpoint и сериализованных artifact hashes, затем атомарно публикует
`accepted_pointer.json`.
Конфигурация задаёт `max_steps=1000`; фактический run остановился на 131-м
шаге из-за 300-секундного deadline.

Первый удалённый run: `E:\code\Tanat Online\ai42_runs\bc5m-e0056a2-01`, CUDA,
131 optimizer steps за 300 секунд; accepted generation SHA-256:
`fe3c111789c9594a76a5ab7f125566ebc2ceae5642b94243c44b44c9c9482f3c`.
Validation loss снизился с `13.2954028457` до `12.5962044984` (`-5.26%`),
train probe — с `7.021938324` до `5.387267113` (`-23.28%`). При этом control
accuracy снизилась с `.5592` до `.3164`, несмотря на улучшение control loss;
offset составил `1.346%`. Это accepted BC candidate, но не доказательство
live-интеграции или корректности gameplay.

Следующие gates: frozen global class weights; confusion/per-class metrics;
macro-F1 и balanced accuracy; end-to-end action correctness; offset top-k и
distance; затем повторные 5-minute/resume прогоны и headless evaluation. В
этом результате не выполнялись live deployment, ONNX или PPO.

### 9.3 Compact actor v2

После аудита архитектуры экспериментальные Q3-Q13 конфиги, неиспользуемые
timing heads, standalone macro network и centralized critic удалены из текущего
BC product surface. Actor уменьшен до width 192, двух attention blocks и
2,181,609 параметров. Skill1-Skill4 получают отдельные logits из собственных
ability tokens; обучение всегда end-to-end, class weights формируются только по
immutable train profile. Accepted checkpoint публикуется одним механизмом:
immutable generation плюс атомарный `accepted_pointer.json`.

Actor-v2 checkpoint-несовместим с actor-v1. Первый v2 run начинается с
детерминированной инициализации seed 4242; следующие итерации используют
warm-start после native preflight. До нового измерения старые результаты 9.1 и
9.2 являются историческим baseline, а не оценкой compact v2.

### 9.4 Аудит compact actor и structured heads

На immutable dataset из раздела 9.1 выполнены три изолированных пятиминутных
scratch-прогона compact actor v2. Во всех случаях optimizer работал примерно
438–441 шаг, однако ни один candidate не прошёл acceptance: end-to-end action
accuracy снизилась с начальных `1.1325%` до нуля.

- Базовая конфигурация снизила validation loss с `13.1541` до `9.2346`, но
  offset Manhattan error ухудшился с `7.4106` до `7.8550`; control ISSUE и
  CANCEL получили нулевой recall.
- Class-balance power `0.75` и uniform offset weights дали offset top-1
  `3.5295%`, но Manhattan error `7.7023`, нулевой end-to-end result и тот же
  нулевой recall ISSUE/CANCEL.
- Полные inverse semantic weights и дополнительный coordinate loss снизили
  validation loss с `14.6077` до `12.2998`, но дали offset top-1 `3.5180%`,
  Manhattan error `7.7028` и снова нулевой end-to-end result.

Распределение первых 441 optimizer batch совпало с полным train profile по
control и kind proportions. Следовательно, проблема не вызвана неудачным
префиксом детерминированного batch stream; дальнейшая настройка scalar weights
без изменения представления признана исчерпанной.

В revision `b189594` изменены только две выходные головы:

- control факторизован как `ISSUE / continuation`, затем
  `WAIT / HOLD / CANCEL` внутри continuation;
- navigation offset факторизован на независимые row/column logits размером
  9×9 и собирается обратно в прежний публичный массив из 81 logit.

Внешний action/ONNX-контракт не изменился: control по-прежнему имеет четыре, а
offset — 81 значение. Flat 81-way cross-entropy для аддитивной grid-head уже
эквивалентна сумме row/column cross-entropy, поэтому экспериментальный
coordinate auxiliary loss и его конфигурация удалены как дублирующее
масштабирование той же ошибки. Encoder, два entity-attention блока, recurrent
memory, ability specialization, kind, target и anchor heads не уменьшались.

Итоговый actor содержит `2,169,643` параметра против `2,181,609` у предыдущей
версии; уменьшение на 11,966 параметров полностью приходится на замену плоской
81-way navigation projection факторизованной головой с поправкой на новый
control head. Фактический parameter count совпал с независимыми estimators в
smoke и native-worker контрактах. На RTX 5090 прошли 70 actor/runtime/BC/smoke
тестов и 10 export/ONNX-тестов; Go preflight packages также прошли.

Следующий ограниченный эксперимент — один scratch-run structured actor с тем же
dataset, seed и пятиминутным optimizer budget. Он должен сравниваться с тремя
результатами выше по end-to-end action accuracy, ISSUE/CANCEL recall, offset
top-1 и Manhattan error; снижение только общего loss не является основанием для
promotion.
