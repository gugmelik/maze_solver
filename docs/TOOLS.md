Этот файл описывает инструменты, доступные в `tools/`. `Purpose` -- краткое описание, зачем нужен инструмент. `Roles` -- какие роли должны его использовать. `Arguments` -- какие аргументы есть у инструмента. `Outputs` -- что будет результатом работы скрипта. `Examples` -- примеры запуска соответствующего инструмента.

Ниже пересичлены описание для инструментов:



## `tools/compare_metrics.py`

### Purpose

Сравнивает одну validation-метрику для нескольких ранов и чекпойнтов.
Печатает численные значения и строит один PNG-график.

### Roles

- Validator — основная роль.
- Debugger — поиск подозрительной динамики метрик.
- Researcher — быстрое сравнение выбранных результатов.

### Arguments

- `--runs` — названия ранов или пути к ним;
- `--metric` — одна метрика в формате `dataset/metric`;
- `--checkpoints` — номера эпох;
- `--output` — путь для временного PNG.

### Outputs

- численные значения в stdout;
- один PNG-график выбранной метрики.

При написании выводов следует опираться на напечатанные значения,
а график использовать для визуальной проверки динамики.

### Examples

Сравнить один ран:

```bash
python tools/compare_metrics.py \
  --runs test_flux_fsdp_new \
  --metric db_liv_small/dino_score \
  --checkpoints 1 5 10 \
  --output /tmp/dino.png
```

Сравнить несколько ранов:

```bash
  python tools/compare_metrics.py \
  --runs baseline experiment_a experiment_b \
  --metric db_liv_small/dino_score \
  --checkpoints 1 5 10 \
  --output /tmp/dino_comparison.png
```

Показать все validation-итерации:

```bash
  python tools/compare_metrics.py \
  --runs experiment_a experiment_b \
  --metric db_liv_small/text_sim
```