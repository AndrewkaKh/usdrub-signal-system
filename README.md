# USD/RUB Signal System

Курсовой проект — система рыночных сигналов для спот-курса USD/RUB на основе данных срочного рынка MOEX, метрик волатильности и внешних макро-финансовых факторов.

## О проекте

Система автоматически собирает рыночные данные, рассчитывает ключевые метрики волатильности и формирует прогноз изменения implied volatility на следующий торговый день. Прогноз используется для оценки ожидаемого диапазона курса USD/RUB.

## Что анализируется

### Опционные метрики
- IV 1M / 3M — implied volatility по ATM-страйкам
- HV 1M / 3M — историческая волатильность по фьючерсам Si
- спред IV-HV, наклон term structure
- улыбка волатильности: RR25, BF25, RR10, skew

### Внешние факторы
- нефть Brent, индекс доллара DXY, индекс стресса VIX
- ставка Банка России и события комитета по ДКП

## Модель

**Алгоритм:** CatBoostRegressor, loss = MAE  
**Target:** `sigma_normalized_delta` — нормированное изменение IV:

```
z(t) = (iv(t+1) - iv(t)) / rolling_std(past_delta, 20)
```

На инференсе восстанавливается через обратное преобразование:

```
pred_delta = predicted_z × rolling_std_20
pred_iv    = current_iv + pred_delta
```

**Признаки:** 47 признаков — IV/HV-лаги, скользящие средние, term structure, внешние факторы (Brent lag-1/2, DXY lag-1/2, VIX lag-1/2 + roll5), календарные признаки.

**Выбор target:** сравнение трёх постановок по 8-фолдовому walk-forward CV (step=3m, eval=3m):

| Target | RMSE | MAE | Sign Acc | Beats baseline (MAE) |
|---|---|---|---|---|
| **sigma_normalized** | **0.01322** | **0.00881** | **58.5%** | **6/8** |
| log_return_iv | 0.01331 | 0.00883 | 58.3% | 5/8 |
| raw_delta | 0.01336 | 0.00896 | 55.8% | 5/8 |

Все метрики пересчитаны в единое raw-delta пространство через inverse transform.

**Почему это работает:** модель предсказывает направление изменения IV в 58.5% случаев при полностью out-of-sample оценке — это устойчивое преимущество над случайным угадыванием на рынке с kurtosis=53.7 и 46% «тихих» дней. Нормировка на rolling std делает target более стационарным и снижает доминирование выбросов в функции потерь, что выражается в лучших MAE и sign accuracy по всем фолдам.

## Структура проекта

```
├── pipeline_runner.py       # ETL-пайплайн: backfill → shortlist → dataset → export
├── bot_runner.py            # Telegram-бот: утренний и вечерний отчёты
├── model_runner.py          # CLI: обучение, предсказание, walk-forward CV
├── cli_utils.py             # Общие утилиты CLI (parse_date)
│
├── model/                   # Продуктовый ML-код
│   ├── features.py          # Список признаков, TARGET_COL, PRODUCT_TARGET_TYPE
│   ├── targets.py           # Варианты target и inverse transforms
│   ├── data_prep.py         # Загрузка датасета, feature engineering, target
│   ├── train.py             # Обучение CatBoost, walk-forward CV, метрики
│   ├── predict.py           # Инференс: z → delta → IV → диапазон USD/RUB
│   ├── range_forecast.py    # Black-Scholes диапазон по predicted IV
│   └── artifacts/           # catboost_iv_1m.cbm, metadata.json
│
├── research/                # Исследовательский код (не влияет на прод)
│   ├── target_analysis.py   # Анализ распределения delta IV, подбор threshold
│   ├── eval_targets.py      # Walk-forward сравнение: raw / log_return / sigma_norm
│   ├── eval_thresholds.py   # 3-классовый CatBoostClassifier, 7 стратегий threshold
│   └── eval_intervals.py    # Интервальный прогноз (quantile q10/q50/q90)
│
├── processing/              # Библиотека обработки данных
│   ├── moex_client.py       # HTTP-клиент MOEX ISS API
│   ├── daily_pipeline.py    # Оркестрация пайплайна
│   ├── backfill/            # Историческая загрузка в SQLite
│   ├── iv/                  # Расчёт IV (Black-Scholes, py_vollib)
│   ├── hv/                  # Расчёт HV (log-returns, rolling std)
│   └── dataset/             # Построение датасетов, IV smile, внешние данные
│
├── bot/                     # Telegram: клиент, форматирование, состояние
├── scripts/                 # Вспомогательные однофункциональные скрипты
└── data/                    # SQLite-база, CSV-экспорты
    ├── backfill/            # moex_backfill.sqlite3
    └── exports/             # model_dataset_daily.csv, eval_*.csv, iv_*.csv
```

## Запуск

### Полный ETL-пайплайн

```bash
python pipeline_runner.py --start-date 2026-01-01 --end-date 2026-04-12
```

### Telegram-бот

```bash
python bot_runner.py --morning
python bot_runner.py --evening
python bot_runner.py --schedule   # 10:00 и 20:30 по расписанию
```

### Отдельные этапы пайплайна

```bash
python scripts/backfill_runner.py  --start-date 2026-01-01 --end-date 2026-04-12
python scripts/shortlist_runner.py --start-date 2026-01-01 --end-date 2026-04-12
python scripts/reference_runner.py --start-date 2026-01-01 --end-date 2026-04-12
python scripts/dataset_runner.py   --start-date 2026-01-01 --end-date 2026-04-12
```
