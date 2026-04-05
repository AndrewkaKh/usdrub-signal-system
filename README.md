# USD/RUB Signal System

Курсовой проект, посвящённый построению системы рыночных сигналов для спот-курса USD/RUB на основе данных срочного рынка MOEX, метрик волатильности и внешних макро-финансовых факторов.

## О проекте

Цель проекта — разработать систему, которая будет автоматически собирать рыночные данные, рассчитывать ключевые метрики и формировать сигналы, связанные с изменением режима рынка и ожидаемой волатильности USD/RUB.

## Основные задачи проекта

- автоматизировать расчёт implied volatility по опционам MOEX,
- построить временные ряды IV для разных сроков,
- рассчитать historical volatility по USD/RUB,
- исследовать спред IV-HV и term structure implied volatility,
- добавить внешние факторы, влияющие на рынок,
- сформировать набор интерпретируемых сигналов,
- собрать всё в виде консольного приложения для мониторинга состояния рынка.

## Что анализируется

В проекте рассматриваются следующие группы данных:

### 1. Опционные метрики
- IV 1M / 3M / 6M
- HV 1M / 3M / 6M
- спред IV-HV
- наклон term structure implied volatility

### 2. Внешние факторы
- нефть Brent
- индекс доллара DXY
- индекс рыночного стресса VIX
- ставка и события Банка России
- инфляция
- дополнительные событийные и текстовые признаки

## Структура проекта

```
├── pipeline_runner.py       # Полный ETL-пайплайн (backfill → shortlist → dataset → export)
├── bot_runner.py            # Telegram-бот: утренний и вечерний отчёты
├── model_runner.py          # Обучение CatBoost и предсказание IV
├── dataset_qc.py            # Проверка качества датасета
├── cli_utils.py             # Общие утилиты CLI-слоя (parse_date)
│
├── scripts/                 # Вспомогательные скрипты
│   ├── backfill_runner.py   # Только этап загрузки данных
│   ├── shortlist_runner.py  # Только этап отбора контрактов
│   ├── reference_runner.py  # Только этап обогащения метаданными
│   ├── dataset_runner.py    # Только этап сборки датасета
│   ├── moex_api.py          # Real-time snapshot IV/HV
│   ├── publish_morning.py   # Быстрый запуск утреннего отчёта
│   └── publish_evening.py   # Быстрый запуск вечернего отчёта
│
├── processing/              # Библиотека обработки данных
│   ├── utils.py             # Общие утилиты (normalize_date, normalize_frame)
│   ├── moex_client.py       # HTTP-клиент MOEX ISS API
│   ├── daily_pipeline.py    # Оркестрация пайплайна
│   ├── backfill/            # Историческая загрузка данных
│   ├── iv/                  # Расчёт implied volatility
│   ├── hv/                  # Расчёт historical volatility
│   ├── dataset/             # Построение датасетов
│   │   └── selection.py     # Общая логика выбора серий и фьючерсов
│   └── features/            # Сохранение real-time снимков
│
├── model/                   # CatBoost: признаки, обучение, предсказание
├── bot/                     # Telegram: клиент, отчёты, состояние
└── data/                    # SQLite-база, CSV-экспорты, артефакты модели
```

## Запуск

### Полный пайплайн

```bash
python pipeline_runner.py --start-date 2026-01-01 --end-date 2026-04-12
```

### Telegram-бот

```bash
python bot_runner.py --morning
python bot_runner.py --evening
python bot_runner.py --schedule   # запуск по расписанию (10:00 / 20:30)
```

### Обучение и предсказание

```bash
python model_runner.py --retrain
python model_runner.py --predict
```

### Отдельные этапы пайплайна (из корня проекта)

```bash
python scripts/backfill_runner.py --start-date 2026-01-01 --end-date 2026-04-12
python scripts/shortlist_runner.py --start-date 2026-01-01 --end-date 2026-04-12
python scripts/reference_runner.py --start-date 2026-01-01 --end-date 2026-04-12
python scripts/dataset_runner.py --start-date 2026-01-01 --end-date 2026-04-12
```
