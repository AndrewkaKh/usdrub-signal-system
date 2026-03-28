from __future__ import annotations

from pathlib import Path

import pandas as pd


def _ensure_parent_dir(path: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def export_table_to_csv(
    connection,
    table_name: str,
    output_path: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Path:
    query = f'SELECT * FROM {table_name}'
    params: list[str] = []

    if start_date is not None and end_date is not None:
        query += ' WHERE date BETWEEN ? AND ?'
        params.extend([start_date, end_date])

    query += ' ORDER BY date'

    frame = pd.read_sql_query(query, connection, params=params)
    target = _ensure_parent_dir(output_path)
    frame.to_csv(target, sep=';', index=False, encoding='utf-8-sig')
    return target


def export_model_dataset_daily(
    connection,
    output_path: str = 'data/exports/model_dataset_daily.csv',
    start_date: str | None = None,
    end_date: str | None = None,
) -> Path:
    return export_table_to_csv(
        connection=connection,
        table_name='model_dataset_daily',
        output_path=output_path,
        start_date=start_date,
        end_date=end_date,
    )


def export_iv_daily(
    connection,
    output_path: str = 'data/exports/iv_daily.csv',
    start_date: str | None = None,
    end_date: str | None = None,
) -> Path:
    return export_table_to_csv(
        connection=connection,
        table_name='iv_daily',
        output_path=output_path,
        start_date=start_date,
        end_date=end_date,
    )


def export_hv_daily(
    connection,
    output_path: str = 'data/exports/hv_daily.csv',
    start_date: str | None = None,
    end_date: str | None = None,
) -> Path:
    return export_table_to_csv(
        connection=connection,
        table_name='hv_daily',
        output_path=output_path,
        start_date=start_date,
        end_date=end_date,
    )