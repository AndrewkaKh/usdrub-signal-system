from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def flatten_iv_showcase(snapshot: dict[str, Any]) -> dict[str, Any]:
    metrics = snapshot.get('metrics', {}) or {}

    row = {
        'timestamp': snapshot.get('timestamp'),
        'asset_code': snapshot.get('asset_code'),
        'overall_status': snapshot.get('status'),
    }

    for tenor_label in ['1m', '3m', '6m']:
        metric = metrics.get(tenor_label, {}) or {}
        row[f'iv_{tenor_label}'] = metric.get('iv')
        row[f'status_{tenor_label}'] = metric.get('status')

    return row


def save_iv_showcase(snapshot: dict[str, Any], file_path: str = 'data/iv_timeseries.csv') -> Path:
    row = flatten_iv_showcase(snapshot)
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists() and path.stat().st_size > 0

    with path.open('a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), delimiter=';')
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return path