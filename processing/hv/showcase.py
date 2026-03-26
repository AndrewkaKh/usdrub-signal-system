from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


TENOR_ORDER = ['1m', '3m', '6m']


def flatten_hv_showcase(snapshot: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        'timestamp': snapshot.get('timestamp'),
        'asset_code': snapshot.get('asset_code'),
        'overall_status': snapshot.get('status'),
    }

    metrics = snapshot.get('metrics', {}) or {}
    for tenor_label in TENOR_ORDER:
        metric = metrics.get(tenor_label, {}) or {}
        row[f'underlying_{tenor_label}'] = metric.get('underlying')
        row[f'hv_{tenor_label}'] = metric.get('hv')
        row[f'status_{tenor_label}'] = metric.get('status')

    return row


def save_hv_showcase(snapshot: dict[str, Any], file_path: str = 'data/hv_timeseries.csv') -> Path:
    row = flatten_hv_showcase(snapshot)
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists() and path.stat().st_size > 0
    with path.open('a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), delimiter=';')
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return path
