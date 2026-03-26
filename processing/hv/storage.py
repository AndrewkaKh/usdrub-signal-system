from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


TENOR_ORDER = ['1m', '3m', '6m']


def flatten_hv_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        'timestamp': snapshot.get('timestamp'),
        'asset_code': snapshot.get('asset_code'),
        'overall_status': snapshot.get('status'),
        'overall_message': snapshot.get('message'),
    }

    metrics = snapshot.get('metrics', {}) or {}
    for tenor_label in TENOR_ORDER:
        metric = metrics.get(tenor_label, {}) or {}
        row.update({
            f'underlying_{tenor_label}': metric.get('underlying'),
            f'window_{tenor_label}': metric.get('window'),
            f'annualization_days_{tenor_label}': metric.get('annualization_days'),
            f'interval_{tenor_label}': metric.get('interval'),
            f'from_date_{tenor_label}': metric.get('from_date'),
            f'till_date_{tenor_label}': metric.get('till_date'),
            f'price_field_{tenor_label}': metric.get('price_field'),
            f'source_rows_{tenor_label}': metric.get('source_rows'),
            f'price_observations_{tenor_label}': metric.get('price_observations'),
            f'returns_observations_{tenor_label}': metric.get('returns_observations'),
            f'hv_{tenor_label}': metric.get('hv'),
            f'status_{tenor_label}': metric.get('status'),
            f'message_{tenor_label}': metric.get('message'),
        })

    return row


def save_hv_snapshot(snapshot: dict[str, Any], file_path: str = 'data/hv_snapshots.csv') -> Path:
    row = flatten_hv_snapshot(snapshot)
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists() and path.stat().st_size > 0
    with path.open('a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), delimiter=';')
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return path
