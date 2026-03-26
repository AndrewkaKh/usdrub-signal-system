from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


TENOR_ORDER = ['1m', '3m', '6m']


def _safe_spread(iv_value: Any, hv_value: Any) -> float | None:
    if iv_value is None or hv_value is None:
        return None
    return float(iv_value) - float(hv_value)


def _safe_diff(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def flatten_feature_row(iv_snapshot: dict[str, Any], hv_snapshot: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        'timestamp': iv_snapshot.get('timestamp') or hv_snapshot.get('timestamp'),
        'asset_code': iv_snapshot.get('asset_code') or hv_snapshot.get('asset_code'),
        'iv_overall_status': iv_snapshot.get('status'),
        'hv_overall_status': hv_snapshot.get('status'),
    }

    iv_metrics = iv_snapshot.get('metrics', {}) or {}
    hv_metrics = hv_snapshot.get('metrics', {}) or {}

    iv_values: dict[str, float | None] = {}
    hv_values: dict[str, float | None] = {}

    for tenor_label in TENOR_ORDER:
        iv_metric = iv_metrics.get(tenor_label, {}) or {}
        hv_metric = hv_metrics.get(tenor_label, {}) or {}

        iv_value = iv_metric.get('iv')
        hv_value = hv_metric.get('hv')
        iv_values[tenor_label] = iv_value
        hv_values[tenor_label] = hv_value

        row.update({
            f'underlying_{tenor_label}': iv_metric.get('underlying') or hv_metric.get('underlying'),
            f'iv_{tenor_label}': iv_value,
            f'hv_{tenor_label}': hv_value,
            f'spread_{tenor_label}': _safe_spread(iv_value, hv_value),
            f'iv_status_{tenor_label}': iv_metric.get('status'),
            f'hv_status_{tenor_label}': hv_metric.get('status'),
        })

    row['ts_3m_1m'] = _safe_diff(iv_values.get('3m'), iv_values.get('1m'))
    row['ts_6m_3m'] = _safe_diff(iv_values.get('6m'), iv_values.get('3m'))
    row['ts_6m_1m'] = _safe_diff(iv_values.get('6m'), iv_values.get('1m'))

    return row


def save_feature_row(
    iv_snapshot: dict[str, Any],
    hv_snapshot: dict[str, Any],
    file_path: str = 'data/features_timeseries.csv',
) -> Path:
    row = flatten_feature_row(iv_snapshot, hv_snapshot)
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists() and path.stat().st_size > 0
    with path.open('a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), delimiter=';')
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return path
