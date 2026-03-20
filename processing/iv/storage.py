from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def _stringify_attempted_secids(value: Any) -> str:
    if not value:
        return ''
    if isinstance(value, list):
        return '|'.join(str(item) for item in value)
    return str(value)


def _extract_leg_fields(metric: dict[str, Any], tenor_label: str, side: str) -> dict[str, Any]:
    leg = metric.get(side, {}) or {}
    return {
        f'{side}_secid_{tenor_label}': leg.get('secid'),
        f'{side}_strike_{tenor_label}': leg.get('strike'),
        f'{side}_price_{tenor_label}': leg.get('price'),
        f'{side}_price_source_{tenor_label}': leg.get('price_source'),
        f'{side}_iv_{tenor_label}': leg.get('iv'),
        f'{side}_status_{tenor_label}': leg.get('status'),
        f'{side}_message_{tenor_label}': leg.get('message'),
        f'{side}_attempted_secids_{tenor_label}': _stringify_attempted_secids(leg.get('attempted_secids')),
    }


def flatten_iv_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        'timestamp': snapshot.get('timestamp'),
        'asset_code': snapshot.get('asset_code'),
        'requested_underlying': snapshot.get('requested_underlying'),
        'overall_status': snapshot.get('status'),
        'overall_message': snapshot.get('message'),
    }

    metrics = snapshot.get('metrics', {}) or {}

    for tenor_label in ['1m', '3m', '6m']:
        metric = metrics.get(tenor_label, {}) or {}

        row.update({
            f'underlying_{tenor_label}': metric.get('underlying'),
            f'futures_price_{tenor_label}': metric.get('futures_price'),
            f'futures_price_source_{tenor_label}': metric.get('futures_price_source'),
            f'target_days_{tenor_label}': metric.get('target_days'),
            f'tolerance_days_{tenor_label}': metric.get('tolerance_days'),
            f'expiry_{tenor_label}': metric.get('expiry'),
            f'days_to_expiry_{tenor_label}': metric.get('days_to_expiry'),
            f't_{tenor_label}': metric.get('t'),
            f'iv_{tenor_label}': metric.get('iv'),
            f'status_{tenor_label}': metric.get('status'),
            f'message_{tenor_label}': metric.get('message'),
        })

        row.update(_extract_leg_fields(metric, tenor_label, 'call'))
        row.update(_extract_leg_fields(metric, tenor_label, 'put'))

    return row


def save_iv_snapshot(snapshot: dict[str, Any], file_path: str = 'data/iv_snapshots.csv') -> Path:
    row = flatten_iv_snapshot(snapshot)
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists() and path.stat().st_size > 0

    with path.open('a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), delimiter=';')
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return path