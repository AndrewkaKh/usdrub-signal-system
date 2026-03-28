from processing.features import save_feature_row
from processing.hv import calculate_hv_snapshot, save_hv_snapshot
from processing.iv import calculate_iv_snapshot, save_iv_snapshot


def fmt(value: float | None, digits: int = 6) -> str:
    if value is None:
        return 'None'
    return f'{value:.{digits}f}'


def get_metric_value(snapshot: dict, tenor: str, field: str):
    return snapshot.get('metrics', {}).get(tenor, {}).get(field)


if __name__ == '__main__':
    iv_snapshot = calculate_iv_snapshot(asset_code='Si')
    hv_snapshot = calculate_hv_snapshot(iv_snapshot)

    iv_snapshots_path = save_iv_snapshot(iv_snapshot)
    hv_snapshots_path = save_hv_snapshot(hv_snapshot)
    features_path = save_feature_row(iv_snapshot, hv_snapshot)

    timestamp = iv_snapshot.get('timestamp')
    asset_code = iv_snapshot.get('asset_code')
    iv_status = iv_snapshot.get('status')
    hv_status = hv_snapshot.get('status')

    iv_1m = get_metric_value(iv_snapshot, '1m', 'iv')
    iv_3m = get_metric_value(iv_snapshot, '3m', 'iv')
    iv_6m = get_metric_value(iv_snapshot, '6m', 'iv')

    hv_1m = get_metric_value(hv_snapshot, '1m', 'hv')
    hv_3m = get_metric_value(hv_snapshot, '3m', 'hv')
    hv_6m = get_metric_value(hv_snapshot, '6m', 'hv')

    print(
        f'[{timestamp}] {asset_code} | '
        f'IV status={iv_status}, HV status={hv_status} | '
        f'IV: 1m={fmt(iv_1m)}, 3m={fmt(iv_3m)}, 6m={fmt(iv_6m)} | '
        f'HV: 1m={fmt(hv_1m)}, 3m={fmt(hv_3m)}, 6m={fmt(hv_6m)}'
    )

    print(f'iv_snapshots: {iv_snapshots_path}')
    print(f'hv_snapshots: {hv_snapshots_path}')
    print(f'features: {features_path}')