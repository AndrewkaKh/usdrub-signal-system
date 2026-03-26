from pprint import pprint

from processing.features import save_feature_row
from processing.hv import calculate_hv_snapshot, save_hv_showcase, save_hv_snapshot
from processing.iv import calculate_iv_snapshot, save_iv_showcase, save_iv_snapshot


if __name__ == '__main__':
    iv_snapshot = calculate_iv_snapshot(asset_code='Si')
    hv_snapshot = calculate_hv_snapshot(iv_snapshot)

    iv_snapshots_path = save_iv_snapshot(iv_snapshot)
    iv_showcase_path = save_iv_showcase(iv_snapshot)

    hv_snapshots_path = save_hv_snapshot(hv_snapshot)
    hv_showcase_path = save_hv_showcase(hv_snapshot)

    features_path = save_feature_row(iv_snapshot, hv_snapshot)

    print('IV snapshot:')
    pprint(iv_snapshot)
    print('\nHV snapshot:')
    pprint(hv_snapshot)

    print(f'\nIV snapshot saved to: {iv_snapshots_path}')
    print(f'IV showcase saved to: {iv_showcase_path}')
    print(f'HV snapshot saved to: {hv_snapshots_path}')
    print(f'HV showcase saved to: {hv_showcase_path}')
    print(f'Features showcase saved to: {features_path}')
