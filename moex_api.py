from pprint import pprint

from processing.iv import calculate_iv_snapshot, save_iv_showcase, save_iv_snapshot


if __name__ == '__main__':
    snapshot = calculate_iv_snapshot(asset_code='Si')
    snapshots_path = save_iv_snapshot(snapshot)
    showcase_path = save_iv_showcase(snapshot)
    pprint(snapshot)
    print(f'\nSnapshot saved to: {snapshots_path}')
    print(f'IV showcase saved to: {showcase_path}')