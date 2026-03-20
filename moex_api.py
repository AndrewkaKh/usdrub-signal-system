from pprint import pprint

from processing.iv import calculate_iv_snapshot, save_iv_snapshot


if __name__ == '__main__':
    snapshot = calculate_iv_snapshot(asset_code='Si')
    save_path = save_iv_snapshot(snapshot)
    pprint(snapshot)
    print(f'\nSnapshot saved to: {save_path}')