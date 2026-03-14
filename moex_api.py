from pprint import pprint

from processing.iv import calculate_iv_snapshot


if __name__ == '__main__':
    pprint(calculate_iv_snapshot(asset_code='Si'))