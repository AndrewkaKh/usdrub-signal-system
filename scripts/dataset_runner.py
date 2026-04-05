from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cli_utils import parse_date
from processing.backfill import get_connection, initialize_database
from processing.dataset import (
    build_hv_daily,
    build_iv_daily,
    build_model_dataset_daily,
    save_hv_daily,
    save_iv_daily,
    save_model_dataset_daily,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', type=parse_date, required=True)
    parser.add_argument('--end-date', type=parse_date, required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    connection = get_connection()
    initialize_database(connection)

    iv_daily = build_iv_daily(
        connection=connection,
        start_date=args.start_date.isoformat(),
        end_date=args.end_date.isoformat(),
    )
    iv_saved = save_iv_daily(connection, iv_daily)

    hv_daily = build_hv_daily(
        connection=connection,
        start_date=args.start_date.isoformat(),
        end_date=args.end_date.isoformat(),
    )
    hv_saved = save_hv_daily(connection, hv_daily)

    model_dataset = build_model_dataset_daily(
        connection=connection,
        start_date=args.start_date.isoformat(),
        end_date=args.end_date.isoformat(),
    )
    dataset_saved = save_model_dataset_daily(connection, model_dataset)

    print(
        f'Dataset build completed | '
        f'start_date={args.start_date.isoformat()} | '
        f'end_date={args.end_date.isoformat()} | '
        f'iv_daily_rows={len(iv_daily)} | '
        f'hv_daily_rows={len(hv_daily)} | '
        f'model_dataset_rows={len(model_dataset)} | '
        f'iv_saved={iv_saved} | '
        f'hv_saved={hv_saved} | '
        f'dataset_saved={dataset_saved}'
    )

    connection.close()


if __name__ == '__main__':
    main()