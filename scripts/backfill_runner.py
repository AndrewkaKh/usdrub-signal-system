from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cli_utils import parse_date
from processing.backfill import (
    get_connection,
    initialize_database,
    load_futures_backfill,
    load_options_backfill,
    save_futures_raw,
    save_options_raw,
)
from processing.backfill.config import BACKFILL_END_DATE, BACKFILL_START_DATE, SQLITE_DB_PATH


def resolve_end_date() -> date:
    if BACKFILL_END_DATE is not None:
        return BACKFILL_END_DATE
    return date.today()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', type=parse_date, default=BACKFILL_START_DATE)
    parser.add_argument('--end-date', type=parse_date, default=resolve_end_date())
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    connection = get_connection()
    initialize_database(connection)

    futures_frame = load_futures_backfill(args.start_date, args.end_date)
    options_frame = load_options_backfill(args.start_date, args.end_date)

    futures_saved = save_futures_raw(connection, futures_frame)
    options_saved = save_options_raw(connection, options_frame)

    print(
        f'Backfill completed | '
        f'start_date={args.start_date.isoformat()} | '
        f'end_date={args.end_date.isoformat()} | '
        f'futures_rows_loaded={len(futures_frame)} | '
        f'options_rows_loaded={len(options_frame)} | '
        f'futures_rows_saved={futures_saved} | '
        f'options_rows_saved={options_saved} | '
        f'db_path={SQLITE_DB_PATH}'
    )

    connection.close()


if __name__ == '__main__':
    main()