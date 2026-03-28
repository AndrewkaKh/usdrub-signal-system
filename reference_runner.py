from __future__ import annotations

import argparse
from datetime import date, datetime

from processing.backfill import (
    build_missing_contract_references,
    get_connection,
    initialize_database,
    save_option_contracts_reference,
)


def parse_date(value: str) -> date:
    return datetime.strptime(value, '%Y-%m-%d').date()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', type=parse_date, required=True)
    parser.add_argument('--end-date', type=parse_date, required=True)
    parser.add_argument('--max-workers', type=int, default=8)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    connection = get_connection()
    initialize_database(connection)

    reference_frame = build_missing_contract_references(
        connection=connection,
        start_date=args.start_date.isoformat(),
        end_date=args.end_date.isoformat(),
        max_workers=args.max_workers,
    )

    saved = save_option_contracts_reference(connection, reference_frame)

    print(
        f'Reference loading completed | '
        f'start_date={args.start_date.isoformat()} | '
        f'end_date={args.end_date.isoformat()} | '
        f'reference_rows={len(reference_frame)} | '
        f'reference_saved={saved}'
    )

    connection.close()


if __name__ == '__main__':
    main()