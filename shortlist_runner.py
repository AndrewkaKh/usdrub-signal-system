from __future__ import annotations

import argparse
from datetime import date, datetime

from processing.backfill import (
    build_candidate_tables,
    get_connection,
    initialize_database,
    save_option_contract_candidates,
    save_option_series_candidates,
)


def parse_date(value: str) -> date:
    return datetime.strptime(value, '%Y-%m-%d').date()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', type=parse_date, required=True)
    parser.add_argument('--end-date', type=parse_date, required=True)
    parser.add_argument('--series-pool-size', type=int, default=2)
    parser.add_argument('--max-strikes-per-series', type=int, default=10)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    connection = get_connection()
    initialize_database(connection)

    series_candidates, contract_candidates = build_candidate_tables(
        connection=connection,
        start_date=args.start_date,
        end_date=args.end_date,
        series_pool_size=args.series_pool_size,
        max_strikes_per_series=args.max_strikes_per_series,
    )

    series_saved = save_option_series_candidates(connection, series_candidates)
    contracts_saved = save_option_contract_candidates(connection, contract_candidates)

    print(
        f'Candidate selection completed | '
        f'start_date={args.start_date.isoformat()} | '
        f'end_date={args.end_date.isoformat()} | '
        f'series_candidates={len(series_candidates)} | '
        f'contract_candidates={len(contract_candidates)} | '
        f'series_saved={series_saved} | '
        f'contracts_saved={contracts_saved}'
    )

    connection.close()


if __name__ == '__main__':
    main()