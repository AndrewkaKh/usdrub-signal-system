from __future__ import annotations

import argparse
import logging

from cli_utils import parse_date
from processing.backfill import get_connection, initialize_database
from processing.daily_pipeline import run_daily_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', type=parse_date, required=True)
    parser.add_argument('--end-date', type=parse_date, required=True)
    parser.add_argument('--series-pool-size', type=int, default=2)
    parser.add_argument('--max-strikes-per-series', type=int, default=10)
    parser.add_argument('--reference-workers', type=int, default=8)
    parser.add_argument('--skip-backfill', action='store_true')
    parser.add_argument('--skip-shortlist', action='store_true')
    parser.add_argument('--skip-reference', action='store_true')
    parser.add_argument('--skip-dataset', action='store_true')
    parser.add_argument('--skip-export', action='store_true')
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    connection = get_connection()
    initialize_database(connection)

    result = run_daily_pipeline(
        connection=connection,
        start_date=args.start_date,
        end_date=args.end_date,
        series_pool_size=args.series_pool_size,
        max_strikes_per_series=args.max_strikes_per_series,
        reference_workers=args.reference_workers,
        skip_backfill=args.skip_backfill,
        skip_shortlist=args.skip_shortlist,
        skip_reference=args.skip_reference,
        skip_dataset=args.skip_dataset,
        skip_export=args.skip_export,
    )

    connection.close()

    print(
        f'Pipeline completed | '
        f'start_date={result["start_date"]} | '
        f'end_date={result["end_date"]}'
    )
    print(
        f'backfill | futures_loaded={result["futures_loaded"]} | '
        f'options_loaded={result["options_loaded"]} | '
        f'futures_saved={result["futures_saved"]} | '
        f'options_saved={result["options_saved"]}'
    )
    print(
        f'shortlist | series_candidates={result["series_candidates"]} | '
        f'contract_candidates={result["contract_candidates"]} | '
        f'series_saved={result["series_saved"]} | '
        f'contracts_saved={result["contracts_saved"]}'
    )
    print(
        f'reference | reference_rows={result["reference_rows"]} | '
        f'reference_saved={result["reference_saved"]}'
    )
    print(
        f'dataset | iv_rows={result["iv_rows"]} | hv_rows={result["hv_rows"]} | '
        f'model_rows={result["dataset_rows"]} | '
        f'iv_saved={result["iv_saved"]} | hv_saved={result["hv_saved"]} | '
        f'dataset_saved={result["dataset_saved"]}'
    )
    if result['model_csv']:
        print(f'export | model_dataset={result["model_csv"]}')
        print(f'export | iv_daily={result["iv_csv"]}')
        print(f'export | hv_daily={result["hv_csv"]}')


if __name__ == '__main__':
    main()
