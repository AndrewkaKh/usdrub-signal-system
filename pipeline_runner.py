from __future__ import annotations

import argparse
from datetime import date, datetime

from processing.backfill import (
    build_candidate_tables,
    build_missing_contract_references,
    get_connection,
    initialize_database,
    load_futures_backfill,
    load_options_backfill,
    save_futures_raw,
    save_option_contract_candidates,
    save_option_contracts_reference,
    save_option_series_candidates,
    save_options_raw,
)
from processing.dataset import (
    build_hv_daily,
    build_iv_daily,
    build_model_dataset_daily,
    save_hv_daily,
    save_iv_daily,
    save_model_dataset_daily,
)
from processing.dataset.exporter import export_hv_daily, export_iv_daily, export_model_dataset_daily


def parse_date(value: str) -> date:
    return datetime.strptime(value, '%Y-%m-%d').date()


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

    start_date = args.start_date.isoformat()
    end_date = args.end_date.isoformat()

    connection = get_connection()
    initialize_database(connection)

    futures_loaded = 0
    options_loaded = 0
    futures_saved = 0
    options_saved = 0

    if not args.skip_backfill:
        futures_frame = load_futures_backfill(args.start_date, args.end_date)
        options_frame = load_options_backfill(args.start_date, args.end_date)

        futures_loaded = len(futures_frame)
        options_loaded = len(options_frame)

        futures_saved = save_futures_raw(connection, futures_frame)
        options_saved = save_options_raw(connection, options_frame)

    series_candidates_len = 0
    contract_candidates_len = 0
    series_saved = 0
    contracts_saved = 0

    if not args.skip_shortlist:
        series_candidates, contract_candidates = build_candidate_tables(
            connection=connection,
            start_date=args.start_date,
            end_date=args.end_date,
            series_pool_size=args.series_pool_size,
            max_strikes_per_series=args.max_strikes_per_series,
        )

        series_candidates_len = len(series_candidates)
        contract_candidates_len = len(contract_candidates)

        series_saved = save_option_series_candidates(connection, series_candidates)
        contracts_saved = save_option_contract_candidates(connection, contract_candidates)

    reference_len = 0
    reference_saved = 0

    if not args.skip_reference:
        reference_frame = build_missing_contract_references(
            connection=connection,
            start_date=start_date,
            end_date=end_date,
            max_workers=args.reference_workers,
        )

        reference_len = len(reference_frame)
        reference_saved = save_option_contracts_reference(connection, reference_frame)

    iv_len = 0
    hv_len = 0
    dataset_len = 0
    iv_saved = 0
    hv_saved = 0
    dataset_saved = 0

    if not args.skip_dataset:
        iv_daily = build_iv_daily(
            connection=connection,
            start_date=start_date,
            end_date=end_date,
        )
        hv_daily = build_hv_daily(
            connection=connection,
            start_date=start_date,
            end_date=end_date,
        )
        model_dataset = build_model_dataset_daily(
            connection=connection,
            start_date=start_date,
            end_date=end_date,
        )

        iv_len = len(iv_daily)
        hv_len = len(hv_daily)
        dataset_len = len(model_dataset)

        iv_saved = save_iv_daily(connection, iv_daily)
        hv_saved = save_hv_daily(connection, hv_daily)
        dataset_saved = save_model_dataset_daily(connection, model_dataset)

    exported_model_path = None
    exported_iv_path = None
    exported_hv_path = None

    if not args.skip_export:
        exported_model_path = export_model_dataset_daily(
            connection=connection,
            output_path='data/exports/model_dataset_daily.csv',
            start_date=start_date,
            end_date=end_date,
        )
        exported_iv_path = export_iv_daily(
            connection=connection,
            output_path='data/exports/iv_daily.csv',
            start_date=start_date,
            end_date=end_date,
        )
        exported_hv_path = export_hv_daily(
            connection=connection,
            output_path='data/exports/hv_daily.csv',
            start_date=start_date,
            end_date=end_date,
        )

    print(
        f'Pipeline completed | '
        f'start_date={start_date} | '
        f'end_date={end_date}'
    )
    print(
        f'backfill | futures_loaded={futures_loaded} | options_loaded={options_loaded} | '
        f'futures_saved={futures_saved} | options_saved={options_saved}'
    )
    print(
        f'shortlist | series_candidates={series_candidates_len} | contract_candidates={contract_candidates_len} | '
        f'series_saved={series_saved} | contracts_saved={contracts_saved}'
    )
    print(
        f'reference | reference_rows={reference_len} | reference_saved={reference_saved}'
    )
    print(
        f'dataset | iv_rows={iv_len} | hv_rows={hv_len} | model_rows={dataset_len} | '
        f'iv_saved={iv_saved} | hv_saved={hv_saved} | dataset_saved={dataset_saved}'
    )

    if not args.skip_export:
        print(f'export | model_dataset={exported_model_path}')
        print(f'export | iv_daily={exported_iv_path}')
        print(f'export | hv_daily={exported_hv_path}')

    connection.close()


if __name__ == '__main__':
    main()