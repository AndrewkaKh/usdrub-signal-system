"""CLI entry point for model training and inference.

Usage:
    python model_runner.py --retrain
    python model_runner.py --predict --spot 85.50
    python model_runner.py --predict              # spot price auto-detected from DB
    python model_runner.py --smile                # vol smile for latest date, both tenors
    python model_runner.py --smile --date 2025-04-01 --tenor 1m
"""
from __future__ import annotations

import argparse
import sqlite3
import sys

DATASET_CSV = 'data/exports/model_dataset_daily.csv'
DB_PATH = 'data/backfill/moex_backfill.sqlite3'
SMILE_CSV = 'data/exports/iv_smile_daily.csv'
SMILE_METRICS_CSV = 'data/exports/iv_smile_metrics.csv'


def cmd_eval_walkforward() -> None:
    from model.data_prep import prepare_dataset
    from model.train import walk_forward_cv
    from model.features import FEATURE_COLS

    print(f'Loading dataset from {DATASET_CSV} ...')
    df = prepare_dataset(DATASET_CSV)
    print(f'Dataset ready: {len(df)} rows  ({df["date"].min().date()} – {df["date"].max().date()})')
    print('\nRunning walk-forward cross-validation (step=3m, eval=3m, min_train=36m) ...\n')

    results = walk_forward_cv(df, feature_cols=FEATURE_COLS)

    print('\n=== Walk-Forward CV Summary ===')
    print(results[['fold', 'train_end', 'eval_start', 'eval_end', 'n_eval',
                    'rmse', 'baseline_rmse', 'beats_baseline']].to_string(index=False))

    mean_rmse = results['rmse'].mean()
    mean_baseline = results['baseline_rmse'].mean()
    beats_n = results['beats_baseline'].sum()
    print(f'\nMean RMSE      : {mean_rmse:.5f}')
    print(f'Mean baseline  : {mean_baseline:.5f}')
    print(f'Beats baseline : {beats_n}/{len(results)} folds')
    print(f'Improvement    : {(mean_baseline - mean_rmse) / mean_baseline * 100:.1f}%')


def cmd_retrain() -> None:
    from model.data_prep import prepare_dataset, split_dataset
    from model.train import train_model, evaluate_and_save
    from model.features import FEATURE_COLS

    print(f'Loading dataset from {DATASET_CSV} ...')
    df = prepare_dataset(DATASET_CSV)
    print(f'Dataset ready: {len(df)} rows  ({df["date"].min().date()} - {df["date"].max().date()})')

    train_df, val_df, test_df = split_dataset(df)
    print(f'Split: train={len(train_df)}  val={len(val_df)}  test={len(test_df)}')

    print('\nTraining CatBoost model ...')
    model = train_model(train_df, val_df, feature_cols=FEATURE_COLS)

    evaluate_and_save(
        model=model,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=FEATURE_COLS,
        dataset_path=DATASET_CSV,
    )


def cmd_predict(spot: float | None) -> None:
    from model.predict import predict_next_day, get_spot_price_from_db

    if spot is None:
        spot = get_spot_price_from_db(DB_PATH)
        if spot is None:
            print(
                'ERROR: Could not determine spot price from DB. '
                'Pass it explicitly with --spot <value>.',
                file=sys.stderr,
            )
            sys.exit(1)
        print(f'Spot price (from DB): {spot}')

    result = predict_next_day(
        dataset_csv_path=DATASET_CSV,
        spot_price=spot,
        confidence=0.90,
    )

    print('\n=== IV Forecast ===')
    print(f'As of date        : {result["date"]}')
    print(f'Current IV 1M     : {result["current_iv_1m"]:.4f}  ({result["current_iv_1m"]*100:.2f}%)')
    print(f'Predicted IV 1M   : {result["predicted_iv_1m"]:.4f}  ({result["predicted_iv_1m"]*100:.2f}%)')
    print(f'IV change (pred)  : {result["iv_change_pct"]:+.2f}%')
    print(f'\n=== Expected USD/RUB range (next trading day, {int(result["confidence"]*100)}% CI) ===')
    print(f'Spot price        : {result["spot_price"]:.4f}')
    print(f'Lower bound       : {result["range_lower"]:.4f}')
    print(f'Upper bound       : {result["range_upper"]:.4f}')
    print(f'Expected move     : +/-{result["move_pct"]:.2f}%')
    print(f'\nModel trained at  : {result["model_trained_at"]}')

    return result


def _fmt_iv(value: float | None) -> str:
    """Format IV as percentage string, or '---' if unavailable."""
    if value is None:
        return '   --- '
    return f'{value * 100:6.2f}%'


def _fmt_metric(value: float | None, pct: bool = True) -> str:
    """Format a smile metric (pct=True → multiply by 100 for display)."""
    if value is None or (isinstance(value, float) and (value != value)):  # NaN check
        return 'n/a'
    if pct:
        return f'{value * 100:+.2f}%'
    return f'{value:.4f}'


def _resolve_smile_date(connection, tenor: str) -> str | None:
    """Return latest date in option_contract_candidates with valid data for tenor."""
    cursor = connection.execute(
        '''
        SELECT MAX(date) FROM option_contract_candidates
        WHERE target_tenor = ?
        ''',
        (tenor,),
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else None


def cmd_smile(date_str: str | None, tenor_filter: str | None) -> None:
    from processing.dataset.iv_smile_builder import build_iv_smile
    from processing.dataset.smile_metrics import compute_smile_metrics

    tenors = [tenor_filter] if tenor_filter else ['1m', '3m']

    conn = sqlite3.connect(DB_PATH)
    try:
        # Resolve date
        if date_str is None:
            # Find latest date that has data for at least one requested tenor
            candidate_dates = [_resolve_smile_date(conn, t) for t in tenors]
            candidate_dates = [d for d in candidate_dates if d]
            if not candidate_dates:
                print('ERROR: No smile data found in DB.', file=sys.stderr)
                sys.exit(1)
            date_str = max(candidate_dates)
            print(f'Latest available date: {date_str}')

        smile_df = build_iv_smile(conn, date_str, date_str)
    finally:
        conn.close()

    if smile_df.empty:
        print(f'ERROR: No smile data found for date {date_str}.', file=sys.stderr)
        sys.exit(1)

    # Filter by tenor if requested
    if tenor_filter:
        smile_df = smile_df[smile_df['tenor'] == tenor_filter]
        if smile_df.empty:
            print(
                f'ERROR: No smile data for tenor={tenor_filter} on {date_str}.',
                file=sys.stderr,
            )
            sys.exit(1)

    # Save CSVs
    smile_df.to_csv(SMILE_CSV, index=False, sep=';')
    metrics_df = compute_smile_metrics(smile_df)
    metrics_df.to_csv(SMILE_METRICS_CSV, index=False, sep=';')

    # Print per-tenor smile table
    import math
    for tenor, group in smile_df.groupby('tenor'):
        group = group.sort_values('strike').copy()
        futures_price = group['futures_price'].iloc[0]
        spot_proxy = futures_price / 1000.0  # MOEX Si contract: 1 lot = 1000 USD

        print(f'\n=== Volatility Smile: {date_str} | Tenor {tenor} | F={spot_proxy:.4f} (futures {int(futures_price)}) ===\n')
        print(f'  {"Delta":>6}  {"Strike":>8}  {"Call IV":>8}  {"Put IV":>8}  {"Mid IV":>8}  Status')
        print(f'  {"-"*6}  {"-"*8}  {"-"*8}  {"-"*8}  {"-"*8}  ------')

        for _, row in group.iterrows():
            delta_str = f'{row["delta_call"]:6.2f}' if row['delta_call'] is not None else '   --- '
            strike_str = f'{int(row["strike"]):8d}'
            # Mark ATM row (|moneyness| closest to 0)
            atm_marker = ' <- ATM' if abs(row['moneyness']) == group['moneyness'].abs().min() else ''
            print(
                f'  {delta_str}  {strike_str}  '
                f'{_fmt_iv(row["call_iv"])}  '
                f'{_fmt_iv(row["put_iv"])}  '
                f'{_fmt_iv(row["mid_iv"])}  '
                f'{row["status"]}{atm_marker}'
            )

        # Print smile metrics for this tenor
        m_row = metrics_df[metrics_df['tenor'] == tenor]
        if not m_row.empty:
            m = m_row.iloc[0]
            print(f'\n  Smile metrics ({int(m["n_strikes"])} strikes):')
            print(f'    ATM vol : {_fmt_iv(m["atm_vol"])}')
            rr25_skew = ' (put skew)' if (isinstance(m['rr25'], float) and m['rr25'] == m['rr25'] and m['rr25'] < 0) else ''
            print(f'    RR25    : {_fmt_metric(m["rr25"])}{rr25_skew}')
            print(f'    BF25    : {_fmt_metric(m["bf25"])}')
            print(f'    RR10    : {_fmt_metric(m["rr10"])}')

    print(f'\nSaved: {SMILE_CSV}')
    print(f'       {SMILE_METRICS_CSV}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='IV model: train or predict next-day implied volatility.'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--retrain', action='store_true', help='Retrain the model')
    group.add_argument('--predict', action='store_true', help='Run inference')
    group.add_argument('--eval-walkforward', action='store_true',
                       help='Walk-forward cross-validation (honest OOS evaluation)')
    group.add_argument('--smile', action='store_true',
                       help='Compute and display volatility smile (per-strike IV)')
    parser.add_argument(
        '--spot',
        type=float,
        default=None,
        help='Current USD/RUB spot price (required for --predict if DB unavailable)',
    )
    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help='Date for --smile in YYYY-MM-DD format (default: latest available)',
    )
    parser.add_argument(
        '--tenor',
        type=str,
        default=None,
        choices=['1m', '3m'],
        help='Tenor filter for --smile (default: both 1m and 3m)',
    )
    args = parser.parse_args()

    if args.retrain:
        cmd_retrain()
    elif args.predict:
        cmd_predict(args.spot)
    elif args.eval_walkforward:
        cmd_eval_walkforward()
    elif args.smile:
        cmd_smile(args.date, args.tenor)


if __name__ == '__main__':
    main()
