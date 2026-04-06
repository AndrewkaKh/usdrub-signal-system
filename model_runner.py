"""CLI entry point for model training and inference.

Usage:
    python model_runner.py --retrain
    python model_runner.py --predict --spot 85.50
    python model_runner.py --predict              # spot price auto-detected from DB
    python model_runner.py --eval-walkforward     # baseline walk-forward CV
    python model_runner.py --smile                # vol smile for latest date, both tenors
    python model_runner.py --smile --date 2025-04-01 --tenor 1m

Research commands (new):
    python model_runner.py --analyze-target       # distribution analysis of delta IV
    python model_runner.py --eval-targets         # compare raw_delta / log_return / sigma_norm
    python model_runner.py --eval-thresholds      # 3-class classifier with threshold variants
    python model_runner.py --eval-intervals       # quantile interval forecast evaluation
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

    from model.features import PRODUCT_TARGET_TYPE
    print(f'\n=== Walk-Forward CV Summary  [target: {PRODUCT_TARGET_TYPE}] ===')

    show_cols = ['fold', 'train_end', 'eval_start', 'eval_end', 'n_eval',
                 'rmse_sigma', 'baseline_rmse_sigma', 'beats_baseline_sigma',
                 'rmse_raw', 'sign_accuracy']
    show_cols = [c for c in show_cols if c in results.columns]
    print(results[show_cols].to_string(index=False))

    mean_rmse_s = results['rmse_sigma'].mean()
    mean_base_s = results['baseline_rmse_sigma'].mean()
    beats_s = results['beats_baseline_sigma'].sum()
    print(f'\n[sigma space]  Mean RMSE: {mean_rmse_s:.5f}  Baseline: {mean_base_s:.5f}  '
          f'Beats: {beats_s}/{len(results)}  '
          f'Improvement: {(mean_base_s - mean_rmse_s) / mean_base_s * 100:.1f}%')

    if 'rmse_raw' in results.columns and results['rmse_raw'].notna().any():
        mean_rmse_r = results['rmse_raw'].mean()
        mean_base_r = results['baseline_rmse_raw'].mean()
        beats_r = results['beats_baseline_raw'].sum()
        mean_sign = results['sign_accuracy'].mean()
        print(f'[raw delta IV] Mean RMSE: {mean_rmse_r:.5f}  Baseline: {mean_base_r:.5f}  '
              f'Beats: {beats_r}/{len(results)}  '
              f'Improvement: {(mean_base_r - mean_rmse_r) / mean_base_r * 100:.1f}%  '
              f'SignAcc: {mean_sign:.3f}')


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

    ttype = result.get('target_type', 'unknown')
    print(f'\n=== IV Forecast  [target: {ttype}] ===')
    print(f'As of date        : {result["date"]}')
    print(f'Current IV 1M     : {result["current_iv_1m"]:.4f}  ({result["current_iv_1m"]*100:.2f}%)')
    if result.get('predicted_z') is not None:
        print(f'Predicted z       : {result["predicted_z"]:+.4f}  (sigma_normalized space)')
        print(f'Rolling std (20d) : {result["current_rolling_std"]:.5f}')
        print(f'Predicted Δ IV    : {result["predicted_delta_iv"]:+.5f}  (= z × rolling_std)')
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


def cmd_analyze_target() -> None:
    from model.data_prep import prepare_dataset
    from research.target_analysis import analyze_target, print_analysis

    print(f'Loading dataset from {DATASET_CSV} ...')
    df = prepare_dataset(DATASET_CSV)
    print(f'Dataset ready: {len(df)} rows  ({df["date"].min().date()} – {df["date"].max().date()})')
    print('\nAnalysing target distribution ...')

    result = analyze_target(df)
    print_analysis(result)


def cmd_eval_targets() -> None:
    from model.data_prep import prepare_dataset
    from model.features import FEATURE_COLS
    from research.eval_targets import (
        eval_all_targets,
        summarize_target_results,
        print_target_comparison,
        save_results,
    )

    print(f'Loading dataset from {DATASET_CSV} ...')
    df = prepare_dataset(DATASET_CSV)
    print(f'Dataset ready: {len(df)} rows  ({df["date"].min().date()} – {df["date"].max().date()})')
    print('\nRunning walk-forward CV for all three target variants ...')

    results = eval_all_targets(df, feature_cols=FEATURE_COLS)
    summary = summarize_target_results(results)
    print_target_comparison(results, summary)

    folds_path, summary_path = save_results(results, summary)
    print(f'\nSaved: {folds_path}')
    print(f'       {summary_path}')


def cmd_eval_thresholds() -> None:
    from model.data_prep import prepare_dataset
    from model.features import FEATURE_COLS
    from research.eval_thresholds import (
        eval_all_thresholds,
        summarize_threshold_results,
        print_threshold_comparison,
        save_results,
    )

    print(f'Loading dataset from {DATASET_CSV} ...')
    df = prepare_dataset(DATASET_CSV)
    print(f'Dataset ready: {len(df)} rows  ({df["date"].min().date()} – {df["date"].max().date()})')
    print('\nRunning walk-forward CV for 3-class classification (all threshold variants) ...')

    results = eval_all_thresholds(df, feature_cols=FEATURE_COLS)
    if results.empty:
        print('No results produced (check dataset size).')
        return

    summary = summarize_threshold_results(results)
    print_threshold_comparison(summary)

    folds_path, summary_path = save_results(results, summary)
    print(f'\nSaved: {folds_path}')
    print(f'       {summary_path}')


def cmd_eval_intervals(interval_target: str) -> None:
    from model.data_prep import prepare_dataset
    from model.features import FEATURE_COLS
    from model.targets import SIGMA_NORM, RAW_DELTA, ALL_TARGETS
    from research.eval_intervals import (
        eval_intervals,
        print_interval_summary,
        save_results,
    )
    import pandas as pd

    print(f'Loading dataset from {DATASET_CSV} ...')
    df = prepare_dataset(DATASET_CSV)
    print(f'Dataset ready: {len(df)} rows  ({df["date"].min().date()} – {df["date"].max().date()})')

    targets_to_run = [RAW_DELTA, SIGMA_NORM] if interval_target == 'both' else [interval_target]

    all_folds = []
    all_preds = []

    for tname in targets_to_run:
        print(f'\n--- target: {tname} ---')
        print('Metrics computed in raw-delta-IV space after inverse transform.\n')
        fold_df, pred_df = eval_intervals(
            df, feature_cols=FEATURE_COLS, target_name=tname
        )
        print_interval_summary(fold_df)
        all_folds.append(fold_df)
        all_preds.append(pred_df)

    combined_folds = pd.concat(all_folds, ignore_index=True)
    combined_preds = pd.concat(all_preds, ignore_index=True)

    if interval_target == 'both' and len(targets_to_run) == 2:
        _print_interval_comparison(combined_folds, targets_to_run)

    folds_path, pred_path = save_results(combined_folds, combined_preds)
    print(f'\nSaved: {folds_path}')
    print(f'       {pred_path}')


def _print_interval_comparison(fold_df, targets: list) -> None:
    """Side-by-side comparison of two interval variants across folds."""
    import pandas as pd
    print('\n=== Comparison: interval forecast by target ===')
    print(f'  {"":>28}  {"coverage":>8}  {"width":>10}  {"sign_acc":>8}')
    for tname in targets:
        sub = fold_df[fold_df['target_name'] == tname]
        if sub.empty:
            continue
        cov = sub['empirical_coverage'].mean()
        width = sub['avg_width_delta'].mean()
        sign = sub['sign_accuracy_median'].mean()
        target_cov = sub['target_coverage'].iloc[0]
        print(
            f'  {tname:<28}  {cov:.4f} ({cov-target_cov:+.3f})  '
            f'{width:.6f}  {sign:.4f}'
        )


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
    group.add_argument('--analyze-target', action='store_true',
                       help='Analyse delta-IV distribution and suggest thresholds')
    group.add_argument('--eval-targets', action='store_true',
                       help='Compare raw_delta / log_return / sigma_norm targets (walk-forward)')
    group.add_argument('--eval-thresholds', action='store_true',
                       help='Evaluate 3-class classifier with multiple threshold strategies')
    group.add_argument('--eval-intervals', action='store_true',
                       help='Evaluate quantile interval forecasts (q10/q50/q90)')
    parser.add_argument(
        '--interval-target',
        type=str,
        default='both',
        choices=['raw_delta', 'sigma_normalized_delta', 'both'],
        help='Internal target for --eval-intervals (default: both)',
    )
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
    elif args.analyze_target:
        cmd_analyze_target()
    elif args.eval_targets:
        cmd_eval_targets()
    elif args.eval_thresholds:
        cmd_eval_thresholds()
    elif args.eval_intervals:
        cmd_eval_intervals(args.interval_target)


if __name__ == '__main__':
    main()
