"""CLI entry point for model training and inference.

Usage:
    python model_runner.py --retrain
    python model_runner.py --predict --spot 85.50
    python model_runner.py --predict              # spot price auto-detected from DB
"""
from __future__ import annotations

import argparse
import sys

DATASET_CSV = 'data/exports/model_dataset_daily.csv'
DB_PATH = 'data/backfill/moex_backfill.sqlite3'


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description='IV model: train or predict next-day implied volatility.'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--retrain', action='store_true', help='Retrain the model')
    group.add_argument('--predict', action='store_true', help='Run inference')
    group.add_argument('--eval-walkforward', action='store_true',
                       help='Walk-forward cross-validation (honest OOS evaluation)')
    parser.add_argument(
        '--spot',
        type=float,
        default=None,
        help='Current USD/RUB spot price (required for --predict if DB unavailable)',
    )
    args = parser.parse_args()

    if args.retrain:
        cmd_retrain()
    elif args.predict:
        cmd_predict(args.spot)
    elif args.eval_walkforward:
        cmd_eval_walkforward()


if __name__ == '__main__':
    main()
