"""Walk-forward evaluation for all three target variants.

For each variant a separate CatBoost regressor is trained on the
corresponding target column.  Predictions are then inverse-transformed
back to raw-delta-IV space so that all three variants are compared on
identical metrics:

  • RMSE   (vs. zero-delta baseline)
  • MAE    (vs. zero-delta baseline)
  • sign accuracy  — fraction of folds where predicted sign == true sign
  • beats_baseline_rmse count across folds

All training uses the same expanding-window walk-forward protocol as the
existing baseline (step=3 m, eval=3 m, min_train=36 m, early_stop=60 rows).

Run via:  python model_runner.py --eval-targets

Outputs (written to data/exports/)
-----------------------------------
eval_targets_folds.csv    — per-fold metrics for all three variants
eval_targets_summary.csv  — aggregated comparison table
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from dateutil.relativedelta import relativedelta

from model.features import FEATURE_COLS
from model.targets import (
    ALL_TARGETS,
    TARGET_COL_MAP,
    TARGET_LABELS,
    RAW_DELTA,
    SIGMA_NORM,
    add_target_variants,
    inverse_transform,
)

_EXPORTS_DIR = Path(__file__).parents[1] / 'data' / 'exports'


# ── helpers ───────────────────────────────────────────────────────────────────

def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _sign_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions with correct direction (excludes y_true == 0)."""
    mask = y_true != 0
    if mask.sum() == 0:
        return float('nan')
    return float((np.sign(y_true[mask]) == np.sign(y_pred[mask])).mean())


# ── per-target walk-forward ───────────────────────────────────────────────────

def _walk_forward_single(
    df: pd.DataFrame,
    target_name: str,
    feature_cols: list[str],
    min_train_months: int,
    step_months: int,
    eval_months: int,
    early_stop_rows: int,
) -> pd.DataFrame:
    """Walk-forward CV for one target variant.

    Predictions are inverse-transformed to raw delta space; all metrics
    are reported in that common space for cross-variant comparability.
    """
    target_col = TARGET_COL_MAP[target_name]

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    date_min = df['date'].min()
    date_max = df['date'].max()

    first_train_end = date_min + relativedelta(months=min_train_months)
    last_train_end = date_max - relativedelta(months=eval_months)

    if first_train_end > last_train_end:
        raise ValueError(
            f'Not enough data for walk-forward CV with target {target_name!r}.'
        )

    feat = [c for c in feature_cols if c in df.columns]

    results = []
    fold = 1
    train_end = first_train_end

    while train_end <= last_train_end:
        eval_start = train_end + relativedelta(days=1)
        eval_end = train_end + relativedelta(months=eval_months)

        # Both target_col (for training) and target_raw_delta (for evaluation)
        # must be non-NaN in the respective splits.
        train_df = df[df['date'] <= train_end].dropna(
            subset=[target_col, 'target_raw_delta']
        )
        eval_df = df[
            (df['date'] >= eval_start) & (df['date'] <= eval_end)
        ].dropna(subset=[target_col, 'target_raw_delta'])

        if len(train_df) < early_stop_rows + 20 or len(eval_df) == 0:
            train_end += relativedelta(months=step_months)
            fold += 1
            continue

        internal_val = train_df.iloc[-early_stop_rows:]
        internal_train = train_df.iloc[:-early_stop_rows]

        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            loss_function='MAE',
            early_stopping_rounds=50,
            random_seed=42,
            verbose=0,
        )
        model.fit(
            Pool(internal_train[feat], internal_train[target_col].values),
            eval_set=Pool(internal_val[feat], internal_val[target_col].values),
        )

        y_pred_native = model.predict(eval_df[feat])

        # ── Inverse transform → raw delta space ──────────────────────────────
        current_iv = eval_df['iv_1m'].values
        rolling_std = (
            eval_df['iv_rolling_std_20'].values
            if 'iv_rolling_std_20' in eval_df.columns
            else None
        )
        y_pred_delta = inverse_transform(
            target_name, y_pred_native, current_iv, rolling_std
        )

        y_true_delta = eval_df['target_raw_delta'].values
        baseline_pred = np.zeros(len(y_true_delta))

        rmse_model = _rmse(y_true_delta, y_pred_delta)
        mae_model = _mae(y_true_delta, y_pred_delta)
        sign_acc = _sign_accuracy(y_true_delta, y_pred_delta)
        baseline_rmse = _rmse(y_true_delta, baseline_pred)
        baseline_mae = _mae(y_true_delta, baseline_pred)

        row = {
            'fold': fold,
            'target': target_name,
            'train_end': train_end.date().isoformat(),
            'eval_start': eval_start.date().isoformat(),
            'eval_end': min(eval_end, date_max).date().isoformat(),
            'n_eval': len(eval_df),
            'rmse': round(rmse_model, 6),
            'mae': round(mae_model, 6),
            'sign_accuracy': round(sign_acc, 4),
            'baseline_rmse': round(baseline_rmse, 6),
            'baseline_mae': round(baseline_mae, 6),
            'beats_baseline_rmse': bool(rmse_model < baseline_rmse),
            'beats_baseline_mae': bool(mae_model < baseline_mae),
        }
        results.append(row)

        beat = 'beats' if rmse_model < baseline_rmse else 'loses'
        print(
            f'  Fold {fold:2d} [{target_name:24s}] '
            f'train->{train_end.date()} '
            f'RMSE={rmse_model:.5f} (base={baseline_rmse:.5f}) '
            f'sign={sign_acc:.3f} {beat}'
        )

        train_end += relativedelta(months=step_months)
        fold += 1

    return pd.DataFrame(results)


# ── public API ────────────────────────────────────────────────────────────────

def eval_all_targets(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    min_train_months: int = 36,
    step_months: int = 3,
    eval_months: int = 3,
    early_stop_rows: int = 60,
) -> pd.DataFrame:
    """Run walk-forward CV for all three target variants.

    Returns a combined DataFrame with one row per (fold, target).
    All metrics are expressed in raw-delta-IV space.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    # Add all target columns + iv_rolling_std_20 to the full dataset
    df = add_target_variants(df)

    frames = []
    for target_name in ALL_TARGETS:
        label = TARGET_LABELS[target_name]
        print(f'\n=== Evaluating {label} ===')
        result = _walk_forward_single(
            df,
            target_name,
            feature_cols,
            min_train_months=min_train_months,
            step_months=step_months,
            eval_months=eval_months,
            early_stop_rows=early_stop_rows,
        )
        frames.append(result)

    return pd.concat(frames, ignore_index=True)


def summarize_target_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate walk-forward results into a per-target comparison table."""
    n_folds_total = results.groupby('target')['fold'].count()

    agg = results.groupby('target').agg(
        n_folds=('fold', 'count'),
        mean_rmse=('rmse', 'mean'),
        mean_mae=('mae', 'mean'),
        mean_sign_acc=('sign_accuracy', 'mean'),
        mean_baseline_rmse=('baseline_rmse', 'mean'),
        beats_baseline_rmse=('beats_baseline_rmse', 'sum'),
        beats_baseline_mae=('beats_baseline_mae', 'sum'),
    ).reset_index()

    agg['rmse_vs_baseline_pct'] = (
        (agg['mean_baseline_rmse'] - agg['mean_rmse'])
        / agg['mean_baseline_rmse'] * 100
    ).round(2)

    # Reorder columns for clarity
    agg = agg[[
        'target', 'n_folds',
        'mean_rmse', 'mean_mae', 'mean_sign_acc',
        'mean_baseline_rmse', 'beats_baseline_rmse', 'beats_baseline_mae',
        'rmse_vs_baseline_pct',
    ]].round(6)

    return agg


def print_target_comparison(results: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Print a formatted comparison table to stdout."""
    print('\n=== Walk-Forward Comparison — All Targets ===')
    print(
        f'  {"Target":<28} {"Folds":>5}  {"RMSE":>9}  {"MAE":>9}  '
        f'{"SignAcc":>7}  {"Beats↑":>6}  {"RMSE vs base":>12}'
    )
    print('  ' + '-' * 90)
    for _, row in summary.iterrows():
        label = TARGET_LABELS.get(row['target'], row['target'])
        print(
            f'  {label:<28} {int(row["n_folds"]):>5}  '
            f'{row["mean_rmse"]:.5f}  {row["mean_mae"]:.5f}  '
            f'{row["mean_sign_acc"]:.3f}    '
            f'{int(row["beats_baseline_rmse"])}/{int(row["n_folds"])}   '
            f'{row["rmse_vs_baseline_pct"]:+.1f}%'
        )

    # Per-fold detail
    print('\n--- Per-fold sign accuracy ---')
    pivot = results.pivot_table(
        values='sign_accuracy', index='fold', columns='target', aggfunc='first'
    )
    print(pivot.round(3).to_string())


def save_results(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path | None = None,
) -> tuple[str, str]:
    if output_dir is None:
        output_dir = _EXPORTS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    folds_path = output_dir / 'eval_targets_folds.csv'
    summary_path = output_dir / 'eval_targets_summary.csv'
    results.to_csv(folds_path, index=False, sep=';')
    summary.to_csv(summary_path, index=False, sep=';')
    return str(folds_path), str(summary_path)
