"""3-class directional classification with threshold-based labelling.

Classes
-------
  0 : down  (delta_iv < -threshold)
  1 : flat  (|delta_iv| <= threshold)
  2 : up    (delta_iv > +threshold)

Threshold strategies
--------------------
  fixed       — absolute value (e.g. 0.005 IV points)
  quantile    — flat zone covers a target fraction of training observations
  sigma       — flat zone = |delta| < k * rolling_std_20

The base target for classification is always raw_delta IV (the most
natural space for direction), regardless of which regression target
variant performs best.  Classification is compared to the regression
baseline.

Walk-forward protocol is identical to the regression experiments:
  min_train=36 m, step=3 m, eval=3 m, early_stop=60 rows.

Metrics per fold
----------------
  overall_accuracy, balanced_accuracy, macro_f1,
  acc_no_flat (accuracy on down / up only), class distribution

Run via:  python model_runner.py --eval-thresholds

Outputs (written to data/exports/)
-----------------------------------
eval_thresholds_folds.csv    — per-fold metrics for all threshold variants
eval_thresholds_summary.csv  — aggregated comparison table
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from dateutil.relativedelta import relativedelta
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)

from model.features import FEATURE_COLS
from model.targets import add_target_variants

_EXPORTS_DIR = Path(__file__).parents[1] / 'data' / 'exports'

# ── Class encoding ────────────────────────────────────────────────────────────
# 0 = down, 1 = flat, 2 = up
CLASS_DOWN = 0
CLASS_FLAT = 1
CLASS_UP = 2
CLASS_NAMES = {CLASS_DOWN: 'down', CLASS_FLAT: 'flat', CLASS_UP: 'up'}

# ── Threshold variants to evaluate ───────────────────────────────────────────
# Format: name → (type, param)
#   type='fixed'    param = absolute IV threshold
#   type='quantile' param = fraction assigned to flat zone
#   type='sigma'    param = multiplier on rolling_std_20
THRESHOLD_VARIANTS: dict[str, tuple[str, float]] = {
    'fixed_003':    ('fixed',    0.003),
    'fixed_005':    ('fixed',    0.005),
    'fixed_010':    ('fixed',    0.010),
    'quantile_20':  ('quantile', 0.20),   # middle 20 % → flat
    'quantile_30':  ('quantile', 0.30),   # middle 30 % → flat
    'sigma_050':    ('sigma',    0.50),
    'sigma_075':    ('sigma',    0.75),
}


# ── Label construction ────────────────────────────────────────────────────────

def _build_labels(
    delta: pd.Series,
    rolling_std: pd.Series | None,
    threshold_type: str,
    threshold_param: float,
    *,
    train_delta: pd.Series | None = None,
) -> np.ndarray:
    """Return integer class labels (0/1/2) for *delta* values.

    For quantile-based thresholds the boundaries are computed on
    *train_delta* to prevent leakage into the evaluation set.
    """
    labels = np.full(len(delta), CLASS_FLAT, dtype=np.int8)

    if threshold_type == 'fixed':
        t = threshold_param
        labels[delta.values > t] = CLASS_UP
        labels[delta.values < -t] = CLASS_DOWN

    elif threshold_type == 'quantile':
        # Use training-set delta to compute boundaries
        ref = train_delta if train_delta is not None else delta
        lo = float(ref.quantile(threshold_param / 2))
        hi = float(ref.quantile(1 - threshold_param / 2))
        labels[delta.values > hi] = CLASS_UP
        labels[delta.values < lo] = CLASS_DOWN

    elif threshold_type == 'sigma':
        if rolling_std is None:
            raise ValueError('rolling_std is required for sigma-based threshold')
        t = threshold_param * rolling_std.values
        labels[delta.values > t] = CLASS_UP
        labels[delta.values < -t] = CLASS_DOWN

    else:
        raise ValueError(f'Unknown threshold type: {threshold_type!r}')

    return labels


# ── Per-threshold walk-forward ────────────────────────────────────────────────

def _walk_forward_classification(
    df: pd.DataFrame,
    threshold_name: str,
    threshold_type: str,
    threshold_param: float,
    feature_cols: list[str],
    min_train_months: int,
    step_months: int,
    eval_months: int,
    early_stop_rows: int,
) -> pd.DataFrame:
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    date_min = df['date'].min()
    date_max = df['date'].max()

    first_train_end = date_min + relativedelta(months=min_train_months)
    last_train_end = date_max - relativedelta(months=eval_months)

    feat = [c for c in feature_cols if c in df.columns]
    has_rolling_std = 'iv_rolling_std_20' in df.columns

    results = []
    fold = 1
    train_end = first_train_end

    while train_end <= last_train_end:
        eval_start = train_end + relativedelta(days=1)
        eval_end = train_end + relativedelta(months=eval_months)

        train_df = df[df['date'] <= train_end].dropna(subset=['target_raw_delta'])
        eval_df = df[
            (df['date'] >= eval_start) & (df['date'] <= eval_end)
        ].dropna(subset=['target_raw_delta'])

        if len(train_df) < early_stop_rows + 20 or len(eval_df) == 0:
            train_end += relativedelta(months=step_months)
            fold += 1
            continue

        # ── Build labels ──────────────────────────────────────────────────────
        train_rolling = train_df['iv_rolling_std_20'] if has_rolling_std else None
        eval_rolling = eval_df['iv_rolling_std_20'] if has_rolling_std else None

        y_train_all = _build_labels(
            train_df['target_raw_delta'],
            train_rolling,
            threshold_type,
            threshold_param,
            train_delta=train_df['target_raw_delta'],  # always use train for quantile
        )
        y_eval = _build_labels(
            eval_df['target_raw_delta'],
            eval_rolling,
            threshold_type,
            threshold_param,
            train_delta=train_df['target_raw_delta'],  # ← no leakage
        )

        # Internal train / val split
        n_ival = early_stop_rows
        y_ival = y_train_all[-n_ival:]
        y_int_train = y_train_all[:-n_ival]
        X_int_train = train_df.iloc[:-n_ival][feat]
        X_ival = train_df.iloc[-n_ival:][feat]
        X_eval = eval_df[feat]

        # Need at least 2 classes in internal train to fit
        if len(np.unique(y_int_train)) < 2:
            train_end += relativedelta(months=step_months)
            fold += 1
            continue

        model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            loss_function='MultiClass',
            classes_count=3,
            early_stopping_rounds=50,
            random_seed=42,
            verbose=0,
        )
        model.fit(
            Pool(X_int_train, y_int_train),
            eval_set=Pool(X_ival, y_ival),
        )

        y_pred = model.predict(X_eval).flatten().astype(int)

        # ── Metrics ───────────────────────────────────────────────────────────
        overall_acc = float(accuracy_score(y_eval, y_pred))
        balanced_acc = float(balanced_accuracy_score(y_eval, y_pred))
        macro_f1 = float(f1_score(y_eval, y_pred, average='macro', zero_division=0))

        # Accuracy on directional cases only (exclude flat)
        dir_mask = y_eval != CLASS_FLAT
        acc_no_flat = (
            float(accuracy_score(y_eval[dir_mask], y_pred[dir_mask]))
            if dir_mask.sum() > 0 else float('nan')
        )

        # Class distribution in eval set
        unique, counts = np.unique(y_eval, return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))

        row = {
            'fold': fold,
            'threshold': threshold_name,
            'train_end': train_end.date().isoformat(),
            'eval_start': eval_start.date().isoformat(),
            'eval_end': min(eval_end, date_max).date().isoformat(),
            'n_eval': len(eval_df),
            'n_down': dist.get(CLASS_DOWN, 0),
            'n_flat': dist.get(CLASS_FLAT, 0),
            'n_up': dist.get(CLASS_UP, 0),
            'overall_accuracy': round(overall_acc, 4),
            'balanced_accuracy': round(balanced_acc, 4),
            'macro_f1': round(macro_f1, 4),
            'acc_no_flat': round(acc_no_flat, 4) if not np.isnan(acc_no_flat) else None,
        }
        results.append(row)

        no_flat_str = f'{acc_no_flat:.3f}' if not np.isnan(acc_no_flat) else ' n/a'
        print(
            f'  Fold {fold:2d} [{threshold_name:12s}] '
            f'acc={overall_acc:.3f}  bal={balanced_acc:.3f}  '
            f'f1={macro_f1:.3f}  dir={no_flat_str}  '
            f'(↓{dist.get(CLASS_DOWN,0)} ={dist.get(CLASS_FLAT,0)} ↑{dist.get(CLASS_UP,0)})'
        )

        train_end += relativedelta(months=step_months)
        fold += 1

    return pd.DataFrame(results)


# ── Public API ────────────────────────────────────────────────────────────────

def eval_all_thresholds(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    threshold_variants: dict | None = None,
    min_train_months: int = 36,
    step_months: int = 3,
    eval_months: int = 3,
    early_stop_rows: int = 60,
) -> pd.DataFrame:
    """Run walk-forward classification CV for all threshold strategies.

    Returns a combined DataFrame with one row per (fold, threshold).
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    if threshold_variants is None:
        threshold_variants = THRESHOLD_VARIANTS

    # Adds target_raw_delta + iv_rolling_std_20 to the dataset
    df = add_target_variants(df)

    frames = []
    for tname, (ttype, tparam) in threshold_variants.items():
        print(f'\n=== Threshold: {tname}  ({ttype}, param={tparam}) ===')
        result = _walk_forward_classification(
            df, tname, ttype, tparam, feature_cols,
            min_train_months=min_train_months,
            step_months=step_months,
            eval_months=eval_months,
            early_stop_rows=early_stop_rows,
        )
        frames.append(result)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_threshold_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-fold results into a threshold comparison table."""
    agg = results.groupby('threshold').agg(
        n_folds=('fold', 'count'),
        mean_overall_acc=('overall_accuracy', 'mean'),
        mean_balanced_acc=('balanced_accuracy', 'mean'),
        mean_macro_f1=('macro_f1', 'mean'),
        mean_acc_no_flat=('acc_no_flat', 'mean'),
        mean_n_down=('n_down', 'mean'),
        mean_n_flat=('n_flat', 'mean'),
        mean_n_up=('n_up', 'mean'),
    ).reset_index().round(4)
    return agg


def print_threshold_comparison(summary: pd.DataFrame) -> None:
    """Print a formatted comparison table to stdout."""
    print('\n=== Walk-Forward Comparison — Threshold Variants ===')
    print(
        f'  {"Threshold":<14} {"Folds":>5}  {"OvAcc":>6}  '
        f'{"BalAcc":>6}  {"MacroF1":>7}  {"DirAcc":>6}  '
        f'{"n_flat/fold":>10}'
    )
    print('  ' + '-' * 70)
    for _, row in summary.iterrows():
        flat_str = f'{row["mean_n_flat"]:.0f}'
        no_flat = row['mean_acc_no_flat']
        no_flat_str = f'{no_flat:.3f}' if pd.notna(no_flat) else '  n/a'
        print(
            f'  {row["threshold"]:<14} {int(row["n_folds"]):>5}  '
            f'{row["mean_overall_acc"]:.3f}   '
            f'{row["mean_balanced_acc"]:.3f}   '
            f'{row["mean_macro_f1"]:.4f}   '
            f'{no_flat_str}   '
            f'{flat_str:>10}'
        )


def save_results(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path | None = None,
) -> tuple[str, str]:
    if output_dir is None:
        output_dir = _EXPORTS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    folds_path = output_dir / 'eval_thresholds_folds.csv'
    summary_path = output_dir / 'eval_thresholds_summary.csv'
    results.to_csv(folds_path, index=False, sep=';')
    summary.to_csv(summary_path, index=False, sep=';')
    return str(folds_path), str(summary_path)
