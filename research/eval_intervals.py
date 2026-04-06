"""Interval forecast evaluation via quantile regression.

Three separate CatBoost regressors are trained per fold using the
pinball (quantile) loss:

  q_low  (default 0.10) — lower bound
  q_mid  (default 0.50) — median / point forecast
  q_high (default 0.90) — upper bound

This gives an 80% prediction interval by default.

The internal training target is controlled by *target_name*:

  'sigma_normalized_delta' (default)
      Models are trained on the z-score target (delta / rolling_std20).
      Quantile predictions are inverse-transformed back to raw-delta space
      for evaluation.  Because the normalised target is more stationary,
      the quantile models should learn better-calibrated intervals.

  'raw_delta'
      Models are trained directly on the raw IV delta.  Use this as a
      comparison baseline for the interval quality.

Evaluation metrics are always in raw-delta-IV space regardless of
which internal target is used.

Run via:  python model_runner.py --eval-intervals

Outputs (written to data/exports/)
-----------------------------------
eval_intervals_folds.csv       — per-fold coverage / width / calibration
eval_intervals_predictions.csv — per-row predictions with lower/mid/upper
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from dateutil.relativedelta import relativedelta

from model.features import FEATURE_COLS
from model.targets import (
    RAW_DELTA,
    SIGMA_NORM,
    TARGET_COL_MAP,
    add_target_variants,
    inverse_transform,
)

_EXPORTS_DIR = Path(__file__).parents[1] / 'data' / 'exports'

DEFAULT_QUANTILES = (0.10, 0.50, 0.90)


# ── helpers ───────────────────────────────────────────────────────────────────

def _train_quantile_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_ival: pd.DataFrame,
    y_ival: np.ndarray,
    alpha: float,
) -> CatBoostRegressor:
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function=f'Quantile:alpha={alpha}',
        early_stopping_rounds=50,
        random_seed=42,
        verbose=0,
    )
    model.fit(
        Pool(X_train, y_train),
        eval_set=Pool(X_ival, y_ival),
    )
    return model


def _sign_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return float('nan')
    return float((np.sign(y_true[mask]) == np.sign(y_pred[mask])).mean())


# ── main evaluation ───────────────────────────────────────────────────────────

def eval_intervals(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    quantiles: tuple[float, float, float] = DEFAULT_QUANTILES,
    target_name: str = SIGMA_NORM,
    min_train_months: int = 36,
    step_months: int = 3,
    eval_months: int = 3,
    early_stop_rows: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward interval forecast evaluation.

    Parameters
    ----------
    df           : prepared dataset (output of prepare_dataset).
    feature_cols : feature columns (default: FEATURE_COLS).
    quantiles    : (q_low, q_mid, q_high) — defines the interval.
    target_name  : internal training target — SIGMA_NORM (default) or RAW_DELTA.
                   Predictions are always inverse-transformed to raw-delta space
                   before computing coverage / width / calibration.
    min_train_months, step_months, eval_months, early_stop_rows :
                 walk-forward parameters (same defaults as other evaluators).

    Returns
    -------
    fold_df  : per-fold summary (coverage, width, calibration)
    pred_df  : per-row predictions (lower / mid / upper in delta and IV space)
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    q_low, q_mid, q_high = quantiles
    target_coverage = q_high - q_low

    # Ensure target variant columns (incl. iv_rolling_std_20) are present
    df = add_target_variants(df)

    train_col = TARGET_COL_MAP[target_name]     # column used for model training
    eval_col = TARGET_COL_MAP[RAW_DELTA]        # always evaluate in raw delta space
    # Both must be non-NaN for a row to be usable
    required_cols = list({train_col, eval_col})

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    date_min = df['date'].min()
    date_max = df['date'].max()

    first_train_end = date_min + relativedelta(months=min_train_months)
    last_train_end = date_max - relativedelta(months=eval_months)

    feat = [c for c in feature_cols if c in df.columns]

    fold_rows: list[dict] = []
    pred_rows: list[dict] = []
    fold = 1
    train_end = first_train_end

    while train_end <= last_train_end:
        eval_start = train_end + relativedelta(days=1)
        eval_end = train_end + relativedelta(months=eval_months)

        train_df = df[df['date'] <= train_end].dropna(subset=required_cols)
        eval_df = df[
            (df['date'] >= eval_start) & (df['date'] <= eval_end)
        ].dropna(subset=required_cols)

        if len(train_df) < early_stop_rows + 20 or len(eval_df) == 0:
            train_end += relativedelta(months=step_months)
            fold += 1
            continue

        internal_val = train_df.iloc[-early_stop_rows:]
        internal_train = train_df.iloc[:-early_stop_rows]

        X_train = internal_train[feat]
        y_train = internal_train[train_col].values   # target space for training
        X_ival = internal_val[feat]
        y_ival = internal_val[train_col].values
        X_eval = eval_df[feat]

        # Ground truth always in raw-delta space
        y_eval_delta = eval_df[eval_col].values

        # ── Train three quantile models in target space ───────────────────────
        preds_native: dict[float, np.ndarray] = {}
        for alpha in (q_low, q_mid, q_high):
            m = _train_quantile_model(X_train, y_train, X_ival, y_ival, alpha)
            preds_native[alpha] = m.predict(X_eval)

        # ── Inverse-transform quantile predictions → raw delta space ─────────
        current_iv = eval_df['iv_1m'].values
        rolling_std = (
            eval_df['iv_rolling_std_20'].values
            if 'iv_rolling_std_20' in eval_df.columns
            else None
        )

        lower  = inverse_transform(target_name, preds_native[q_low],  current_iv, rolling_std)
        median = inverse_transform(target_name, preds_native[q_mid],  current_iv, rolling_std)
        upper  = inverse_transform(target_name, preds_native[q_high], current_iv, rolling_std)

        # ── Monotonicity guard ────────────────────────────────────────────────
        # After inverse-transform of a non-linear target (sigma_norm),
        # ordering is not guaranteed row-by-row when rolling_std is small.
        lower  = np.minimum(lower,  median)
        upper  = np.maximum(upper,  median)

        # ── Coverage and width (always in raw delta space) ────────────────────
        in_interval = (y_eval_delta >= lower) & (y_eval_delta <= upper)
        empirical_coverage = float(in_interval.mean())
        avg_width = float(np.mean(upper - lower))
        sign_acc = _sign_accuracy(y_eval_delta, median)

        # ── Calibration ───────────────────────────────────────────────────────
        calib_low  = float((y_eval_delta <= lower).mean())
        calib_mid  = float((y_eval_delta <= median).mean())
        calib_high = float((y_eval_delta <= upper).mean())

        # ── IV-level predictions ──────────────────────────────────────────────
        pred_iv_lower  = current_iv + lower
        pred_iv_median = current_iv + median
        pred_iv_upper  = current_iv + upper

        fold_row = {
            'fold': fold,
            'target_name': target_name,
            'train_end': train_end.date().isoformat(),
            'eval_start': eval_start.date().isoformat(),
            'eval_end': min(eval_end, date_max).date().isoformat(),
            'n_eval': len(eval_df),
            'target_coverage': round(target_coverage, 2),
            'empirical_coverage': round(empirical_coverage, 4),
            'avg_width_delta': round(avg_width, 6),
            'sign_accuracy_median': round(sign_acc, 4) if not np.isnan(sign_acc) else None,
            f'calibration_q{int(q_low*100):02d}': round(calib_low, 4),
            f'calibration_q{int(q_mid*100):02d}': round(calib_mid, 4),
            f'calibration_q{int(q_high*100):02d}': round(calib_high, 4),
        }
        fold_rows.append(fold_row)

        # ── Per-row prediction records ────────────────────────────────────────
        dates = eval_df['date'].values
        for i in range(len(eval_df)):
            pred_rows.append({
                'date': dates[i],
                'fold': fold,
                'target_name': target_name,
                'current_iv': float(current_iv[i]),
                'true_delta': float(y_eval_delta[i]),
                f'pred_delta_q{int(q_low*100):02d}': float(lower[i]),
                'pred_delta_q50': float(median[i]),
                f'pred_delta_q{int(q_high*100):02d}': float(upper[i]),
                'pred_iv_lower': float(pred_iv_lower[i]),
                'pred_iv_median': float(pred_iv_median[i]),
                'pred_iv_upper': float(pred_iv_upper[i]),
                'in_interval': bool(in_interval[i]),
            })

        cov_delta = empirical_coverage - target_coverage
        sign_str = f'{sign_acc:.3f}' if not np.isnan(sign_acc) else ' n/a'
        print(
            f'  Fold {fold:2d} [{target_name}] '
            f'coverage={empirical_coverage:.3f} (target={target_coverage:.2f}, Δ={cov_delta:+.3f}) '
            f'| width={avg_width:.5f} | sign={sign_str} '
            f'| cal=[{calib_low:.2f}, {calib_mid:.2f}, {calib_high:.2f}]'
        )

        train_end += relativedelta(months=step_months)
        fold += 1

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.DataFrame(pred_rows)
    return fold_df, pred_df


# ── Summary and output ────────────────────────────────────────────────────────

def print_interval_summary(fold_df: pd.DataFrame) -> None:
    """Print calibration and coverage summary to stdout."""
    if fold_df.empty:
        print('No interval results.')
        return

    target_cov = fold_df['target_coverage'].iloc[0]
    tname = fold_df.get('target_name', pd.Series(['?'])).iloc[0]
    print(f'\n=== Interval Forecast Summary ===')
    print(f'    Internal target : {tname}')
    print(f'    Target coverage : {target_cov:.0%}')
    print(
        f'  {"Fold":>4}  {"Coverage":>8}  {"Width(Δ)":>10}  '
        f'{"SignAcc":>7}  {"CalLow":>7}  {"CalMid":>7}  {"CalHigh":>7}'
    )
    print('  ' + '-' * 68)

    cal_cols = sorted(c for c in fold_df.columns if c.startswith('calibration_'))

    for _, row in fold_df.iterrows():
        sign_str = f'{row["sign_accuracy_median"]:.3f}' if pd.notna(row.get('sign_accuracy_median')) else '  n/a'
        cal_vals = [f'{row[c]:.3f}' for c in cal_cols]
        print(
            f'  {int(row["fold"]):>4}  {row["empirical_coverage"]:.4f}    '
            f'{row["avg_width_delta"]:.6f}  '
            f'{sign_str}   '
            + '  '.join(cal_vals)
        )

    print('  ' + '-' * 68)
    print(
        f'  {"Mean":>4}  {fold_df["empirical_coverage"].mean():.4f}    '
        f'{fold_df["avg_width_delta"].mean():.6f}  '
        f'{fold_df["sign_accuracy_median"].mean():.3f}   '
        + '  '.join(f'{fold_df[c].mean():.3f}' for c in cal_cols)
    )

    mean_coverage = fold_df['empirical_coverage'].mean()
    coverage_gap = mean_coverage - target_cov
    print(
        f'\n  Coverage gap: {coverage_gap:+.3f} '
        f'(+ = over-covered, - = under-covered)'
    )


def save_results(
    fold_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    output_dir: Path | None = None,
) -> tuple[str, str]:
    if output_dir is None:
        output_dir = _EXPORTS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    folds_path = output_dir / 'eval_intervals_folds.csv'
    pred_path = output_dir / 'eval_intervals_predictions.csv'
    fold_df.to_csv(folds_path, index=False, sep=';')
    pred_df.to_csv(pred_path, index=False, sep=';')
    return str(folds_path), str(pred_path)
