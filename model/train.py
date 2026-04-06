from __future__ import annotations

import json
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from .features import FEATURE_COLS, TARGET_COL, TARGET_RAW_DELTA_COL, PRODUCT_TARGET_TYPE

ARTIFACTS_DIR = Path(__file__).parent / 'artifacts'
MODEL_PATH = ARTIFACTS_DIR / 'catboost_iv_1m.cbm'
METADATA_PATH = ARTIFACTS_DIR / 'metadata.json'


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        'rmse': round(_rmse(y_true, y_pred), 6),
        'mae': round(_mae(y_true, y_pred), 6),
        'mape': round(_mape(y_true, y_pred), 4),
    }


def _prepare_pool(df: pd.DataFrame, feature_cols: list[str]) -> Pool:
    present = [c for c in feature_cols if c in df.columns]
    X = df[present].copy()
    y = df[TARGET_COL].values
    return Pool(X, label=y)


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> CatBoostRegressor:
    """Train CatBoost regressor with early stopping on val_df.

    Saves model and metadata.json to model/artifacts/.
    Returns the fitted model.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    # Drop rows where target is NaN (last row of dataset has no next-day target)
    train_df = train_df.dropna(subset=[TARGET_COL])
    val_df = val_df.dropna(subset=[TARGET_COL])

    # Only use features that are present in the data
    feature_cols = [c for c in feature_cols if c in train_df.columns]

    train_pool = _prepare_pool(train_df, feature_cols)
    val_pool = _prepare_pool(val_df, feature_cols)

    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function='MAE',
        early_stopping_rounds=50,
        random_seed=42,
        verbose=50,
    )
    model.fit(train_pool, eval_set=val_pool)

    return model


def evaluate_and_save(
    model: CatBoostRegressor,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    dataset_path: str = 'data/exports/model_dataset_daily.csv',
) -> dict:
    """Compute metrics, print summary, save model + metadata.json."""
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    feature_cols = [c for c in feature_cols if c in train_df.columns]

    def _predict(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        df = df.dropna(subset=[TARGET_COL])
        X = df[[c for c in feature_cols if c in df.columns]]
        return df[TARGET_COL].values, model.predict(X)

    val_true, val_pred = _predict(val_df)
    test_true, test_pred = _predict(test_df)
    train_clean = train_df.dropna(subset=[TARGET_COL])

    # Naive baseline for delta target: predict 0 (no change in IV)
    baseline_test_pred = np.zeros(len(test_true))
    baseline_rmse = round(_rmse(test_true, baseline_test_pred), 6)

    metrics = {
        'val': _compute_metrics(val_true, val_pred),
        'test': _compute_metrics(test_true, test_pred),
    }

    # Feature importance (top 15)
    fi = dict(
        zip(feature_cols, model.get_feature_importance())
    )
    top15 = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:15]

    # ── Raw-delta metrics for interpretability ────────────────────────────────
    # Convert sigma_normalized predictions back to raw-delta space so output
    # is directly comparable with the historical baseline numbers.
    raw_metrics: dict = {}
    test_clean2 = test_df.dropna(subset=[TARGET_COL])
    has_raw = (
        'iv_rolling_std_20' in test_clean2.columns
        and TARGET_RAW_DELTA_COL in test_clean2.columns
    )
    if has_raw:
        rolling_std_test = test_clean2['iv_rolling_std_20'].values
        pred_delta_test = test_pred * rolling_std_test
        true_delta_test = test_clean2[TARGET_RAW_DELTA_COL].values
        raw_metrics = {
            'rmse': round(_rmse(true_delta_test, pred_delta_test), 6),
            'mae': round(_mae(true_delta_test, pred_delta_test), 6),
        }
        raw_baseline_rmse = round(_rmse(true_delta_test, np.zeros(len(true_delta_test))), 6)

    # Print summary
    print('\n=== Training complete ===')
    print(f'Target: {PRODUCT_TARGET_TYPE}  (production target)')
    print(f'Train rows: {len(train_clean)}  |  Val rows: {len(val_df)}  |  Test rows: {len(test_df)}')
    print()
    print(f'  [sigma space]  Val  RMSE: {metrics["val"]["rmse"]:.4f}  MAE: {metrics["val"]["mae"]:.4f}')
    print(f'  [sigma space]  Test RMSE: {metrics["test"]["rmse"]:.4f}  MAE: {metrics["test"]["mae"]:.4f}')
    print(f'  [sigma space]  Baseline (predict z=0): RMSE = {baseline_rmse:.4f}')
    if raw_metrics:
        print()
        print(f'  [raw delta IV] Test RMSE: {raw_metrics["rmse"]:.6f}  MAE: {raw_metrics["mae"]:.6f}')
        print(f'  [raw delta IV] Baseline (predict Δ=0): RMSE = {raw_baseline_rmse:.6f}')
        pct = (raw_baseline_rmse - raw_metrics["rmse"]) / raw_baseline_rmse * 100
        print(f'  [raw delta IV] Model vs baseline: {pct:+.1f}%')
    print('\nTop-15 feature importances:')
    for name, score in top15:
        print(f'  {name:<30} {score:.2f}')

    # Save artifacts
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))

    train_dates = train_clean['date'].astype(str)
    val_dates = val_df['date'].astype(str)
    test_clean = test_df.dropna(subset=[TARGET_COL])
    test_dates = test_clean['date'].astype(str)

    metadata = {
        'trained_at': datetime.now().isoformat(timespec='seconds'),
        'target_col': TARGET_COL,
        'target_type': PRODUCT_TARGET_TYPE,
        'target_note': (
            'Model predicts z = delta_iv / rolling_std_20. '
            'Inverse transform: pred_delta = z * iv_rolling_std_20 (from dataset last row). '
            'pred_iv = current_iv + pred_delta.'
        ),
        'feature_cols': feature_cols,
        'train_period': {
            'start': train_dates.min(),
            'end': train_dates.max(),
        },
        'val_period': {
            'start': val_dates.min(),
            'end': val_dates.max(),
        },
        'test_period': {
            'start': test_dates.min(),
            'end': test_dates.max(),
        },
        'metrics_sigma_space': metrics,
        'metrics_raw_delta': raw_metrics if raw_metrics else {},
        'baseline_test_rmse_sigma': baseline_rmse,
        'baseline_test_rmse_raw': raw_baseline_rmse if has_raw else None,
        'n_train_rows': len(train_clean),
        'n_val_rows': len(val_df),
        'n_test_rows': len(test_clean),
        'dataset_version': os.path.basename(dataset_path),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    print(f'\nModel saved: {MODEL_PATH}')
    print(f'Metadata:    {METADATA_PATH}')

    return metadata


def walk_forward_cv(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    min_train_months: int = 36,
    step_months: int = 3,
    eval_months: int = 3,
    early_stop_rows: int = 60,
) -> pd.DataFrame:
    """Walk-forward cross-validation for time series.

    Trains on an expanding window, evaluates on the next ``eval_months`` out-of-sample.
    Early stopping uses the last ``early_stop_rows`` of each training window as
    an internal validation set.

    Parameters
    ----------
    df:
        Full prepared dataset (output of prepare_dataset).
    feature_cols:
        Feature columns to use (defaults to FEATURE_COLS).
    min_train_months:
        Minimum months of data before the first evaluation window.
    step_months:
        How many months to advance per fold.
    eval_months:
        Length of each evaluation window in months.
    early_stop_rows:
        Number of rows at the end of each training window used for early stopping.

    Returns
    -------
    pd.DataFrame with one row per fold: fold, train_end, eval_start, eval_end,
    n_train, n_eval, rmse, mae, mape, baseline_rmse.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    feature_cols = [c for c in feature_cols if c in df.columns]

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    date_min = df['date'].min()
    date_max = df['date'].max()

    # First training cutoff: min_train_months after start
    first_train_end = date_min + relativedelta(months=min_train_months)
    # Last possible training cutoff: leave at least eval_months for final fold
    last_train_end = date_max - relativedelta(months=eval_months)

    if first_train_end > last_train_end:
        raise ValueError(
            f'Not enough data for walk-forward CV: need at least '
            f'{min_train_months + eval_months} months, have '
            f'{(date_max - date_min).days // 30} months.'
        )

    results = []
    fold = 1
    train_end = first_train_end

    while train_end <= last_train_end:
        eval_start = train_end + relativedelta(days=1)
        eval_end = train_end + relativedelta(months=eval_months)

        train_df = df[df['date'] <= train_end].dropna(subset=[TARGET_COL])
        eval_df = df[(df['date'] >= eval_start) & (df['date'] <= eval_end)].dropna(
            subset=[TARGET_COL]
        )

        if len(train_df) < early_stop_rows + 20 or len(eval_df) == 0:
            train_end += relativedelta(months=step_months)
            fold += 1
            continue

        # Internal val for early stopping = last early_stop_rows of training
        internal_val = train_df.iloc[-early_stop_rows:]
        internal_train = train_df.iloc[:-early_stop_rows]

        X_train = internal_train[feature_cols]
        y_train = internal_train[TARGET_COL].values
        X_ival = internal_val[feature_cols]
        y_ival = internal_val[TARGET_COL].values

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
            Pool(X_train, y_train),
            eval_set=Pool(X_ival, y_ival),
        )

        X_eval = eval_df[feature_cols]
        y_eval = eval_df[TARGET_COL].values   # sigma_normalized space
        y_pred = model.predict(X_eval)

        # Baseline: predict 0 (no IV change → z = 0 in sigma space)
        baseline_pred = np.zeros(len(y_eval))

        metrics = _compute_metrics(y_eval, y_pred)
        baseline_rmse = round(_rmse(y_eval, baseline_pred), 6)

        # ── Raw-delta metrics (for interpretability) ──────────────────────────
        raw_rmse = raw_baseline_rmse = raw_sign_acc = float('nan')
        if (
            'iv_rolling_std_20' in eval_df.columns
            and TARGET_RAW_DELTA_COL in eval_df.columns
        ):
            rolling_std = eval_df['iv_rolling_std_20'].values
            pred_delta = y_pred * rolling_std
            true_delta = eval_df[TARGET_RAW_DELTA_COL].values
            raw_rmse = round(_rmse(true_delta, pred_delta), 6)
            raw_baseline_rmse = round(_rmse(true_delta, np.zeros(len(true_delta))), 6)
            nz = true_delta != 0
            raw_sign_acc = round(
                float((np.sign(true_delta[nz]) == np.sign(pred_delta[nz])).mean()), 4
            ) if nz.sum() > 0 else float('nan')

        results.append({
            'fold': fold,
            'train_end': train_end.date().isoformat(),
            'eval_start': eval_start.date().isoformat(),
            'eval_end': min(eval_end, date_max).date().isoformat(),
            'n_train': len(internal_train),
            'n_eval': len(eval_df),
            # sigma space (training loss space)
            'rmse_sigma': metrics['rmse'],
            'mae_sigma': metrics['mae'],
            'baseline_rmse_sigma': baseline_rmse,
            'beats_baseline_sigma': metrics['rmse'] < baseline_rmse,
            # raw delta space (interpretable)
            'rmse_raw': raw_rmse,
            'baseline_rmse_raw': raw_baseline_rmse,
            'beats_baseline_raw': raw_rmse < raw_baseline_rmse if not np.isnan(raw_rmse) else False,
            'sign_accuracy': raw_sign_acc,
        })

        beat_str = 'beats' if metrics['rmse'] < baseline_rmse else 'loses'
        sign_str = f'{raw_sign_acc:.3f}' if not np.isnan(raw_sign_acc) else '  n/a'
        print(
            f'  Fold {fold:2d} | train->{train_end.date()} | '
            f'eval {eval_start.date()}-{min(eval_end, date_max).date()} '
            f'| n_eval={len(eval_df):3d} '
            f'| RMSE_σ={metrics["rmse"]:.5f} (base={baseline_rmse:.5f}) {beat_str} '
            f'| RMSE_Δ={raw_rmse:.5f} | sign={sign_str}'
        )

        train_end += relativedelta(months=step_months)
        fold += 1

    return pd.DataFrame(results)
