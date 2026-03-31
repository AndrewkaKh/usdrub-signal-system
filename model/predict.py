from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .data_prep import prepare_dataset
from .features import TARGET_COL
from .range_forecast import compute_spot_range

ARTIFACTS_DIR = Path(__file__).parent / 'artifacts'
MODEL_PATH = ARTIFACTS_DIR / 'catboost_iv_1m.cbm'
METADATA_PATH = ARTIFACTS_DIR / 'metadata.json'

# How many rows to keep when preparing features for a single prediction.
# Must be >= max lag (5) + max rolling window (21) + buffer.
_TAIL_ROWS = 60


def _load_artifacts() -> tuple[CatBoostRegressor, dict]:
    """Load model and metadata, raise if missing."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f'Model not found at {MODEL_PATH}. Run --retrain first.'
        )
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f'metadata.json not found at {METADATA_PATH}. Run --retrain first.'
        )
    model = CatBoostRegressor()
    model.load_model(str(MODEL_PATH))
    metadata = json.loads(METADATA_PATH.read_text())
    return model, metadata


def predict_next_day(
    dataset_csv_path: str,
    spot_price: float,
    confidence: float = 0.90,
) -> dict:
    """Predict IV for the next trading day and compute the expected spot range.

    Parameters
    ----------
    dataset_csv_path:
        Path to model_dataset_daily.csv.
    spot_price:
        Current USD/RUB spot price.
    confidence:
        Confidence level for the price range (0.68, 0.90, 0.95, or 0.99).

    Returns
    -------
    dict ready to be formatted into a Telegram report.
    """
    model, metadata = _load_artifacts()
    feature_cols: list[str] = metadata['feature_cols']

    # Prepare dataset and take the last meaningful window of rows for features
    df = prepare_dataset(dataset_csv_path)

    # The last row's target is NaN (no next-day observation) — that is exactly
    # what we want to predict. Use the last row with a valid feature vector.
    # After prepare_dataset the last row always has features but may have NaN target.
    last_row = df.iloc[[-1]]

    # Verify feature columns match
    missing = [c for c in feature_cols if c not in last_row.columns]
    if missing:
        raise ValueError(
            f'Features in metadata are missing from the dataset: {missing}. '
            'Run --retrain to rebuild the model with the current dataset.'
        )

    X = last_row[feature_cols]
    predicted_delta = float(model.predict(X)[0])
    current_iv = float(last_row['iv_1m'].iloc[0])
    predicted_iv = current_iv + predicted_delta
    last_date = str(last_row['date'].iloc[0].date())

    iv_change_pct = round(predicted_delta / current_iv * 100, 2)

    range_result = compute_spot_range(
        predicted_iv=predicted_iv,
        spot_price=spot_price,
        confidence=confidence,
    )

    return {
        'date': last_date,
        'current_iv_1m': round(current_iv, 6),
        'predicted_iv_1m': round(predicted_iv, 6),
        'iv_change_pct': iv_change_pct,
        'spot_price': spot_price,
        'range_lower': range_result['lower'],
        'range_upper': range_result['upper'],
        'move_pct': range_result['move_pct'],
        'confidence': confidence,
        'model_trained_at': metadata.get('trained_at', 'unknown'),
    }


def get_spot_price_from_db(db_path: str) -> float | None:
    """Get the latest futures settlement price as a proxy for spot USD/RUB.

    MOEX Si futures are priced in rubles per 1000 USD (e.g. 85000 means 85.0 RUB/USD).
    Returns the spot-equivalent price (futures_price / 1000).
    Returns None if the DB is unavailable or empty.
    """
    import sqlite3

    try:
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            '''
            SELECT settlement_price
            FROM futures_raw
            WHERE settlement_price > 0
              AND secid GLOB 'Si[FGHJKMNQUVXZ][0-9]*'
            ORDER BY date DESC, settlement_price ASC
            LIMIT 1
            '''
        ).fetchone()
        conn.close()
        if row and row[0]:
            return round(row[0] / 1000.0, 4)
    except Exception:
        pass
    return None
