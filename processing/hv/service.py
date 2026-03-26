from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any

from processing.iv.utils import parse_date
from processing.moex_client import MoexRequestError, fetch_futures_candles

from .calculator import calculate_hv_from_prices, prepare_price_series
from .config import ANNUALIZATION_DAYS, CALENDAR_LOOKBACK_MULTIPLIER, CANDLES_INTERVAL, HV_WINDOW_CONFIG, MIN_CALENDAR_LOOKBACK_DAYS


TENOR_ORDER = ['1m', '3m', '6m']


def _calendar_lookback_days(window: int) -> int:
    scaled = int(math.ceil(window * CALENDAR_LOOKBACK_MULTIPLIER))
    return max(MIN_CALENDAR_LOOKBACK_DAYS, scaled)


def _build_empty_metric(window: int, status: str, message: str) -> dict[str, Any]:
    return {
        'underlying': None,
        'window': window,
        'annualization_days': ANNUALIZATION_DAYS,
        'interval': CANDLES_INTERVAL,
        'from_date': None,
        'till_date': None,
        'price_field': None,
        'source_rows': 0,
        'price_observations': 0,
        'returns_observations': 0,
        'hv': None,
        'status': status,
        'message': message,
    }


def calculate_hv_snapshot(
    iv_snapshot: dict[str, Any],
    as_of: datetime | None = None,
) -> dict[str, Any]:
    snapshot_timestamp = parse_date(iv_snapshot.get('timestamp'))
    as_of = as_of or snapshot_timestamp or datetime.now()

    asset_code = iv_snapshot.get('asset_code')
    iv_metrics = iv_snapshot.get('metrics', {}) or {}
    metrics: dict[str, Any] = {}

    for tenor_label in TENOR_ORDER:
        window = HV_WINDOW_CONFIG[tenor_label]
        iv_metric = iv_metrics.get(tenor_label, {}) or {}
        underlying = iv_metric.get('underlying')

        if not underlying:
            metric = _build_empty_metric(window, 'underlying_missing', 'Underlying futures contract is missing in IV snapshot')
            metrics[tenor_label] = metric
            continue

        lookback_days = _calendar_lookback_days(window + 1)
        from_date = (as_of.date() - timedelta(days=lookback_days)).isoformat()
        till_date = as_of.date().isoformat()

        metric = _build_empty_metric(window, 'hv_failed', 'HV calculation did not complete')
        metric['underlying'] = underlying
        metric['from_date'] = from_date
        metric['till_date'] = till_date

        try:
            candles = fetch_futures_candles(
                underlying,
                from_date=from_date,
                till_date=till_date,
                interval=CANDLES_INTERVAL,
            )
        except MoexRequestError as exc:
            metric['status'] = 'source_unavailable'
            metric['message'] = str(exc)
            metrics[tenor_label] = metric
            continue

        metric['source_rows'] = int(len(candles))
        prices, price_field, preparation_error = prepare_price_series(candles)
        metric['price_field'] = price_field

        if not prices.empty:
            prices = prices[prices.index.date < as_of.date()]

        metric['price_observations'] = int(len(prices))

        if preparation_error is not None:
            metric['status'] = preparation_error
            metric['message'] = 'Could not prepare daily futures price series from MOEX candles'
            metrics[tenor_label] = metric
            continue

        hv_value, returns, hv_error = calculate_hv_from_prices(
            prices,
            window=window,
            annualization_days=ANNUALIZATION_DAYS,
        )
        metric['returns_observations'] = int(len(returns))

        if hv_error is not None:
            metric['status'] = hv_error
            if hv_error == 'insufficient_history':
                metric['message'] = f'Need at least {window} daily log returns for this tenor-specific futures contract'
            else:
                metric['message'] = 'Historical volatility is not finite'
            metrics[tenor_label] = metric
            continue

        metric['hv'] = hv_value
        metric['status'] = 'ok'
        metric['message'] = None
        metrics[tenor_label] = metric

    hv_values = [metric.get('hv') for metric in metrics.values() if metric.get('hv') is not None]
    status_set = {metric.get('status') for metric in metrics.values()}

    if not hv_values:
        overall_status = 'source_unavailable' if status_set == {'source_unavailable'} else 'hv_failed'
    elif all(metric.get('status') == 'ok' for metric in metrics.values()):
        overall_status = 'ok'
    else:
        overall_status = 'partial'

    return {
        'timestamp': as_of.isoformat(),
        'asset_code': asset_code,
        'status': overall_status,
        'message': None,
        'metrics': metrics,
    }
