from __future__ import annotations

from datetime import datetime
from typing import Any

from processing.moex_client import fetch_futures_price, fetch_option_marketdata, fetch_options_table

from .calculator import calculate_option_iv, extract_option_price
from .config import TENOR_CONFIG
from .selector import prepare_options_dataframe, select_atm_pair, select_expiry_for_tenor, select_market_subset, select_underlying_for_expiry
from .utils import compute_time_to_expiry


def _empty_leg() -> dict[str, Any]:
    return {
        'secid': None,
        'strike': None,
        'price': None,
        'price_source': None,
        'iv': None,
        'status': 'not_used',
        'message': None,
    }


def _build_empty_metric(target_days: int, tolerance_days: int, status: str, message: str) -> dict[str, Any]:
    return {
        'underlying': None,
        'futures_price': None,
        'futures_price_source': None,
        'target_days': target_days,
        'tolerance_days': tolerance_days,
        'expiry': None,
        'days_to_expiry': None,
        't': None,
        'iv': None,
        'status': status,
        'message': message,
        'call': _empty_leg(),
        'put': _empty_leg(),
    }


def _evaluate_leg(row: Any, futures_price: float, t: float) -> dict[str, Any]:
    if row is None:
        leg = _empty_leg()
        leg['status'] = 'atm_option_not_found'
        leg['message'] = 'ATM option for this side was not found'
        return leg

    secid = row['SECID']
    strike = float(row['strike_num'])
    marketdata = fetch_option_marketdata(secid)
    price, price_source = extract_option_price(marketdata)

    leg = {
        'secid': secid,
        'strike': strike,
        'price': price,
        'price_source': price_source,
        'iv': None,
        'status': 'option_price_missing',
        'message': 'Could not extract valid market price',
    }

    if price is None:
        return leg

    iv, error = calculate_option_iv(price, futures_price, strike, t, row['option_type_norm'])
    if iv is None:
        leg['status'] = 'iv_failed'
        leg['message'] = error
        return leg

    leg['iv'] = iv
    leg['status'] = 'ok'
    leg['message'] = None
    return leg


def _combine_metric(
    target_days: int,
    tolerance_days: int,
    expiry: datetime,
    days_to_expiry: int,
    t: float,
    underlying: str,
    futures_price: float,
    futures_price_source: str | None,
    call_leg: dict[str, Any],
    put_leg: dict[str, Any],
) -> dict[str, Any]:
    valid_ivs = [value for value in [call_leg['iv'], put_leg['iv']] if value is not None]

    if len(valid_ivs) == 2:
        status = 'ok'
        message = None
        metric_iv = sum(valid_ivs) / 2.0
    elif len(valid_ivs) == 1:
        status = 'partial'
        message = 'Only one side produced valid IV'
        metric_iv = valid_ivs[0]
    else:
        status = 'iv_failed'
        message = 'Neither call nor put produced valid IV'
        metric_iv = None

    return {
        'underlying': underlying,
        'futures_price': futures_price,
        'futures_price_source': futures_price_source,
        'target_days': target_days,
        'tolerance_days': tolerance_days,
        'expiry': expiry.isoformat(),
        'days_to_expiry': days_to_expiry,
        't': t,
        'iv': metric_iv,
        'status': status,
        'message': message,
        'call': call_leg,
        'put': put_leg,
    }


def calculate_iv_snapshot(
    as_of: datetime | None = None,
    asset_code: str | None = 'Si',
    underlying: str | None = None,
) -> dict[str, Any]:
    as_of = as_of or datetime.now()

    options_raw = fetch_options_table()
    options_prepared = prepare_options_dataframe(options_raw)
    market_subset = select_market_subset(options_prepared, asset_code=asset_code, preferred_underlying=underlying)

    if market_subset.empty:
        return {
            'timestamp': as_of.isoformat(),
            'asset_code': asset_code,
            'requested_underlying': underlying,
            'status': 'market_subset_empty',
            'message': 'Could not find option series for requested asset code or underlying',
            'metrics': {
                tenor: _build_empty_metric(cfg['target_days'], cfg['tolerance_days'], 'market_subset_empty', 'No matching option series found')
                for tenor, cfg in TENOR_CONFIG.items()
            },
        }

    metrics = {}

    for tenor_label, cfg in TENOR_CONFIG.items():
        target_days = cfg['target_days']
        tolerance_days = cfg['tolerance_days']
        expiry, days_to_expiry = select_expiry_for_tenor(
            market_subset,
            as_of,
            target_days,
            tolerance_days,
        )

        if expiry is None or days_to_expiry is None:
            metrics[tenor_label] = _build_empty_metric(
                target_days,
                tolerance_days,
                'expiry_not_found',
                f'No expiry found within +/- {tolerance_days} days of target tenor',
            )
            continue

        selected_underlying = select_underlying_for_expiry(market_subset, expiry)
        if selected_underlying is None:
            metrics[tenor_label] = _build_empty_metric(
                target_days,
                tolerance_days,
                'underlying_not_found',
                'Could not determine futures contract for selected expiry',
            )
            continue

        futures_price, futures_price_source = fetch_futures_price(selected_underlying)
        if futures_price is None:
            metric = _build_empty_metric(
                target_days,
                tolerance_days,
                'futures_price_missing',
                'Could not fetch futures price for selected underlying',
            )
            metric['underlying'] = selected_underlying
            metrics[tenor_label] = metric
            continue

        t = compute_time_to_expiry(as_of, expiry)
        pair = select_atm_pair(market_subset, expiry, selected_underlying, futures_price)
        call_leg = _evaluate_leg(pair['call'], futures_price, t)
        put_leg = _evaluate_leg(pair['put'], futures_price, t)
        metrics[tenor_label] = _combine_metric(
            target_days,
            tolerance_days,
            expiry,
            days_to_expiry,
            t,
            selected_underlying,
            futures_price,
            futures_price_source,
            call_leg,
            put_leg,
        )

    overall_status = 'ok' if any(metric['iv'] is not None for metric in metrics.values()) else 'iv_failed'

    return {
        'timestamp': as_of.isoformat(),
        'asset_code': asset_code,
        'requested_underlying': underlying,
        'status': overall_status,
        'message': None,
        'metrics': metrics,
    }