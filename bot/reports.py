"""Format morning and evening report payloads into Telegram Markdown text."""
from __future__ import annotations

import math
from datetime import datetime


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _iv(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ' —  '
    return f'{v * 100:.2f}%'


def _spread(iv, hv) -> str:
    if iv is None or hv is None:
        return ' —  '
    if isinstance(iv, float) and math.isnan(iv):
        return ' —  '
    if isinstance(hv, float) and math.isnan(hv):
        return ' —  '
    s = (iv - hv) * 100
    sign = '+' if s >= 0 else ''
    return f'{sign}{s:.2f}%'


def _metric(v, pct: bool = True) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return 'n/a'
    if pct:
        sign = '+' if v >= 0 else ''
        return f'{sign}{v * 100:.2f}%'
    return f'{v:.4f}'


def _price(v) -> str:
    if v is None:
        return '—'
    return f'{v:.2f}'


def _date_ru(iso_str: str) -> str:
    """Convert '2026-04-10' to '10 апр 2026'."""
    months = {
        1: 'янв', 2: 'фев', 3: 'мар', 4: 'апр', 5: 'май', 6: 'июн',
        7: 'июл', 8: 'авг', 9: 'сен', 10: 'окт', 11: 'ноя', 12: 'дек',
    }
    try:
        dt = datetime.strptime(iso_str, '%Y-%m-%d')
        return f'{dt.day} {months[dt.month]} {dt.year}'
    except Exception:
        return iso_str


def _prev_date_ru(iso_str: str) -> str:
    """Return the human-readable string for the day before iso_str."""
    try:
        from datetime import date, timedelta
        d = date.fromisoformat(iso_str)
        return _date_ru((d - timedelta(days=1)).isoformat())
    except Exception:
        return 'вчера'


# ---------------------------------------------------------------------------
# Volatility block helpers
# ---------------------------------------------------------------------------

def _vol_block_from_snapshot(iv_snap, hv_snap) -> str:
    """Build IV/HV/Spread table from real-time snapshots."""
    if iv_snap is None or iv_snap.get('status') == 'network_error':
        return '⚠️ _данные недоступны_'

    iv_m = iv_snap.get('metrics', {})
    hv_m = hv_snap.get('metrics', {}) if hv_snap else {}

    lines = ['```', f'{"":5s}  {"IV":>7}  {"HV":>7}  {"Спред":>7}']
    for tenor in ('1m', '3m'):
        iv_val = iv_m.get(tenor, {}).get('iv') if iv_m.get(tenor) else None
        hv_val = hv_m.get(tenor, {}).get('hv') if hv_m.get(tenor) else None
        iv_str = _iv(iv_val)
        hv_str = _iv(hv_val)
        sp_str = _spread(iv_val, hv_val)
        lines.append(f'{tenor.upper():5s}  {iv_str:>7}  {hv_str:>7}  {sp_str:>7}')
    lines.append('```')
    return '\n'.join(lines)


def _vol_block_from_end(iv_end: dict | None, non_trading_day: bool = False) -> str:
    """Build IV table from end-of-day smile ATM values."""
    if non_trading_day:
        return '_не торговый день_'
    if not iv_end:
        return '⚠️ _данные недоступны_'
    lines = ['```', f'{"":5s}  {"IV ATM":>7}']
    for tenor in ('1m', '3m'):
        v = iv_end.get(tenor)
        lines.append(f'{tenor.upper():5s}  {_iv(v):>7}')
    lines.append('```')
    return '\n'.join(lines)


def _smile_block(smile_metrics: dict | None, non_trading_day: bool = False) -> str:
    """Format smile metrics for 1m tenor."""
    if non_trading_day:
        return '_не торговый день_'
    if not smile_metrics:
        return '⚠️ _данные улыбки недоступны_'
    m1 = smile_metrics.get('1m')
    if not m1:
        return '⚠️ _данные улыбки недоступны (1M)_'
    parts = [f'ATM: {_iv(m1.get("atm_vol"))}']
    parts.append(f'RR25: {_metric(m1.get("rr25"))}')
    parts.append(f'BF25: {_metric(m1.get("bf25"))}')
    return '  '.join(parts)


def _ext_block(ext: dict | None, is_fresh: bool, prev_date: str) -> str:
    """Format external features block."""
    if not ext:
        return '⚠️ _данные недоступны_'
    brent = ext.get('brent')
    dxy = ext.get('dxy')
    vix = ext.get('vix')
    cbr = ext.get('cbr_rate')
    staleness = '' if is_fresh else f' _(данные от {_date_ru(str(ext.get("date", ""))[:10])})_'
    lines = [
        f'🛢 Brent: ${_price(brent)} | 💵 DXY: {_price(dxy)} | 😰 VIX: {_price(vix)}{staleness}',
        f'🏦 ЦБ: {cbr:.1f}%' if cbr is not None else '🏦 ЦБ: —',
    ]
    return '\n'.join(lines)


def _prediction_block(pred: dict | None) -> str:
    """Format model prediction block."""
    if not pred:
        return '⚠️ _прогноз недоступен_'
    curr = pred.get('current_iv_1m', 0)
    pred_iv = pred.get('predicted_iv_1m', 0)
    delta = pred.get('iv_change_pct', 0)
    lo = pred.get('range_lower')
    hi = pred.get('range_upper')
    sign = '+' if delta >= 0 else ''
    return (
        f'IV 1M → {_iv(pred_iv)} ({sign}{delta:.2f}%)\n'
        f'Диапазон 90% ДИ: {_price(lo)} – {_price(hi)}'
    )


# ---------------------------------------------------------------------------
# Public formatters
# ---------------------------------------------------------------------------

def format_morning(payload: dict) -> str:
    date_str = payload.get('report_date', '')
    ts = payload.get('timestamp', '')
    spot = payload.get('spot')
    iv_snap = payload.get('iv_snap')
    hv_snap = payload.get('hv_snap')
    ext = payload.get('ext_features')
    ext_fresh = payload.get('ext_fresh', False)
    pred = payload.get('prediction')

    spot_str = f'*{_price(spot)} ₽*' if spot else '—'

    lines = [
        f'📊 *Утренний отчёт USD/RUB* | {_date_ru(date_str)}',
        '',
        f'💹 *Рынок сейчас* ({ts})',
        f'Курс (Si): {spot_str}',
        '',
        '📈 *Волатильность:*',
        _vol_block_from_snapshot(iv_snap, hv_snap),
        '',
        '🤖 *Прогноз на сегодня:*',
        _prediction_block(pred),
        '',
        f'🌍 *Внешний фон* ({_prev_date_ru(date_str)}):',
        _ext_block(ext, ext_fresh, date_str),
    ]

    errors = payload.get('errors', [])
    if errors:
        lines += ['', f'_⚠️ {"; ".join(errors)}_']

    return '\n'.join(lines)


def format_evening(payload: dict) -> str:
    date_str = payload.get('report_date', '')
    ts = payload.get('timestamp', '')
    spot = payload.get('spot')
    iv_end = payload.get('iv_end')
    smile_metrics = payload.get('smile_metrics')
    ext = payload.get('ext_features')
    ext_fresh = payload.get('ext_fresh', False)
    pred = payload.get('prediction')
    non_trading_day = payload.get('non_trading_day', False)

    spot_str = f'*{_price(spot)} ₽*' if spot else '—'

    lines = [
        f'🌙 *Вечерний отчёт USD/RUB* | {_date_ru(date_str)}',
        '',
        f'📊 *Итоги дня* ({ts})',
        f'Курс закрытия (Si): {spot_str}',
        '',
        '📈 *Волатильность ATM:*',
        _vol_block_from_end(iv_end, non_trading_day),
        '',
        '🎯 *Улыбка (1M):*',
        _smile_block(smile_metrics, non_trading_day),
        '',
        '🤖 *Прогноз на завтра:*',
        _prediction_block(pred),
        '',
        f'🌍 *Внешний фон* ({_date_ru(date_str)}):',
        _ext_block(ext, ext_fresh, date_str),
    ]

    errors = payload.get('errors', [])
    if errors:
        lines += ['', f'_⚠️ {"; ".join(errors)}_']

    return '\n'.join(lines)
