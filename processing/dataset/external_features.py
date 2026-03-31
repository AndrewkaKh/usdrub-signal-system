"""Fetch external market features for model training.

Downloads:
  - Brent crude oil (BZ=F) close price
  - DXY US Dollar Index (DX-Y.NYB) close price
  - VIX volatility index (^VIX) close price
  - CBR key rate (cbr.ru)

Saves to data/exports/external_features.csv with columns:
  date, brent, dxy, vix, cbr_rate

Usage:
    python -m processing.dataset.external_features
    python -m processing.dataset.external_features --start 2021-01-01 --end 2026-04-09
"""
from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

OUTPUT_PATH = Path(__file__).parents[2] / 'data' / 'exports' / 'external_features.csv'
DEFAULT_START = '2021-01-01'


def fetch_yfinance(tickers: dict[str, str], start: str, end: str) -> pd.DataFrame:
    """Download daily close prices from Yahoo Finance via yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError('yfinance not installed. Run: pip install yfinance')

    frames = []
    for col_name, ticker in tickers.items():
        print(f'  Downloading {ticker} ({col_name}) ...')
        raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if raw.empty:
            raise ValueError(f'No data returned for {ticker}')
        # yfinance may return MultiIndex columns
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        series = raw['Close'].rename(col_name)
        series.index = pd.to_datetime(series.index).tz_localize(None)
        series.index.name = 'date'
        frames.append(series)

    return pd.concat(frames, axis=1)


def fetch_cbr_rate(start: str, end: str) -> pd.Series:
    """Fetch CBR key rate history from cbr.ru HTML table.

    Returns a Series indexed by date with the key rate in % per annum.
    The rate is a step function — forward-fill to fill gaps between announcements.
    """
    from_str = pd.to_datetime(start).strftime('%d.%m.%Y')
    to_str = pd.to_datetime(end).strftime('%d.%m.%Y')

    url = 'https://www.cbr.ru/hd_base/KeyRate/'
    params = {
        'UniDbQuery.Posted': 'True',
        'UniDbQuery.From': from_str,
        'UniDbQuery.To': to_str,
    }
    print('  Downloading CBR key rate from cbr.ru ...')
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    tables = pd.read_html(StringIO(resp.text), decimal=',', thousands=None)
    if not tables:
        raise ValueError('No tables found on cbr.ru KeyRate page')

    # The first table has two columns: date and rate
    df = tables[0]
    # Column names vary by locale — take by position
    df.columns = ['date', 'cbr_rate']
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df['cbr_rate'] = (
        df['cbr_rate']
        .astype(str)
        .str.replace(',', '.', regex=False)
        .str.replace('\xa0', '', regex=False)
        .str.strip()
        .astype(float)
    )
    df = df.dropna(subset=['date']).set_index('date').sort_index()
    return df['cbr_rate']


def build_daily_index(start: str, end: str) -> pd.DatetimeIndex:
    """Calendar days index — we will left-join MOEX trading days onto this later."""
    return pd.date_range(start=start, end=end, freq='D', name='date')


def main() -> None:
    parser = argparse.ArgumentParser(description='Fetch external features for ML model.')
    parser.add_argument('--start', default=DEFAULT_START, help='Start date YYYY-MM-DD')
    parser.add_argument('--end', default=pd.Timestamp.today().strftime('%Y-%m-%d'),
                        help='End date YYYY-MM-DD')
    args = parser.parse_args()

    print(f'Fetching external features: {args.start} to {args.end}')

    # Market data
    market = fetch_yfinance(
        tickers={'brent': 'BZ=F', 'dxy': 'DX-Y.NYB', 'vix': '^VIX'},
        start=args.start,
        end=args.end,
    )

    # CBR key rate
    cbr = fetch_cbr_rate(start=args.start, end=args.end)

    # Combine on full calendar index, forward-fill gaps (weekends, holidays)
    idx = build_daily_index(args.start, args.end)
    df = pd.DataFrame(index=idx)
    df = df.join(market, how='left')
    df = df.join(cbr, how='left')

    # Forward-fill: market data fills weekends; CBR rate fills between announcements
    df = df.ffill()

    df.index.name = 'date'
    df = df.reset_index()
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, sep=';')
    print(f'\nSaved {len(df)} rows to {OUTPUT_PATH}')
    print(df[df[['brent', 'dxy', 'vix', 'cbr_rate']].notna().all(axis=1)].tail(5).to_string(index=False))


if __name__ == '__main__':
    main()
