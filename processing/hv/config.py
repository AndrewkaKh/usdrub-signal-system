HV_WINDOW_CONFIG = {
    '1m': 21,
    '3m': 63,
    '6m': 126,
}

ANNUALIZATION_DAYS = 252
CANDLES_INTERVAL = 24
MIN_CALENDAR_LOOKBACK_DAYS = 60
CALENDAR_LOOKBACK_MULTIPLIER = 2.0

DATE_COLUMN_CANDIDATES = ['end', 'begin', 'TRADEDATE', 'tradedate']
PRICE_COLUMN_CANDIDATES = [
    'close',
    'legalcloseprice',
    'settleprice',
    'waprice',
    'CLOSE',
    'LEGALCLOSEPRICE',
    'SETTLEPRICE',
    'WAPRICE',
]
