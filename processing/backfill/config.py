from __future__ import annotations

from datetime import date

BASE_ASSET_CODE = 'Si'

BACKFILL_START_DATE = date(2016, 1, 1)
BACKFILL_END_DATE = None

SQLITE_DB_PATH = 'data/backfill/moex_backfill.sqlite3'

REQUEST_TIMEOUT_SECONDS = 30
ISS_PAGE_SIZE_HINT = 100

FUTURES_HISTORY_URL = 'https://iss.moex.com/iss/history/engines/futures/markets/forts/securities.json'
OPTIONS_HISTORY_URL = 'https://iss.moex.com/iss/history/engines/futures/markets/options/securities.json'