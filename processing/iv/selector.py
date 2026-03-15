from __future__ import annotations

from datetime import datetime

import pandas as pd

from .utils import first_existing_column, normalize_option_type, parse_date, to_float

UNDERLYING_CONTRACT_COLUMN_CANDIDATES = ['UNDERLYINGASSET', 'UNDERLYING']
ASSET_CODE_COLUMN_CANDIDATES = ['ASSETCODE']
UNDERLYING_TYPE_COLUMN_CANDIDATES = ['UNDERLYINGTYPE']
EXPIRY_COLUMN_CANDIDATES = ['LASTTRADEDATE', 'MATDATE', 'EXPIRYDATE']
STRIKE_COLUMN_CANDIDATES = ['STRIKE']
OPTION_TYPE_COLUMN_CANDIDATES = ['OPTIONTYPE']
VOLUME_COLUMN_CANDIDATES = ['VOLTODAY', 'VOLUME']
OPEN_INTEREST_COLUMN_CANDIDATES = ['OPENPOSITION', 'OPENINTEREST', 'OPENPOS']


def prepare_options_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()

    underlying_contract_col = first_existing_column(prepared, UNDERLYING_CONTRACT_COLUMN_CANDIDATES)
    asset_code_col = first_existing_column(prepared, ASSET_CODE_COLUMN_CANDIDATES)
    underlying_type_col = first_existing_column(prepared, UNDERLYING_TYPE_COLUMN_CANDIDATES)
    expiry_col = first_existing_column(prepared, EXPIRY_COLUMN_CANDIDATES)
    strike_col = first_existing_column(prepared, STRIKE_COLUMN_CANDIDATES)
    option_type_col = first_existing_column(prepared, OPTION_TYPE_COLUMN_CANDIDATES)
    volume_col = first_existing_column(prepared, VOLUME_COLUMN_CANDIDATES)
    oi_col = first_existing_column(prepared, OPEN_INTEREST_COLUMN_CANDIDATES)

    if underlying_contract_col is None:
        raise ValueError('Underlying contract column not found in options table')
    if expiry_col is None:
        raise ValueError('Expiry column not found in options table')
    if strike_col is None:
        raise ValueError('Strike column not found in options table')
    if option_type_col is None:
        raise ValueError('Option type column not found in options table')
    if 'SECID' not in prepared.columns:
        raise ValueError('SECID column not found in options table')

    prepared['underlying_contract'] = prepared[underlying_contract_col].astype(str)
    if asset_code_col:
        prepared['asset_code_norm'] = prepared[asset_code_col].astype(str)
    else:
        prepared['asset_code_norm'] = prepared['underlying_contract'].astype(str)
    if underlying_type_col:
        prepared['underlying_type_norm'] = prepared[underlying_type_col].astype(str).str.upper()
    else:
        prepared['underlying_type_norm'] = 'F'
    prepared['expiry_dt'] = prepared[expiry_col].apply(parse_date)
    prepared['strike_num'] = prepared[strike_col].apply(to_float)
    prepared['option_type_norm'] = prepared[option_type_col].apply(normalize_option_type)
    prepared['volume_num'] = prepared[volume_col].apply(to_float) if volume_col else 0.0
    prepared['open_interest_num'] = prepared[oi_col].apply(to_float) if oi_col else 0.0

    prepared = prepared.dropna(subset=['underlying_contract', 'asset_code_norm', 'expiry_dt', 'strike_num', 'option_type_norm'])
    return prepared


def select_market_subset(
    prepared_df: pd.DataFrame,
    asset_code: str | None = 'Si',
    preferred_underlying: str | None = None,
) -> pd.DataFrame:
    subset = prepared_df.copy()
    subset = subset[subset['underlying_type_norm'].fillna('F') == 'F'].copy()

    if preferred_underlying:
        exact = subset[subset['underlying_contract'] == preferred_underlying].copy()
        if not exact.empty:
            return exact
        return subset.iloc[0:0].copy()

    if asset_code:
        exact_asset = subset[subset['asset_code_norm'] == asset_code].copy()
        if not exact_asset.empty:
            return exact_asset
        prefix_asset = subset[subset['underlying_contract'].astype(str).str.startswith(asset_code)].copy()
        if not prefix_asset.empty:
            return prefix_asset
        return subset.iloc[0:0].copy()

    return subset


def select_expiry_for_tenor(
    prepared_df: pd.DataFrame,
    as_of: datetime,
    target_days: int,
    tolerance_days: int,
) -> tuple[datetime | None, int | None]:
    expiries = prepared_df[['expiry_dt']].drop_duplicates().copy()
    if expiries.empty:
        return None, None

    expiries['days_to_expiry'] = expiries['expiry_dt'].apply(lambda x: (x.date() - as_of.date()).days)
    expiries = expiries[expiries['days_to_expiry'] > 0].copy()
    if expiries.empty:
        return None, None

    expiries['distance_to_target'] = (expiries['days_to_expiry'] - target_days).abs()
    expiries = expiries[expiries['distance_to_target'] <= tolerance_days].copy()
    if expiries.empty:
        return None, None

    expiries = expiries.sort_values(['distance_to_target', 'days_to_expiry'])
    best = expiries.iloc[0]
    return best['expiry_dt'], int(best['days_to_expiry'])


def select_underlying_for_expiry(prepared_df: pd.DataFrame, expiry: datetime) -> str | None:
    subset = prepared_df[prepared_df['expiry_dt'] == expiry].copy()
    if subset.empty:
        return None
    underlying_counts = subset['underlying_contract'].astype(str).value_counts()
    if underlying_counts.empty:
        return None
    return str(underlying_counts.idxmax())


def _select_candidates_by_type(
    prepared_df: pd.DataFrame,
    expiry: datetime,
    underlying_contract: str,
    option_type: str,
    futures_price: float,
    limit: int,
) -> list[pd.Series]:
    subset = prepared_df[
        (prepared_df['expiry_dt'] == expiry)
        & (prepared_df['underlying_contract'] == underlying_contract)
        & (prepared_df['option_type_norm'] == option_type)
    ].copy()
    if subset.empty:
        return []

    subset['distance'] = (subset['strike_num'] - futures_price).abs()
    subset['open_interest_num'] = subset['open_interest_num'].fillna(0.0)
    subset['volume_num'] = subset['volume_num'].fillna(0.0)
    subset = subset.sort_values(['distance', 'open_interest_num', 'volume_num'], ascending=[True, False, False])
    return [row for _, row in subset.head(limit).iterrows()]


def select_atm_candidates(
    prepared_df: pd.DataFrame,
    expiry: datetime,
    underlying_contract: str,
    futures_price: float,
    limit: int,
) -> dict[str, list[pd.Series]]:
    call_rows = _select_candidates_by_type(prepared_df, expiry, underlying_contract, 'c', futures_price, limit)
    put_rows = _select_candidates_by_type(prepared_df, expiry, underlying_contract, 'p', futures_price, limit)
    return {'call': call_rows, 'put': put_rows}