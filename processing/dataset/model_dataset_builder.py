from __future__ import annotations

import pandas as pd


def _load_iv_daily(connection, start_date: str, end_date: str) -> pd.DataFrame:
    query = '''
        SELECT
            date,
            target_tenor,
            underlying_secid,
            iv,
            status,
            days_to_expiry
        FROM iv_daily
        WHERE date BETWEEN ? AND ?
    '''
    return pd.read_sql_query(query, connection, params=[start_date, end_date])


def _load_hv_daily(connection, start_date: str, end_date: str) -> pd.DataFrame:
    query = '''
        SELECT
            date,
            target_tenor,
            underlying_secid,
            hv,
            status
        FROM hv_daily
        WHERE date BETWEEN ? AND ?
    '''
    return pd.read_sql_query(query, connection, params=[start_date, end_date])


def build_model_dataset_daily(connection, start_date: str, end_date: str) -> pd.DataFrame:
    iv_daily = _load_iv_daily(connection, start_date, end_date)
    hv_daily = _load_hv_daily(connection, start_date, end_date)

    if iv_daily.empty:
        return pd.DataFrame()

    iv_daily['date'] = pd.to_datetime(iv_daily['date'])
    hv_daily['date'] = pd.to_datetime(hv_daily['date']) if not hv_daily.empty else pd.Series(dtype='datetime64[ns]')

    iv_pivot = iv_daily.pivot(index='date', columns='target_tenor', values='iv')
    iv_pivot.columns = [f'iv_{column}' for column in iv_pivot.columns]

    iv_status = iv_daily.pivot(index='date', columns='target_tenor', values='status')
    iv_status.columns = [f'iv_status_{column}' for column in iv_status.columns]

    iv_underlying = iv_daily.pivot(index='date', columns='target_tenor', values='underlying_secid')
    iv_underlying.columns = [f'underlying_{column}' for column in iv_underlying.columns]

    iv_dte = iv_daily.pivot(index='date', columns='target_tenor', values='days_to_expiry')
    iv_dte.columns = [f'days_to_expiry_{column}' for column in iv_dte.columns]

    frames = [iv_pivot, iv_status, iv_underlying, iv_dte]

    if not hv_daily.empty:
        hv_pivot = hv_daily.pivot(index='date', columns='target_tenor', values='hv')
        hv_pivot.columns = [f'hv_{column}' for column in hv_pivot.columns]

        hv_status = hv_daily.pivot(index='date', columns='target_tenor', values='status')
        hv_status.columns = [f'hv_status_{column}' for column in hv_status.columns]

        frames.extend([hv_pivot, hv_status])

    dataset = pd.concat(frames, axis=1).sort_index().reset_index()
    dataset = dataset.rename(columns={'date': 'date'})

    if 'iv_1m' in dataset.columns and 'hv_1m' in dataset.columns:
        dataset['spread_1m'] = dataset['iv_1m'] - dataset['hv_1m']
    else:
        dataset['spread_1m'] = None

    if 'iv_3m' in dataset.columns and 'hv_3m' in dataset.columns:
        dataset['spread_3m'] = dataset['iv_3m'] - dataset['hv_3m']
    else:
        dataset['spread_3m'] = None

    if 'iv_3m' in dataset.columns and 'iv_1m' in dataset.columns:
        dataset['ts_3m_1m'] = dataset['iv_3m'] - dataset['iv_1m']
    else:
        dataset['ts_3m_1m'] = None

    if 'iv_1m' in dataset.columns:
        dataset['target_iv_1m_next_day'] = dataset['iv_1m'].shift(-1)
        dataset['target_delta_iv_1m_next_day'] = dataset['target_iv_1m_next_day'] - dataset['iv_1m']
    else:
        dataset['target_iv_1m_next_day'] = None
        dataset['target_delta_iv_1m_next_day'] = None

    dataset['date'] = pd.to_datetime(dataset['date']).dt.strftime('%Y-%m-%d')
    return dataset