# !!! 2 Data collection !!!

import os
import time
import pandas as pd
import yfinance as yf

import config
import names

MAX_TRIES = 3
WAIT = 5  # seconds between tries and between tickers


# Tickers of the IBOV csv from B3
def read_ibov(path):
    # title line before the header, file in latin1
    tab = pd.read_csv(path, sep=';', encoding='latin-1', skiprows=1, index_col=False)

    # last rows are totals without share type
    tks = []
    for code, kind in zip(tab.iloc[:, 0], tab.iloc[:, 2]):
        if pd.notna(kind):
            tks.append(code + names.B3_SUFFIX)
    return tks


def download_stock(tk):
    for i in range(1, MAX_TRIES + 1):
        try:
            df = yf.Ticker(tk).history(period=f'{config.YEARS}y', interval=config.INTERVAL,
                                       auto_adjust=config.AUTO_ADJUST, actions=config.ACTIONS,
                                       repair=config.REPAIR, rounding=config.ROUNDING)
            if not df.empty:
                df.index = df.index.tz_localize(None)  # only the dates matter
                return df
        except Exception as e:
            print(f'Error downloading {tk} (try {i}/{MAX_TRIES}): {e}')

        if i < MAX_TRIES:
            time.sleep(WAIT)
    return None


# Raw history from the cache, or downloaded and saved there
def load_stock(tk):
    path = os.path.join(config.CACHE_DIR, tk + '.csv')
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)

    print('Downloading', tk)
    df = download_stock(tk)
    time.sleep(WAIT)  # so yfinance does not block us
    if df is None:
        print(f'{tk}: NO DATA AFTER {MAX_TRIES} TRIES')
        return None
    df.to_csv(path)
    return df


# Keeps OHLCV and applies the manual fixes of config
def fix_stock(tk, df):
    df = df[names.OHLCV].dropna()

    if tk in config.START_DATES:
        df = df.loc[config.START_DATES[tk]:]

    if tk in config.PRICE_FIXES:
        date, factor = config.PRICE_FIXES[tk]
        df = df.copy()
        df.loc[df.index < date, names.PRICES] *= factor
    return df


# Dict with the OHLCV dataframe of every IBOV stock
def load_data():
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    tks = read_ibov(config.IBOV_FILE)
    print(f'{len(tks)} stocks in {config.IBOV_FILE}')

    data = {}
    for tk in tks:
        if tk in config.EXCLUDED_TICKERS:
            print(f'Excluded {tk}: {config.EXCLUDED_TICKERS[tk]}')
            continue

        df = load_stock(tk)
        if df is None:
            continue

        df = fix_stock(tk, df)
        if len(df) < config.MIN_HISTORY_DAYS:
            print(f'{tk}: SHORT HISTORY ({len(df)} candles), ignored')
            continue
        data[tk] = df

    print(len(data), 'stocks ready')
    return data


if __name__ == '__main__':
    data = load_data()
    print(data[config.FOCUS_TICKER].tail())
