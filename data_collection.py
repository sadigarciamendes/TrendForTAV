# !!! 2-Data collection !!!

import os
import time
import pandas as pd
import yfinance as yf

import config
import names

MAX_RETRIES = 3  # download attempts per ticker
WAIT_TIME = 5  # seconds between attempts and between tickers


# read the IBOV tickers from the CSV downloaded from B3
def read_ibov(path):
    # the file has a title line before the header and is in latin-1 (B3 default)
    table = pd.read_csv(path, sep=';', encoding='latin-1', skiprows=1, index_col=False)
    codes = table.iloc[:, 0]  # first column ticker code
    types = table.iloc[:, 2]  # third column share type (ON, PN, UNT...)

    # the last rows are totals and have no share type, so they are skipped
    tickers = []
    for code, kind in zip(codes, types):
        if pd.notna(kind):
            tickers.append(code + names.B3_SUFFIX)
    return tickers


# Downloads the full history of one ticker, trying again if yfinance fails
def download_stock(tk):
    for i in range(1, MAX_RETRIES + 1):
        try:
            # yfinance fix known errors 
            df = yf.Ticker(tk).history(period=f'{config.YEARS}y',
                                       interval=config.INTERVAL,
                                       auto_adjust=config.AUTO_ADJUST,
                                       actions=config.ACTIONS,
                                       repair=config.REPAIR,
                                       rounding=config.ROUNDING)
            if not df.empty:
                df.index = df.index.tz_localize(None)  # yfinance returns the Sao Paulo time zone, we only need the dates
                return df
        except Exception as e:
            print(f'Error downloading {tk} (attempt {i}/{MAX_RETRIES}): {e}')
        if i < MAX_RETRIES:
            time.sleep(WAIT_TIME)
    return None


# Raw history of one ticker from its cache file, or downloaded and saved there
# each stock has its own file, 
# so deleting one file downloads only that stock again
def load_stock(tk):
    path = os.path.join(config.CACHE_DIR, tk + '.csv')
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)

    print('Downloading', tk)
    df = download_stock(tk)
    time.sleep(WAIT_TIME)  # pause between tickers 
    if df is None:
        print(f'{tk}: NO DATA FOUND AFTER {MAX_RETRIES} RETRIES')
        return None
    df.to_csv(path)
    return df


# Keeps the OHLCV and applies the manual fixes of config.py
def fix_stock(tk, df):
    df = df[names.OHLCV].dropna()

    # drops the bad period at the start of the history
    if tk in config.START_DATES:
        df = df.loc[config.START_DATES[tk]:]

    # puts the prices before the date back on the right scale
    if tk in config.PRICE_FIXES:
        date, factor = config.PRICE_FIXES[tk]
        df = df.copy()
        before = df.index < date
        df.loc[before, names.PRICES] *= factor
    return df


# Prices of every IBOV stock, already cleaned {ticker: OHLCV dataframe}
def load_data():
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    tickers = read_ibov(config.IBOV_FILE)
    print(f'{len(tickers)} stocks in {config.IBOV_FILE}')

    data = {}
    for tk in tickers:
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
