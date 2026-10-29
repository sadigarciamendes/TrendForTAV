# !!! System settings !!!

import os

# Stocks of the current IBOV composition
# CSV downloaded from the B3 website, to update the stocks download a new file and change the name here
# https://www.b3.com.br/pt_br/market-data-e-indices/indices/indices-amplos/indice-ibovespa-ibovespa-composicao-da-carteira.htm
IBOV_FILE = 'data/IBOVDia_27-10-26.csv'

FOCUS_TICKER = 'PETR4.SA'  # ticker of the detailed examples and figures

# stocks dropped because the yfinance data can not be fixed
EXCLUDED_TICKERS = {
    'NATU3.SA': 'no data between 2020-01 and 2025-06 (traded as NTCO3, which yfinance does not have)',
}

# manual fixes checked by hand, applied when reading so the cache keeps the raw data
# START_DATES the history only starts on this date
START_DATES = {
    'SUZB3.SA': '2017-11-10',  # started trading in Nov 2017, before it yfinance repeats a flat price with zero volume
    'TEND3.SA': '2017-05-04',  # Tenda was listed in May 2017 (spin off from Gafisa), before it the price is flat
    'CPFE3.SA': '2017-12-01',  # 2017 price pinned by the State Grid tender offer, almost zero volatility
}
# PRICE_FIXES (date, factor) prices before the date are multiplied by the factor
PRICE_FIXES = {
    'UGPA3.SA': ('2021-06-28', 2.183),  # yfinance prices before this date are 2.183x too low, volume is right
}

# Collection
YEARS = 10
INTERVAL = '1d'
MIN_HISTORY_DAYS = 400  # stocks with fewer candles are ignored
AUTO_ADJUST = False  # raw prices, not adjusted for dividends
ACTIONS = True  # also downloads dividends and splits
REPAIR = True  # yfinance fixes known data errors
ROUNDING = True

# one csv per stock in ~/.cache/TrendForTAV, delete a file to download that stock again
CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'TrendForTAV')

# Labels and windows
HORIZON = 10  # forecast horizon in trading days
K_NEUTRAL = 0.5  # tau = k * daily vol * sqrt(HORIZON)
WINDOW = 30  # trading days in each input window

# Walk forward
WALK_FORWARD = True  # not used yet
N_FOLDS = 5
VAL_RATIO = 0.15  # final share of each training block used for validation

OUTPUT_DIR = 'outputs'
