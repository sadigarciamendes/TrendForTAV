# !!! System settings !!!

import os

# CSV downloaded 
# https://www.b3.com.br/pt_br/market-data-e-indices/indices/indices-amplos/indice-ibovespa-ibovespa-composicao-da-carteira.htm
IBOV_FILE = 'data/IBOVDia_27-10-26.csv'

FOCUS_TICKER = 'PETR4.SA'  # ticker used in examples and figures

# stocks dropped because the yfinance data can not be fixed
EXCLUDED_TICKERS = {
    'NATU3.SA': 'no data between 2020-01 and 2025-06',
}

# manual fixes
# START_DATES: the history only starts on this date
START_DATES = {
    'SUZB3.SA': '2017-11-10',  # SUZB3 only started trading in Nov 2017. Before it yfinance repeats a flat price with zero volume
    'TEND3.SA': '2017-05-04',  # Tenda was listed in May 2017, before it the price is flat with zero volume
    'CPFE3.SA': '2017-12-01',  # 2017 price pinned by the State Grid tender offer, almost zero volatility 
}
# prices before the date are multiplied by the factor
PRICE_FIXES = {
    'UGPA3.SA': ('2021-06-28', 2.183),  # yfinance prices before this date are 2.183x too low volume is ok
}

# --- Collection ---

YEARS = 10  # history to download
INTERVAL = '1d'  # daily
MIN_HISTORY_DAYS = 400  # stocks with fewer candles than this are ignored
AUTO_ADJUST = False  # keep raw prices 
ACTIONS = True  # also download dividends and splits
REPAIR = True  # let yfinance fix data errors
ROUNDING = True  # round prices to 2 decimals

# one csv per stock in the user cache folder (~/.cache/TrendForTAV);
# delete a file to download that stock again
CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'TrendForTAV')

# --- Labels and windows ---

HORIZON = 10  # forecast horizon in trading days
K_NEUTRAL = 0.5  # k of the neutral threshold tau 
WINDOW = 30  # trading days in each input window

# --- Walk-forward ---
# Expanding: the training block always starts on the first shared date and grows by one block per fold
# Sliding: the training block is always one block long and moves one block per fold
SLIDING_BLOCKS = 1 # the first block has the same training size as fold 1 
N_FOLDS = 5  # n folds
VAL_RATIO = 0.15  # final share of each training block used for validation

OUTPUT_DIR = 'outputs'  # saved results 


# --- LSTM ---

HIDDEN = 32  # hidden size of the LSTM, small because the data is noisy 
N_LAYERS = 1 
DROPOUT = 0.1 
BATCH_SIZE = 256
LR = 0.001  # learning rate
WEIGHT_DECAY = 0.0001  # L2 regularization 
EPOCHS = 100
PATIENCE = 30  # early stopping
CLIP_NORM = 1.0 # gradient clipping, RNR can explode the gradients and make the training diverge
SEEDS = [0, 1, 2]  # each network is trained once per seed
 

