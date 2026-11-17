# !!! Names used across the project !!!
# every shared string lives here, so a typo becomes an error and not a silent bug

# yfinance columns
OPEN = 'Open'
HIGH = 'High'
LOW = 'Low'
CLOSE = 'Close'
VOLUME = 'Volume'
PRICES = [OPEN, HIGH, LOW, CLOSE]
OHLCV = [OPEN, HIGH, LOW, CLOSE, VOLUME]

B3_SUFFIX = '.SA'

# indicator groups, one per expert of the committee
RSI = 'RSI'
MACD = 'MACD'
MA = 'MA'
BBANDS = 'BBANDS'
OBV = 'OBV'
INDICATOR_GROUPS = [RSI, MACD, MA, BBANDS, OBV]

# RSI
RSI_FORCE = 'rsi_force'
RSI_OVERSOLD = 'rsi_oversold'
RSI_OVERBOUGHT = 'rsi_overbought'
RSI_ABOVE50 = 'rsi_above50'
RSI_CROSS50 = 'rsi_cross50'

# MACD
MACD_FORCE = 'macd_force'
MACD_SIGNAL_FORCE = 'macd_signal_force'
MACD_HIST = 'macd_hist'
MACD_HIST_DIR = 'macd_hist_dir'
MACD_CROSS = 'macd_cross'

# moving averages
MA_TREND_FORCE = 'ma_trend_force'  # short SMA x long SMA
MA_CROSS = 'ma_cross'

# Bollinger Bands
BB_POSITION = 'bb_position'
BB_VOL_FORCE = 'bb_vol_force'
BB_ABOVE_UPPER = 'bb_above_upper'
BB_BELOW_LOWER = 'bb_below_lower'

# OBV and volume
OBV_TREND_FORCE = 'obv_trend_force'
OBV_CROSS = 'obv_cross'
VOL_FORCE = 'vol_force'
VOL_HIGH = 'vol_high'

# keys of each stock panel
DATES = 'dates'
Y = 'y'
X = 'X'

# walk-forward modes 
EXPANDING = 'expanding'
SLIDING = 'sliding'
WF_MODES = [EXPANDING, SLIDING]

# keys of each fold
FOLD_IDX = 'idx'
STOCKS = 'stocks'
TRAIN_UNTIL = 'train_until'
TEST_UNTIL = 'test_until'

TRAIN = 'train'
VAL = 'val'
TEST = 'test'
PARTS = [TRAIN, VAL, TEST]

# output files and folders inside OUTPUT_DIR
FIGURES_DIR = 'figures'
TENSORS_DIR = 'tensors'

# classes inside the network 

DOWN = 0
NEUTRAL = 1
UP = 2
N_CLASSES = 3

# models besides the 5 experts of the committee
ALL = 'ALL'  # single LSTM with the 20 features, baseline of the committee
COMMITTEE = 'COMMITTEE'
SOFT = 'SOFT' # committee with the mean of the probabilities
MAJORITY = 'MAJORITY'
PERSISTENCE = 'PERSISTENCE'

MODELS_DIR = 'models'
RESULTS_DIR = 'results'