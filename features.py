# !!! 3 Technical indicators as force indexes between minus one and one !!!

import numpy as np
import pandas as pd

import names

EPS = 1e-12


# Wilder RSI, from 0 to 100
def calc_rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder smoothing is an EMA with alpha = 1/n
    avg_gain = gain.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()

    # no losses gives NaN and the day is dropped later
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - 100 / (1 + rs)


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


# Normalized difference of two positive series, positive when a > b
def force(a, b):
    return (a - b) / (a + b)


# +1 when the series crosses zero going up, the opposite going down, 0 otherwise
def crossover(s):
    return np.sign(s).diff().fillna(0.0).clip(-1, 1)


def feats_rsi(close):
    rsi = calc_rsi(close)

    feats = pd.DataFrame(index=close.index)
    feats[names.RSI_FORCE] = (rsi - 50) / 50
    feats[names.RSI_OVERSOLD] = (rsi < 30).astype(float)
    feats[names.RSI_OVERBOUGHT] = (rsi > 70).astype(float)
    feats[names.RSI_ABOVE50] = (rsi > 50).astype(float)
    feats[names.RSI_CROSS50] = crossover(rsi - 50)
    return feats


def feats_macd(close):
    # MACD as a force of the EMAs 12 and 26, so it does not depend on the price level
    macd = force(ema(close, 12), ema(close, 26))
    signal = ema(macd, 9)
    hist = macd - signal

    feats = pd.DataFrame(index=close.index)
    feats[names.MACD_FORCE] = macd
    feats[names.MACD_SIGNAL_FORCE] = signal
    feats[names.MACD_HIST] = hist / 2  # difference of two forces goes up to 2
    feats[names.MACD_HIST_DIR] = np.sign(hist.diff()).fillna(0.0)
    feats[names.MACD_CROSS] = crossover(hist)
    return feats


def feats_ma(close):
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    ema9 = ema(close, 9)
    ema20 = ema(close, 20)

    feats = pd.DataFrame(index=close.index)
    feats[names.PRICE_SMA50_FORCE] = force(close, sma50)
    feats[names.PRICE_SMA200_FORCE] = force(close, sma200)
    feats[names.SMA_TREND_FORCE] = force(sma50, sma200)
    feats[names.EMA_SHORT_FORCE] = force(ema9, ema20)
    feats[names.GOLDEN_DEATH_CROSS] = crossover(sma50 - sma200)
    feats[names.EMA_CROSS] = crossover(ema9 - ema20)
    return feats


def feats_bbands(close, n=20, ndev=2.0):
    mid = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=0)
    upper = mid + ndev * std
    lower = mid - ndev * std

    # position inside the bands, lower band is the negative side and upper band is +1
    pos = (close - mid) / (upper - mid + EPS)

    # band width compared with its own 50 day average
    bw = (upper - lower) / (mid + EPS)
    bw_ref = bw.rolling(50, min_periods=10).mean()

    feats = pd.DataFrame(index=close.index)
    feats[names.BB_POSITION] = pos.clip(-1, 1)
    feats[names.BB_VOL_FORCE] = 2 * bw / (bw + bw_ref + EPS) - 1
    feats[names.BB_ABOVE_UPPER] = (close > upper).astype(float)
    feats[names.BB_BELOW_LOWER] = (close < lower).astype(float)
    return feats


def feats_obv(close, vol):
    obv = (np.sign(close.diff()).fillna(0.0) * vol).cumsum()
    obv_short = ema(obv, 10)
    obv_long = ema(obv, 30)
    vol_ma = vol.rolling(20, min_periods=5).mean()

    # OBV can be negative so force() does not work, divide by the absolute values
    diff = obv_short - obv_long
    size = obv_short.abs() + obv_long.abs() + EPS

    feats = pd.DataFrame(index=close.index)
    feats[names.OBV_TREND_FORCE] = (diff / size).clip(-1, 1)
    feats[names.OBV_CROSS] = crossover(diff)
    feats[names.VOL_FORCE] = force(vol, vol_ma)
    feats[names.VOL_HIGH] = (vol > 2 * vol_ma).astype(float)  # volume spike
    return feats


# Dict with one table per group, one column per feature
def calc_features(df):
    close = df[names.CLOSE].astype(float)
    vol = df[names.VOLUME].astype(float)
    return {
        names.RSI: feats_rsi(close),
        names.MACD: feats_macd(close),
        names.MA: feats_ma(close),
        names.BBANDS: feats_bbands(close),
        names.OBV: feats_obv(close, vol),
    }


if __name__ == '__main__':
    import config
    from data_collection import load_data

    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', None)

    tk = config.FOCUS_TICKER
    feats = calc_features(load_data()[tk])
    for g, tab in feats.items():
        print(f'\n{g} ({tk})')
        print(tab.dropna().tail().round(3))
