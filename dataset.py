# !!! 4- 3 class labels, alignment and input windows of each stock !!!

import numpy as np
import pandas as pd

import config
import names
from features import calc_features


# Return from today until HORIZON trading days ahead
def future_return(close):
    future_close = close.shift(-config.HORIZON)
    return future_close / close - 1

# Neutral threshold: tau = k * daily vol * sqrt(horizon)
# it follows the volatility of the stock, so 2% is a big move for a calm stock
# and just noise for an agitated one
def calc_tau(close):
    daily_vol = close.pct_change().rolling(20, min_periods=10).std()
    return config.K_NEUTRAL * daily_vol * np.sqrt(config.HORIZON)

# Label of each day +1 up, -1 down, 0 neutral
def make_labels(close):
    ret = future_return(close)
    tau = calc_tau(close)

    y = pd.Series(0.0, index=close.index)
    y[ret > tau] = 1.0
    y[ret < -tau] = -1.0

    # last days have no future yet and first days have no volatility, no label
    y[ret.isna() | tau.isna()] = np.nan
    return y

# Keeps only the dates where every indicator and the label exist
def align(df):
    groups = calc_features(df)
    y = make_labels(df[names.CLOSE].astype(float))

    dates = df.index
    for table in groups.values():
        dates = dates.intersection(table.dropna().index)
    dates = dates.intersection(y.dropna().index).sort_values()

    aligned = {}
    for g, table in groups.items():
        aligned[g] = table.loc[dates]
    return aligned, y.loc[dates], dates


# Cuts the (days x features) matrix into sliding windows of n days
# the window that ends on day t contains days t-n+1 .. t
def make_windows(feat, n):
    windows = []
    for t in range(n - 1, len(feat)):
        windows.append(feat[t - n + 1:t + 1])
    return np.asarray(windows, dtype=np.float32)  # shape (windows, n, features)


# Panel of each stock window i ends on dates[i] and has label y[i]
def build_panels(data):
    n = config.WINDOW
    panels = {}
    feature_names = {}
    for tk, df in data.items():
        groups, y, dates = align(df)
        if len(dates) < n:  # not enough days for a single window
            continue

        X = {}
        for g in names.INDICATOR_GROUPS:
            feature_names[g] = list(groups[g].columns)
            X[g] = make_windows(groups[g].values, n)

        # the first window ends on day n-1, so dates and labels start there
        panels[tk] = {
            names.DATES: dates[n - 1:],
            names.Y: y.values[n - 1:].astype(np.float32),
            names.X: X,
        }
    return panels, feature_names


if __name__ == '__main__':
    from data_collection import load_data

    panels, feature_names = build_panels(load_data())
    print(len(panels), 'stocks with panel built')
    for g in names.INDICATOR_GROUPS:
        print(f'  {g:7s} {len(feature_names[g])} features: {", ".join(feature_names[g])}')

    p = panels[config.FOCUS_TICKER]
    start = p[names.DATES][0].date()
    end = p[names.DATES][-1].date()
    print(f'\n{config.FOCUS_TICKER}: {len(p[names.Y])} windows from {start} to {end}')
    for g in names.INDICATOR_GROUPS:
        print(f'  {g:7s} {p[names.X][g].shape}')  # (windows, days per window, features)
