# !!! 4 Labels, alignment and input windows of each stock !!!

import numpy as np
import pandas as pd

import config
import names
from features import calc_features


# Return from today until HORIZON trading days ahead
def future_return(close):
    return close.shift(-config.HORIZON) / close - 1


# tau = k * daily vol * sqrt(horizon)
# follows the volatility, 2% is a big move for a calm stock and noise for an agitated one
def calc_tau(close):
    vol = close.pct_change().rolling(20, min_periods=10).std()
    return config.K_NEUTRAL * vol * np.sqrt(config.HORIZON)


# +1 up, 0 neutral and the negative class for down
def make_labels(close):
    ret = future_return(close)
    tau = calc_tau(close)

    y = pd.Series(0.0, index=close.index)
    y[ret > tau] = 1.0
    y[ret < -tau] = -1.0
    y[ret.isna() | tau.isna()] = np.nan  # no future or no volatility yet
    return y


# Keeps only the dates where every indicator and the label exist
def align(df):
    groups = calc_features(df)
    y = make_labels(df[names.CLOSE].astype(float))

    dates = df.index
    for tab in groups.values():
        dates = dates.intersection(tab.dropna().index)
    dates = dates.intersection(y.dropna().index).sort_values()

    aligned = {}
    for g, tab in groups.items():
        aligned[g] = tab.loc[dates]
    return aligned, y.loc[dates], dates


# Sliding windows of n days, the window of day t ends on t
def make_windows(feat, n):
    wins = []
    for t in range(n - 1, len(feat)):
        wins.append(feat[t - n + 1:t + 1])
    return np.asarray(wins, dtype=np.float32)  # (windows, n, features)


# Panel of each stock, window i ends on dates[i] and has label y[i]
def build_panels(data):
    n = config.WINDOW
    panels = {}
    feat_names = {}
    for tk, df in data.items():
        groups, y, dates = align(df)
        if len(dates) < n:
            continue

        X = {}
        for g in names.INDICATOR_GROUPS:
            feat_names[g] = list(groups[g].columns)
            X[g] = make_windows(groups[g].values, n)

        # first window ends on the nth day
        panels[tk] = {
            names.DATES: dates[n - 1:],
            names.Y: y.values[n - 1:].astype(np.float32),
            names.X: X,
        }
    return panels, feat_names


if __name__ == '__main__':
    from data_collection import load_data

    panels, feat_names = build_panels(load_data())
    print(len(panels), 'stocks with panel built')
    for g in names.INDICATOR_GROUPS:
        print(f'  {g:7s} {len(feat_names[g])} features: {", ".join(feat_names[g])}')

    p = panels[config.FOCUS_TICKER]
    start = p[names.DATES][0].date()
    end = p[names.DATES][-1].date()
    print(f'\n{config.FOCUS_TICKER}: {len(p[names.Y])} windows from {start} to {end}')
    for g in names.INDICATOR_GROUPS:
        print(f'  {g:7s} {p[names.X][g].shape}')
