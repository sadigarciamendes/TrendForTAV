# !!! 5 Walk forward validation, splits the timeline into train, validation and test !!!
#
# The timeline shared by all stocks is split into N_FOLDS+1 blocks of dates
# In fold i the model trains on everything before boundary i and is tested on the next block
#
#    fold 1  |=train=|gap|val|gap|==test==|
#    fold 2  |=====train=====|gap|val|gap|==test==|
#    fold 3  |========train========|gap|val|gap|==test==|
#
# val is the end of the training block (VAL_RATIO), only for early stopping
# gap is HORIZON windows thrown away, the label looks 10 days ahead so without it
# the last training labels would see the start of val and test
#
# The boundaries are calendar dates, the same for every stock, so all stocks
# fall in the same years in the same fold

import numpy as np
import pandas as pd

import config
import names

MIN_TRAIN = 60  # minimum windows for a stock to enter the fold
MIN_VAL = 20
MIN_TEST = 10


# k+1 dates evenly spaced on the shared timeline
def calc_bounds(panels, k):
    all_dates = []
    for p in panels.values():
        all_dates.append(p[names.DATES].values)
    dates = np.unique(np.concatenate(all_dates))
    n = len(dates)

    bounds = []
    for i in range(1, k + 2):
        pos = min(n * i // (k + 1), n - 1)
        bounds.append(pd.Timestamp(dates[pos]))
    return bounds


# Positions of the train, val and test windows of one stock in one fold
def split_stock(dates, train_until, test_until):
    a = int(np.searchsorted(dates, train_until.to_datetime64(), side='left'))  # first test window
    b = int(np.searchsorted(dates, test_until.to_datetime64(), side='right'))  # one past the last
    cut = int(a * (1 - config.VAL_RATIO))
    gap = config.HORIZON
    return {
        names.TRAIN: np.arange(0, cut),
        names.VAL: np.arange(min(cut + gap, a), a),
        names.TEST: np.arange(min(a + gap, b), b),
    }


def has_enough(parts):
    return (len(parts[names.TRAIN]) >= MIN_TRAIN
            and len(parts[names.VAL]) >= MIN_VAL
            and len(parts[names.TEST]) >= MIN_TEST)


def build_folds(panels):
    bounds = calc_bounds(panels, config.N_FOLDS)
    folds = []
    for i in range(config.N_FOLDS):
        fold = {
            names.FOLD_IDX: i + 1,
            names.TRAIN_UNTIL: bounds[i],
            names.TEST_UNTIL: bounds[i + 1],
            names.STOCKS: {},
        }
        for tk, p in panels.items():
            parts = split_stock(p[names.DATES].values, bounds[i], bounds[i + 1])
            if has_enough(parts):
                fold[names.STOCKS][tk] = parts

        if fold[names.STOCKS]:
            folds.append(fold)
    return folds


# Total windows of one part over all stocks of the fold
def count(fold, part):
    total = 0
    for parts in fold[names.STOCKS].values():
        total += len(parts[part])
    return total


# First and last date tested in the fold
def test_period(panels, fold):
    starts, ends = [], []
    for tk, parts in fold[names.STOCKS].items():
        d = panels[tk][names.DATES][parts[names.TEST]]
        starts.append(d[0])
        ends.append(d[-1])
    return min(starts), max(ends)


# Windows of one group from every stock in one part of the fold
def pool(panels, fold, g, part):
    xs, ys = [], []
    for tk, parts in fold[names.STOCKS].items():
        idx = parts[part]
        xs.append(panels[tk][names.X][g][idx])
        ys.append(panels[tk][names.Y][idx])
    return np.concatenate(xs), np.concatenate(ys)


def summarize_folds(panels, folds):
    rows = []
    for fold in folds:
        start, end = test_period(panels, fold)
        rows.append({
            'fold': fold[names.FOLD_IDX],
            'n_stocks': len(fold[names.STOCKS]),
            'train_until': str(fold[names.TRAIN_UNTIL].date()),
            'test_start': str(start.date()),
            'test_end': str(end.date()),
            'n_train': count(fold, names.TRAIN),
            'n_val': count(fold, names.VAL),
            'n_test': count(fold, names.TEST),
        })
    return rows


if __name__ == '__main__':
    from data_collection import load_data
    from dataset import build_panels

    panels, _ = build_panels(load_data())
    folds = build_folds(panels)
    print(pd.DataFrame(summarize_folds(panels, folds)).to_string(index=False))
