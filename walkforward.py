# !!! 5 Walk forward validation, splits the timeline into train, validation and test !!!
#
# The timeline shared by all stocks is split into N_FOLDS+1 blocks of dates.
# In fold i the test is block i+1 and the train comes before it. The two modes
# only change WHERE the train starts:
#
#   expanding: the train always starts at the beginning and grows one block per fold
#      fold 1  |=train=|gap|val|gap|==test==|
#      fold 2  |=====train=====|gap|val|gap|==test==|
#      fold 3  |========train========|gap|val|gap|==test==|
#
#   sliding: the train has SLIDING_BLOCKS blocks and moves one block per fold,
#   the oldest block is dropped
#      fold 1  |=train=|gap|val|gap|==test==|
#      fold 2          |=train=|gap|val|gap|==test==|
#      fold 3                  |=train=|gap|val|gap|==test==|
#
# val is the end of the train (VAL_RATIO), only for early stopping.
# gap is HORIZON windows thrown away: the label looks 10 days ahead, so without
# it the last training labels would see the start of val and test.
#
# The boundaries are calendar dates, the same for every stock, so all stocks
# fall in the same years in the same fold.
#
# python walkforward.py            expanding mode
# python walkforward.py sliding    sliding mode

import sys
import numpy as np
import pandas as pd

import config
import names

# minimum windows for a stock to enter the fold, if any part is short the stock is left out
MIN_TRAIN = 60
MIN_VAL = 20
MIN_TEST = 10


# k+1 dates evenly spaced on the shared timeline (the fold boundaries)
def calc_bounds(panels, k):
    # puts the dates of every stock in one list
    all_dates = []
    for p in panels.values():
        all_dates.append(p[names.DATES].values)
    # np.unique drops the repeated ones and sorts them, this is the shared timeline
    dates = np.unique(np.concatenate(all_dates))
    n = len(dates)

    # splits the n dates into k+1 blocks of the same size and keeps the date at the end of each
    # e.g. n=2500 and k=5 -> positions 416, 833, 1250, 1666, 2083 and 2499
    bounds = []
    for i in range(1, k + 2):
        pos = min(n * i // (k + 1), n - 1)  # min keeps the last boundary inside the list
        bounds.append(pd.Timestamp(dates[pos]))
    return bounds  # bounds[i] = end of the train of fold i+1, bounds[i+1] = end of its test


# Date where the train of fold i starts, the ONLY difference between the two modes
def train_start(bounds, i, mode):
    # expanding always starts at the beginning. Timestamp.min is the smallest possible date,
    # so the searchsorted in split_stock returns 0 (first window of the stock)
    # in sliding, the first folds do not have SLIDING_BLOCKS blocks before them yet,
    # so they also start at the beginning (that is why fold 1 is the same in both modes)
    if mode == names.EXPANDING or i < config.SLIDING_BLOCKS:
        return pd.Timestamp.min
    # sliding: starts SLIDING_BLOCKS blocks before the fold boundary
    # e.g. SLIDING_BLOCKS=1 in fold 3 (i=2) -> starts at bounds[1], uses only the previous block
    return bounds[i - config.SLIDING_BLOCKS]


# Positions of the train, val and test windows of ONE stock in ONE fold
# dates are the window dates of the stock, sorted; the result are indexes of these windows
def split_stock(dates, train_from, train_until, test_until):
    # searchsorted finds where the date would go in the sorted list
    # s: first window with date >= train_from (start of the train)
    s = int(np.searchsorted(dates, train_from.to_datetime64(), side='left'))
    # a: first window with date >= train_until (where the test period starts)
    a = int(np.searchsorted(dates, train_until.to_datetime64(), side='left'))
    # b: one past the last window with date <= test_until (side='right' includes the date itself)
    b = int(np.searchsorted(dates, test_until.to_datetime64(), side='right'))

    # the range [s, a) is the training block; its last VAL_RATIO becomes validation
    # e.g. s=0, a=1000, VAL_RATIO=0.15 -> cut=850, train 0..849 and val 850..999
    cut = s + int((a - s) * (1 - config.VAL_RATIO))
    gap = config.HORIZON
    return {
        # train: from s to cut-1
        names.TRAIN: np.arange(s, cut),
        # val: skips gap windows after the train; min keeps it from starting after a
        names.VAL: np.arange(min(cut + gap, a), a),
        # test: skips gap windows after val, goes up to b-1
        names.TEST: np.arange(min(a + gap, b), b),
    }


# True if the stock has enough windows in the three parts
def has_enough(parts):
    return (len(parts[names.TRAIN]) >= MIN_TRAIN
            and len(parts[names.VAL]) >= MIN_VAL
            and len(parts[names.TEST]) >= MIN_TEST)


# List with the N_FOLDS folds, each one with the parts of every stock
def build_folds(panels, mode=names.EXPANDING):
    bounds = calc_bounds(panels, config.N_FOLDS)  # N_FOLDS+1 boundaries
    folds = []
    for i in range(config.N_FOLDS):
        fold = {
            names.FOLD_IDX: i + 1,             # fold number, starting at 1
            names.TRAIN_UNTIL: bounds[i],      # end of the train, start of the test
            names.TEST_UNTIL: bounds[i + 1],   # end of the test
            names.STOCKS: {},                  # {stock: {train, val, test}}
        }
        train_from = train_start(bounds, i, mode)  # the same for every stock of the fold
        for tk, p in panels.items():
            parts = split_stock(p[names.DATES].values, train_from, bounds[i], bounds[i + 1])
            # a newly listed stock or a short history may not have enough windows
            if has_enough(parts):
                fold[names.STOCKS][tk] = parts

        # a fold with no stock is dropped
        if fold[names.STOCKS]:
            folds.append(fold)
    return folds


# Total windows of one part over all stocks of the fold
def count(fold, part):
    total = 0
    for parts in fold[names.STOCKS].values():
        total += len(parts[part])
    return total


# First and last date tested in the fold, over all stocks
def test_period(panels, fold):
    starts, ends = [], []
    for tk, parts in fold[names.STOCKS].items():
        d = panels[tk][names.DATES][parts[names.TEST]]  # test dates of this stock
        starts.append(d[0])
        ends.append(d[-1])
    return min(starts), max(ends)


# Windows of one indicator group from every stock in one part of the fold
# stacks stock by stock, so the training has to shuffle them later
def pool(panels, fold, g, part):
    xs, ys = [], []
    for tk, parts in fold[names.STOCKS].items():
        idx = parts[part]
        xs.append(panels[tk][names.X][g][idx])  # (windows, 30, n_features)
        ys.append(panels[tk][names.Y][idx])     # (windows,)
    return np.concatenate(xs), np.concatenate(ys)


# One row per fold with the dates and the size of each part, to check on screen
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

    # mode from the command line, expanding if nothing is given
    mode = sys.argv[1] if len(sys.argv) > 1 else names.EXPANDING
    if mode not in names.WF_MODES:
        sys.exit(f'mode must be one of {names.WF_MODES}')

    panels, _ = build_panels(load_data())
    folds = build_folds(panels, mode)
    print(f'\nWalk-forward {mode}')
    print(pd.DataFrame(summarize_folds(panels, folds)).to_string(index=False))
