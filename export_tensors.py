# !!! 7-Exports the tensors ready for the LSTM, one .npz per fold in outputs/tensors/{mode}/ !!!
# python export_tensors.py                     both modes 
# python export_tensors.py expanding           only expanding -> outputs/tensors/expanding/
# python export_tensors.py sliding             only sliding   -> outputs/tensors/sliding/

import os
import sys

import numpy as np

import config
import names
from data_collection import load_data
from dataset import build_panels
from walkforward import build_folds, pool


# Ticker and date of each window, in the same order pool() stacks the windows
def build_meta(panels, fold, part):
    tickers, dates = [], []
    for tk, parts in fold[names.STOCKS].items():
        idx = parts[part]  # indexes of the windows of this stock in this part
        short_tk = tk.replace(names.B3_SUFFIX, '')  # 'PETR4.SA' -> 'PETR4
        tickers.append(np.full(len(idx), short_tk, dtype='<U12'))  # repeats the name once per window
        # datetime64[D] keeps only the day, no time
        dates.append(panels[tk][names.DATES].values[idx].astype('datetime64[D]'))
    return np.concatenate(tickers), np.concatenate(dates)



# All arrays of one fold, each one with the name it will have inside the .npz
def build_arrays(panels, feature_names, fold):
    arrays = {}
    for part in names.PARTS:  # train, val, test
       # one X per group (input of each expert), y is the same for all of them
        for g in names.INDICATOR_GROUPS:
            X, y = pool(panels, fold, g, part)
            arrays[f'X_{g}_{part}'] = X  # shape (windows, 30, n_features)
        arrays[f'y_{part}'] = y.astype(np.int8)  # -1 down, 0 neutral, +1 up

        # ticker and date of each window, to know later where each prediction came from
        tickers, dates = build_meta(panels, fold, part)
        arrays[f'ticker_{part}'] = tickers
        arrays[f'date_{part}'] = dates

    # names of the features of each group and the boundaries of the fold
    for g in names.INDICATOR_GROUPS:
        arrays[f'features_{g}'] = np.array(feature_names[g])
    arrays[names.TRAIN_UNTIL] = np.datetime64(fold[names.TRAIN_UNTIL].date(), 'D')
    arrays[names.TEST_UNTIL] = np.datetime64(fold[names.TEST_UNTIL].date(), 'D')
    return arrays


# Saves one .npz per fold of the mode in outputs/tensors/mode/
def export_mode(panels, feature_names, mode):
    # one folder per mode, mode does not overwrite the other
    folder = os.path.join(config.OUTPUT_DIR, names.TENSORS_DIR, mode)
    os.makedirs(folder, exist_ok=True)

    print(f'\nWalk-forward {mode}')
    for fold in build_folds(panels, mode):
        arrays = build_arrays(panels, feature_names, fold)
        path = os.path.join(folder, f'fold{fold[names.FOLD_IDX]}.npz')

        # neighbor windows share 29 of their 30 rows, so the compressed file is small
        np.savez_compressed(path, **arrays)
        mb = os.path.getsize(path) / 2**20  # size in MB

       # number of windows in each part, to check on screen
        sizes = ''
        for part in names.PARTS:
            sizes += f"{part}={len(arrays[f'y_{part}'])} "
        print(f'fold {fold[names.FOLD_IDX]}: {sizes}-> {path} ({mb:.0f} MB)')


if __name__ == '__main__':
    # modes from the command line, both if none is given
    modes = sys.argv[1:] or names.WF_MODES
    for mode in modes:
        if mode not in names.WF_MODES:
            sys.exit(f'mode must be one of {names.WF_MODES}')

    # the panels are the same in both modes, only the fold split changes
    panels, feature_names = build_panels(load_data())
    for mode in modes:
        export_mode(panels, feature_names, mode)




