# !!! 6 Exports the tensors for the LSTM, one .npz per fold in outputs/tensors !!!

import os
import numpy as np

import config
import names
from data_collection import load_data
from dataset import build_panels
from walkforward import build_folds, pool


# Ticker and date of each window, same order as pool()
def build_meta(panels, fold, part):
    tks, dates = [], []
    for tk, parts in fold[names.STOCKS].items():
        idx = parts[part]
        tks.append(np.full(len(idx), tk.replace(names.B3_SUFFIX, ''), dtype='<U12'))
        dates.append(panels[tk][names.DATES].values[idx].astype('datetime64[D]'))
    return np.concatenate(tks), np.concatenate(dates)


def build_arrays(panels, feat_names, fold):
    arrays = {}
    for part in names.PARTS:
        # one X per group, y is the same for all of them
        for g in names.INDICATOR_GROUPS:
            X, y = pool(panels, fold, g, part)
            arrays[f'X_{g}_{part}'] = X
        arrays[f'y_{part}'] = y.astype(np.int8)

        tks, dates = build_meta(panels, fold, part)
        arrays[f'ticker_{part}'] = tks
        arrays[f'date_{part}'] = dates

    for g in names.INDICATOR_GROUPS:
        arrays[f'features_{g}'] = np.array(feat_names[g])
    arrays[names.TRAIN_UNTIL] = np.datetime64(fold[names.TRAIN_UNTIL].date(), 'D')
    arrays[names.TEST_UNTIL] = np.datetime64(fold[names.TEST_UNTIL].date(), 'D')
    return arrays


if __name__ == '__main__':
    folder = os.path.join(config.OUTPUT_DIR, names.TENSORS_DIR)
    os.makedirs(folder, exist_ok=True)

    panels, feat_names = build_panels(load_data())
    for fold in build_folds(panels):
        arrays = build_arrays(panels, feat_names, fold)
        path = os.path.join(folder, f'fold{fold[names.FOLD_IDX]}.npz')

        # neighbor windows share 29 of 30 rows, so it compresses well
        np.savez_compressed(path, **arrays)
        mb = os.path.getsize(path) / 2**20

        sizes = ''
        for part in names.PARTS:
            sizes += f"{part}={len(arrays[f'y_{part}'])} "
        print(f'fold {fold[names.FOLD_IDX]}: {sizes}-> {path} ({mb:.0f} MB)')
