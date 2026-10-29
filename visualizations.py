# !!! 7 Figures of the data stage for the paper, saved as vector PDF in outputs/figures !!!
# python visualizations.py            uses the focus ticker of config
# python visualizations.py VALE3.SA   another stock in the per stock figures

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # only saves files
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap

import config
import names
from data_collection import load_data
from dataset import build_panels, future_return, calc_tau, make_labels
from features import calc_features, crossover
from walkforward import build_folds, count, test_period, pool

COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
CLASS_COLOR = {-1: '#e34948', 0: '#a3a29b', 1: '#2a78d6'}
CLASS_NAME = {-1: 'Down (-1)', 0: 'Neutral (0)', 1: 'Up (+1)'}
INK = '#0b0b0b'
INK_2 = '#52514e'
GRID = '#e4e3df'

# the other features are 0/1 flags or cross events
CONTINUOUS = [names.RSI_FORCE, names.MACD_FORCE, names.MACD_SIGNAL_FORCE, names.MACD_HIST,
              names.PRICE_SMA50_FORCE, names.PRICE_SMA200_FORCE, names.SMA_TREND_FORCE,
              names.EMA_SHORT_FORCE, names.BB_POSITION, names.BB_VOL_FORCE,
              names.OBV_TREND_FORCE, names.VOL_FORCE]

H = config.HORIZON
TAU_FORMULA = rf'$\tau = {config.K_NEUTRAL} \cdot \sigma_{{20}} \cdot \sqrt{{{H}}}$'


def set_style():
    plt.rcParams.update({
        'savefig.bbox': 'tight',
        'pdf.fonttype': 42,  # text stays editable in the PDF
        'font.size': 9, 'axes.titlesize': 10, 'axes.titleweight': 'bold',
        'axes.titlelocation': 'left', 'axes.labelcolor': INK_2,
        'axes.edgecolor': INK_2, 'axes.linewidth': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
        'xtick.color': INK_2, 'ytick.color': INK_2,
        'legend.frameon': False, 'lines.linewidth': 1.4, 'axes.axisbelow': True,
    })


def short_name(tk):
    return tk.replace(names.B3_SUFFIX, '')


def save(fig, file_name, folder):
    path = os.path.join(folder, file_name)
    fig.savefig(path)
    plt.close(fig)
    print(' ', path)


# Fig 1 price with SMA50, SMA200 and the golden and death crosses
def fig_price_ma(df, tk, folder):
    close = df[names.CLOSE].astype(float)
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    cross = crossover(sma50 - sma200)
    golden = cross[cross > 0].index
    death = cross[cross < 0].index

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(close.index, close, color=COLORS[0], lw=1.0, label='Close (raw)')
    ax.plot(sma50.index, sma50, color=COLORS[1], label='SMA50')
    ax.plot(sma200.index, sma200, color=COLORS[2], label='SMA200')
    ax.scatter(golden, sma50.loc[golden], marker='^', s=60, color=INK,
               zorder=5, label='Golden cross (+1)')
    ax.scatter(death, sma50.loc[death], marker='v', s=60, facecolor='white',
               edgecolor=INK, linewidth=1.2, zorder=5, label='Death cross (-1)')
    ax.set_title(f'{short_name(tk)} - price and long moving averages')
    ax.set_ylabel('R$')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.legend(ncol=5, loc='upper left', bbox_to_anchor=(0, -0.1))
    save(fig, 'fig01_price_moving_averages.pdf', folder)


# Fig 2 main force indexes of each group in the last n days
def fig_indicators(df, tk, folder, n=250):
    feats = calc_features(df)

    # (group, [(column, legend)]) of each subplot
    plots = [
        (names.RSI, [(names.RSI_FORCE, names.RSI_FORCE)]),
        (names.MACD, [(names.MACD_FORCE, names.MACD_FORCE),
                      (names.MACD_SIGNAL_FORCE, names.MACD_SIGNAL_FORCE)]),
        (names.MA, [(names.SMA_TREND_FORCE, f'{names.SMA_TREND_FORCE} (SMA50 x SMA200)'),
                    (names.EMA_SHORT_FORCE, f'{names.EMA_SHORT_FORCE} (EMA9 x EMA20)')]),
        (names.BBANDS, [(names.BB_POSITION, names.BB_POSITION),
                        (names.BB_VOL_FORCE, names.BB_VOL_FORCE)]),
        (names.OBV, [(names.OBV_TREND_FORCE, names.OBV_TREND_FORCE),
                     (names.VOL_FORCE, names.VOL_FORCE)]),
    ]

    fig, axs = plt.subplots(len(plots), 1, figsize=(11, 9), sharex=True)
    for ax, (g, cols) in zip(axs, plots):
        for i, (col, label) in enumerate(cols):
            s = feats[g][col].iloc[-n:]
            ax.plot(s.index, s, color=COLORS[i], label=label)
        ax.axhline(0, color=INK_2, lw=0.6)
        ax.set_title(g)
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    fig.suptitle(f'{short_name(tk)} - force indexes by group (last {n} trading days)',
                 x=0.01, ha='left', fontweight='bold')
    fig.tight_layout()
    save(fig, 'fig02_indicators.pdf', folder)


# Fig 3 future return of each day against the neutral band
def fig_labeling(df, tk, folder, n=250):
    close = df[names.CLOSE].astype(float)
    d = pd.DataFrame({'ret': future_return(close) * 100, 'tau': calc_tau(close) * 100,
                      'y': make_labels(close)})
    d = d.dropna().iloc[-n:]

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.fill_between(d.index, -d.tau, d.tau, color='#f0efec', label=r'Neutral band $\pm\tau$')
    ax.plot(d.index, d.tau, color=INK_2, lw=0.8)
    ax.plot(d.index, -d.tau, color=INK_2, lw=0.8)
    for c in [-1, 0, 1]:
        mask = d.y == c
        ax.scatter(d.index[mask], d.ret[mask], s=10, color=CLASS_COLOR[c],
                   label=CLASS_NAME[c], zorder=3)
    ax.axhline(0, color=INK_2, lw=0.6)
    ax.set_title(f'{short_name(tk)} - labeling: {H}-day return vs. {TAU_FORMULA}')
    ax.set_ylabel('Future return (%)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    ax.legend(ncol=4, loc='upper left', bbox_to_anchor=(0, -0.1))
    save(fig, 'fig03_labeling.pdf', folder)


# Fig 4 share of each class in total and in the test of each fold
def fig_classes(panels, folds, folder):
    rows = [('Total', np.concatenate([p[names.Y] for p in panels.values()]))]
    for fold in folds:
        rows.append((f'Fold {fold[names.FOLD_IDX]} (test)', pool(panels, fold, names.RSI, names.TEST)[1]))
    rows = rows[::-1]  # Total at the top

    fig, ax = plt.subplots(figsize=(8, 3.4))
    for i, (_, y) in enumerate(rows):
        left = 0.0
        for c in [-1, 0, 1]:
            share = float((y == c).mean()) * 100
            legend = CLASS_NAME[c] if i == 0 else None
            ax.barh(i, share, left=left, color=CLASS_COLOR[c], height=0.7,
                    edgecolor='white', linewidth=2, label=legend)
            ax.text(left + share / 2, i, f'{share:.1f}%', ha='center', va='center',
                    fontsize=8, color=INK if c == 0 else 'white')
            left += share
    ax.set_yticks(range(len(rows)), [label for label, _ in rows])
    ax.set_xlim(0, 100)
    ax.set_xlabel('% of windows')
    ax.grid(False)
    ax.set_title('Class distribution')
    ax.legend(ncol=3, loc='upper left', bbox_to_anchor=(0, -0.18))
    save(fig, 'fig04_classes.pdf', folder)


# Median start of the validation of a fold
def val_start(panels, fold):
    starts = []
    for tk, parts in fold[names.STOCKS].items():
        starts.append(panels[tk][names.DATES][parts[names.VAL][0]])
    return pd.Series(starts).median()


# Fig 5 timeline of train, validation and test of each fold
def fig_walkforward(panels, folds, folder):
    start = min(p[names.DATES][0] for p in panels.values())
    end = max(p[names.DATES][-1] for p in panels.values())

    fig, ax = plt.subplots(figsize=(9, 3.2))
    for i, fold in enumerate(folds):
        t0, t1 = test_period(panels, fold)
        v0 = val_start(panels, fold)
        row = len(folds) - 1 - i  # fold 1 at the top
        parts = [(start, v0, COLORS[0], 'Train'),
                 (v0, fold[names.TRAIN_UNTIL], COLORS[1], 'Validation (last 15%)'),
                 (t0, t1, COLORS[2], 'Test')]

        for a, b, color, label in parts:
            a = mdates.date2num(a)
            b = mdates.date2num(b)
            ax.barh(row, b - a, left=a, color=color, height=0.6, label=label if i == 0 else None)

        n_test = count(fold, names.TEST)
        n_stocks = len(fold[names.STOCKS])
        ax.text(mdates.date2num(t1) + 20, row, f'{n_test:,} win. | {n_stocks} stocks',
                va='center', fontsize=8, color=INK_2)

    labels = [f'Fold {fold[names.FOLD_IDX]}' for fold in folds]
    ax.set_yticks(range(len(folds)), labels[::-1])
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlim(right=mdates.date2num(end) + 520)  # room for the text
    ax.grid(axis='y', visible=False)
    ax.set_title(f'Walk-forward cross-validation (gap of {H} windows between the parts)')
    ax.legend(ncol=3, loc='upper left', bbox_to_anchor=(0, -0.12))
    save(fig, 'fig05_walkforward.pdf', folder)


# Fig 6 how many stocks have a window on each date
def fig_stocks_over_time(panels, folds, folder):
    all_dates = np.concatenate([p[names.DATES].values for p in panels.values()])
    n_stocks = pd.Series(all_dates).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(n_stocks.index, n_stocks.values, color=COLORS[0], drawstyle='steps-post')
    for fold in folds:
        ax.axvline(fold[names.TRAIN_UNTIL], color=INK_2, lw=0.8, ls='--')
        ax.text(fold[names.TRAIN_UNTIL], 2, f' F{fold[names.FOLD_IDX]}', fontsize=8, color=INK_2)
    ax.set_ylim(0, len(panels) + 5)
    ax.set_ylabel('Stocks')
    ax.set_title('Stocks with an available window over time (dashed = start of the test of each fold)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    save(fig, 'fig06_stocks_over_time.pdf', folder)


# Fig 7 histogram of tau over all windows
def fig_tau(data, panels, folder):
    taus = []
    for tk, p in panels.items():
        tau = calc_tau(data[tk][names.CLOSE].astype(float))
        taus.append(tau.loc[p[names.DATES]].values)
    tau = np.concatenate(taus) * 100
    mean = tau.mean()
    median = np.median(tau)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    bins = np.linspace(0, np.percentile(tau, 99.5), 80)  # cuts the top 0.5%
    ax.hist(tau, bins=bins, color=COLORS[0], edgecolor='white', linewidth=0.4)

    top = ax.get_ylim()[1] * 0.97
    for v, label, ls, ha in [(mean, 'mean', '-', 'left'), (median, 'median', '--', 'right')]:
        ax.axvline(v, color=INK, lw=1, ls=ls)
        ax.text(v, top, f' {label} {v:.2f}% ', fontsize=8, va='top', ha=ha)
    ax.set_xlabel(r'$\tau$ (%)')
    ax.set_ylabel('Windows')
    ax.set_title(f'Distribution of the threshold {TAU_FORMULA} ({len(tau):,} windows)')
    save(fig, 'fig07_tau_threshold.pdf', folder)


# Fig 8 heatmap of the last input window of the stock
def fig_input_window(p, tk, feat_names, folder):
    last = []
    feats = []
    for g in names.INDICATOR_GROUPS:
        last.append(p[names.X][g][-1])
        feats += feat_names[g]
    X = np.concatenate(last, axis=1)
    n, m = X.shape  # days, features

    cmap = LinearSegmentedColormap.from_list('div', ['#e34948', '#f0efec', '#2a78d6'])
    fig, ax = plt.subplots(figsize=(8, 7))

    # pcolormesh stays vector in the PDF, imshow would be an image
    im = ax.pcolormesh(np.arange(n + 1) - 0.5, np.arange(m + 1) - 0.5, X.T, cmap=cmap, vmin=-1, vmax=1)
    ax.set_ylim(m - 0.5, -0.5)  # first feature at the top
    ax.grid(False)
    ax.set_yticks(range(m), feats, fontsize=8)

    ticks = list(range(0, n, 5))
    tick_labels = []
    for i in ticks:
        tick_labels.append(f't-{n - 1 - i}' if i < n - 1 else 't')
    ax.set_xticks(ticks, tick_labels)

    # line between the groups
    pos = 0
    for g in names.INDICATOR_GROUPS[:-1]:
        pos += len(feat_names[g])
        ax.axhline(pos - 0.5, color=INK, lw=1.2)

    ax.set_xlabel('Position in the window (trading days)')
    end = p[names.DATES][-1].strftime('%Y-%m-%d')
    label = int(p[names.Y][-1])
    ax.set_title(f'{short_name(tk)} - input window ending on {end} (label = {label:+d})')
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.solids.set_rasterized(False)  # by default the color bar is saved as an image
    save(fig, 'fig08_input_window.pdf', folder)


# Last day of every window as a table (windows x 24 features)
def last_day_table(panels, feat_names, sample):
    rows = []
    for p in panels.values():
        rows.append(np.concatenate([p[names.X][g][:, -1, :] for g in names.INDICATOR_GROUPS], axis=1))
    X = np.concatenate(rows)

    # fixed seed so the sample is the same every run
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)

    cols = []
    for g in names.INDICATOR_GROUPS:
        cols += feat_names[g]
    return pd.DataFrame(X[idx], columns=cols)


# Fig 9 scale of the 24 features on the last day of the windows
def fig_features(panels, feat_names, folder, sample=30000):
    df = last_day_table(panels, feat_names, sample)

    color = {}
    for i, g in enumerate(names.INDICATOR_GROUPS):
        for feat in feat_names[g]:
            color[feat] = COLORS[i]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 5.2), gridspec_kw={'width_ratios': [1.3, 1]})

    # left box plot of the continuous features
    cont = CONTINUOUS[::-1]
    bp = a1.boxplot([df[c] for c in cont], vert=False, whis=(1, 99), showfliers=False,
                    patch_artist=True, widths=0.6, medianprops={'color': INK, 'lw': 1})
    for box, c in zip(bp['boxes'], cont):
        box.set_facecolor(color[c])
        box.set_edgecolor(color[c])
    a1.set_yticks(range(1, len(cont) + 1), cont, fontsize=8)
    a1.axvline(0, color=INK_2, lw=0.6)
    a1.set_xlim(-1, 1)
    a1.set_title('Continuous (box = P25-P75; whiskers = P1-P99)')
    for i, g in enumerate(names.INDICATOR_GROUPS):  # empty lines just for the legend
        a1.plot([], [], color=COLORS[i], lw=6, label=g)
    a1.legend(ncol=5, loc='upper left', bbox_to_anchor=(0, -0.05), fontsize=8)

    # right how often the flags and events are not 0
    disc = [feat for feat in df.columns if feat not in CONTINUOUS][::-1]
    pos = [(df[c] > 0).mean() * 100 for c in disc]
    neg = [(df[c] < 0).mean() * 100 for c in disc]
    a2.barh(range(len(disc)), pos, color=CLASS_COLOR[1], height=0.7, label='+1')
    a2.barh(range(len(disc)), neg, left=pos, color=CLASS_COLOR[-1], height=0.7,
            edgecolor='white', linewidth=1, label='-1')
    a2.set_yticks(range(len(disc)), disc, fontsize=8)
    a2.set_xlabel('% of windows (last day)')
    a2.set_title('Flags and events (% different from 0)')
    a2.legend(loc='lower right')

    fig.suptitle(f'Scale of the 24 features on the last day of the window (sample of {len(df):,} windows)',
                 x=0.01, ha='left', fontweight='bold')
    fig.tight_layout()
    save(fig, 'fig09_features.pdf', folder)


if __name__ == '__main__':
    tk = sys.argv[1] if len(sys.argv) > 1 else config.FOCUS_TICKER
    set_style()
    folder = os.path.join(config.OUTPUT_DIR, names.FIGURES_DIR)
    os.makedirs(folder, exist_ok=True)

    data = load_data()
    panels, feat_names = build_panels(data)
    folds = build_folds(panels)
    if tk not in panels:
        sys.exit(f'{tk} is not among the valid stocks')

    print(f'Generating figures ({tk}):')
    fig_price_ma(data[tk], tk, folder)
    fig_indicators(data[tk], tk, folder)
    fig_labeling(data[tk], tk, folder)
    fig_classes(panels, folds, folder)
    fig_walkforward(panels, folds, folder)
    fig_stocks_over_time(panels, folds, folder)
    fig_tau(data, panels, folder)
    fig_input_window(panels[tk], tk, feat_names, folder)
    fig_features(panels, feat_names, folder)
