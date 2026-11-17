# TrendForTAV - Trend Forecasting via Technical Analysis Voting

## B3 - data collection, processing and normalization

Data stage of a committee of LSTM networks that forecasts the trend of B3
stocks over a swing trade horizon (10 trading days). Each indicator group
feeds its own expert LSTM. The project goes from downloading the prices to
the tensors ready to feed the networks, for two walk-forward schemes
(expanding and sliding), so both can be trained and compared.

## Summary

- **Stocks:** current IBOV composition, read from the CSV downloaded from the
  [B3 website](https://www.b3.com.br/pt_br/market-data-e-indices/indices/indices-amplos/indice-ibovespa-ibovespa-composicao-da-carteira.htm)
  (`data/IBOVDia_27-10-26.csv`, 76 stocks).
- **Collection:** daily OHLCV from `yfinance.Ticker(...).history()`, 10 years,
  prices not adjusted for dividends (`AUTO_ADJUST = False`; yfinance still
  adjusts splits) and `repair=True`, so yfinance fixes known errors (100x
  prices, missing split adjustment). Stocks with fewer than 400 trading days
  are dropped.
- **Manual data fixes:** checked by hand and listed in `config.py` (see below).
- **Features:** 20 technical indicators in 5 groups, one per expert of the
  committee: RSI (5), MACD (5), moving averages (2), Bollinger Bands (4) and
  OBV/volume (4). All of them stay in [-1, 1] by construction: normalized
  differences (a - b)/(a + b), crossover events (+-1) and binary flags. There
  is no z-score or min-max scaling, which avoids leaking statistics from the
  future.
- **Moving averages:** only the 20 and 100 day SMAs, as the distance between
  them and their crossover. The EMA9 x EMA20 pair was removed because its
  force had a 0.97 correlation with the MACD force, so the two experts would
  see almost the same input.
- **Label:** fixed horizon method with a dynamic threshold (Lopez de Prado,
  2018). The return 10 trading days ahead is compared with
  tau = 0.5 * sigma20 * sqrt(10), where sigma20 is the rolling standard
  deviation of 20 daily returns. Classes: -1 down, 0 neutral, +1 up.
- **Network input:** windows of 30 trading days, with shape
  `(samples, 30, n_features)` per group.
- **Validation:** walk-forward with 5 calendar folds, the same for every
  stock, in two modes:
  - **expanding:** the train always starts at the beginning of the series and
    grows one block per fold;
  - **sliding:** the train is one block long and moves forward with the fold.

  The last 15% of the train of each fold is used for validation (early
  stopping), with a gap of 10 windows between train, validation and test so
  the label never sees the next part. The test is the same in both modes, and
  so is fold 1.

## Data quality

All 76 stocks were scanned for jumps above 35% in one day, flat prices with
zero volume, gaps in the dates and periods with almost zero volatility.

| Stock | Problem | Treatment |
|---|---|---|
| NATU3 | no data from 2020-01 to 2025-06 (traded as NTCO3, which yfinance does not have) | excluded |
| UGPA3 | prices before 2021-06-28 are 2.183x too low (+119% jump in one day; volume is right) | prices before the date multiplied by 2.183 |
| SUZB3 | SUZB3 only started trading in Nov 2017; before it, flat price with zero volume | history starts on 2017-11-10 |
| TEND3 | listed in May 2017 (spin-off from Gafisa); before it, flat price with zero volume | history starts on 2017-05-04 |
| CPFE3 | price pinned by the State Grid tender offer during 2017 (tau almost zero) | history starts on 2017-12-01 |

The fixes are applied when the data is read, so the cache always keeps the raw
yfinance data. The other big one-day moves are real market events (e.g. AXIA3
+49% on 2017-08-22, Eletrobras privatization announcement; PRIO3 -37% on
2020-03-09, oil crash).

## Results

Download of 2026-10-27 (2016-10-27 to 2026-10-27).

| Metric | Value |
|---|---|
| Valid stocks | 75 |
| Candles | 172,747 |
| Windows of 30 trading days | 162,397 |
| Down / neutral / up | 28.2% / 39.0% / 32.8% |
| Mean / median threshold tau | 3.50% / 3.10% |

The fold boundaries and the test are the same in both modes:

| Fold | Train until | Test | Test windows | Stocks |
|---|---|---|---|---|
| 1 | 2018-11-23 | 2018-12-07 to 2020-06-26 | 22,597 | 59 |
| 2 | 2020-06-26 | 2020-07-10 to 2022-01-26 | 24,129 | 63 |
| 3 | 2022-01-26 | 2022-02-09 to 2023-08-21 | 25,278 | 66 |
| 4 | 2023-08-21 | 2023-09-04 to 2025-03-19 | 28,725 | 75 |
| 5 | 2025-03-19 | 2025-04-02 to 2026-10-13 | 28,725 | 75 |

Only the train and validation change:

| Fold | Expanding train | Expanding val. | Sliding train | Sliding val. |
|---|---|---|---|---|
| 1 | 19,470 | 2,860 | 19,470 | 2,860 |
| 2 | 40,706 | 6,585 | 20,979 | 3,087 |
| 3 | 62,588 | 10,430 | 21,818 | 3,206 |
| 4 | 87,940 | 14,832 | 24,760 | 3,638 |
| 5 | 112,974 | 19,198 | 24,975 | 3,675 |

Walk-forward figures: `outputs/figures/visualizations_expanding/fig05_walkforward.pdf`
and `outputs/figures/visualizations_sliding/fig05_walkforward.pdf`.

## Files

| File | Stage |
|---|---|
| `config.py` | settings: IBOV file, exclusions, manual fixes, period, horizon, window, folds and LSTM |
| `names.py` | string constants shared by all files (columns, groups, features, keys) |
| `data/IBOVDia_*.csv` | IBOV composition downloaded from the B3 website |
| `data_collection.py` | reads the IBOV file, downloads from yfinance with a cache per stock and applies the fixes |
| `features.py` | indicators turned into force indexes in [-1, 1] |
| `dataset.py` | 3-class label, alignment and windows of 30 trading days |
| `walkforward.py` | calendar walk-forward split, expanding and sliding modes, gap of 10 windows |
| `export_tensors.py` | tensors ready for the LSTM, one `.npz` per fold in `outputs/tensors/{mode}/` |
| `visualizations.py` | vector PDF figures in `outputs/figures/visualizations_{mode}/` |
| `model.py`, `train.py` | expert LSTM and its training (network stage, models in `outputs/models/{mode}/`) |

## Usage

```
pip install -r requirements.txt
python export_tensors.py        # .npz tensors of both modes (or: python export_tensors.py sliding)
python visualizations.py        # figures of both modes (optional: python visualizations.py sliding VALE3.SA)
python walkforward.py sliding   # fold table of one mode on screen
```

The first run downloads each stock and saves it as a csv in the user cache
folder (`~/.cache/TrendForTAV/PETR4.SA.csv`, ...). The next runs reuse these
files; delete one file to download only that stock again, or the whole folder
to download everything. Since the period is relative to the download date, a
new download may give numbers slightly different from the ones above.

To update the stocks, download the new IBOV composition on the B3 website,
save it in `data/` and change `IBOV_FILE` in `config.py`.

Each `foldK.npz` contains `X_{GROUP}_{train,val,test}` (`float32`,
`(N, 30, n_features)`, one input per expert), `y_{train,val,test}` (`int8`,
-1/0/+1, the same for every group), `ticker_*`, `date_*`, `features_{GROUP}`,
`train_until` and `test_until`. To use it with `CrossEntropyLoss` or
`sparse_categorical_crossentropy`, add 1 to the label (classes 0, 1, 2).

## Limitations

- The stock universe is the current IBOV composition, used for the whole
  period (survivorship bias).
- Prices are not adjusted for dividends and interest on equity (JCP), so the
  drops on the ex-dates show up in the indicators and in the label.
