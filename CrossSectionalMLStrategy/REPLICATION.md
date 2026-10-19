# THRESHOLD — 8h cross-sectional book: replication package

Reproduces every number in *Quant Report v0.2* and *Operating Procedure v1.1* from raw 1-minute bars and
funding history. Scripts are numbered in the order they must run. Nothing else is needed.

```
01_universe.py            monthly top-20 / top-40 by trailing 30-day median quote volume
02_build_features.py      15m bars, features, 1h/2h/4h labels, per-symbol files; lookahead truncation test
03_add_horizon_label.py   adds the 8h label (exit_ts_480, exit_px_480, y_480) to every feature file
04_walkforward.py         pooled LightGBM, rolling 12-month train / 1-month test, purged; writes predictions
05_backtest.py            book backtest (N per side, inverse-ATR, 2x gross, netting, 3 cost scenarios)
06_maker_fills.py         passive-execution simulation on the 1m path; blended cost; per-year table
07_current_book.py        this month's 20 names + one rebalance worked step by step on real numbers
08_param_walkforward.py   rolling (trailing-6-month) re-selection of N / model variant, lookahead-free

features_lib.py           feature and label formulas (imported by 02 and 04; not run directly)
common.py                 parquet loaders, UTC normalisation
config.py                 all parameters; two RUN SWITCHES (see below)
requirements.txt          pandas, pyarrow, numpy, scipy, lightgbm, matplotlib
```

Mapping to the development numbering used in the earlier README and in the Operating Procedure:
01 = 10_universe, features_lib = 04_features_labels, 02 = 11_build_features_xs, 03 = 20_add_horizon_label,
04 = 12_walkforward_xs, 05 = 13_xs_backtest, 06 = 19_xs_maker_fills, 07 = 21_current_book, 08 = 22_param_walkforward.
The single-asset scripts (01–09 of the old numbering), the 4h turnover study (14) and the daily book (15–18)
are not part of the 8h result and are not included.

## Data

`RAW_OHLCV_DIR/<SYMBOL>.parquet` — 1-minute bars with columns
`timestamp, open, high, low, close, volume, quote_volume, trades, taker_buy_volume` (UTC).
`RAW_FUNDING_DIR/<SYMBOL>.parquet` — `timestamp, funding_rate` (+ optional interval / calc_time columns).
Binance USDT-M perpetuals, 790 symbols, delisted names included, through 2026-09-15.

## Run order

```
pip install -r requirements.txt
python 01_universe.py            # ~minutes; writes data_processed/xs/universe.parquet, results/universe_report.txt
python 02_build_features.py      # ~1-2 h for ~230 symbols; data_processed/xs/feat|slim|fund/<SYMBOL>.parquet
python 03_add_horizon_label.py   # seconds
python 04_walkforward.py         # ~1-2 h; results/xs_predictions_h480_tr12.parquet, xs_folds_h480_tr12.csv, xs_importance_h480_tr12.csv
python 05_backtest.py            # results/xs_report_h480.md, xs_equity_h480.png, xs_periods_h480.parquet
python 06_maker_fills.py         # results/xs_maker_report_h480.md, xs_maker_grid_h480.csv, xs_maker_equity_h480.png
python 07_current_book.py 100000 # results/current_book.md
python 08_param_walkforward.py   # results/xs_pwf_report_h480.md (needs run B below to have more than one model variant)
```

Step 04 prints `symbol identity feature: ON` (run A) or `OFF` (run B) at the start. Check it.

## The two runs

| | RUN_HOLDOUT | XS_USE_SYMCAT | output suffix | reported as |
|---|---|---|---|---|
| Development | False | True | `_h480_tr12` (Mar-2026 onward absent) | in-sample figures |
| **Run A** (final) | True | True | `_h480_tr12` | Report v0.2 main tables |
| **Run B** (ablation) | True | False | `_h480_tr12_nosym` | Report v0.2 "symbol-identity ablation" |

Run A and run B need only steps 04, 05, 06 to be re-run; 01–03 are unchanged. Change the switch in
`config.py`, not in the scripts. Note `XS_USE_SYMCAT` is the only identity switch in this package (the
development config also had `D_USE_SYMCAT`, which belonged to the rejected daily book).

## Numbers to check against (run A, holdout included)

| quantity | where | value |
|---|---|---|
| panel | xs_report_h480.md header | 5,708 periods, 136 names, 2021-07-01 to 2026-09-15 |
| book IC mean / months > 0 / worst | xs_report_h480.md "Cross-sectional IC" | +0.0519 / 97% / −0.0457 |
| gross, long_gross, short_gross (N=5, k=0) | xs_report_h480.md main table | 4.935, 2.316, 2.620 |
| turnover per period | same | 1.879 |
| flat taker net / Sharpe / max DD | same | +0.464 / 0.21 / −0.744 |
| random-book z | same | 8.79 |
| maker sim W=15 through=1: fill / cost / net / Sharpe / DD | xs_maker_report_h480.md | 95.4% / 1.09 bps / +3.582 / 1.58 / −0.575 |
| maker sim W=15 through=2 | same | 93.1% / 2.04 / +2.563 / 1.13 / −0.643 |
| per-year maker sim (W15/1) | same | 2021 +0.63, 2022 +1.09, 2023 +0.37, 2024 +0.54, 2025 +1.03, 2026 −0.085 |
| 2026 edge / sim cost / fill | same | 1.39 bps / 1.46 bps / 91.2% |
| holdout month ICs (book) | xs_folds_h480_tr12.csv | Mar +0.001, Apr −0.021, May +0.027, Jun +0.019, Jul +0.018, Aug +0.099, Sep −0.046 |
| top feature | xs_report_h480.md | sym_cat (gain ~914k), then fr_std12, fr_mean6 |

Run B: book IC +0.0329, 86% months > 0; gross 3.14; turnover 2.35; maker sim W15/1 net +1.78, Sharpe 0.76; 2026 net −0.169.

LightGBM is seeded (seed=7, deterministic given the same lightgbm version and thread count); expect
agreement to 3 significant figures, not bit-for-bit, across machines.

## Lookahead safeguards built into the scripts (what a reviewer should verify)

1. Universe for month M uses daily volumes on days strictly before the 1st of M (`01`, `win = ... < M`).
2. 15m bars are labelled by open time; a bar is usable only at `known_at_ts = bar_open + 15m` (`features_lib`).
3. Funding enters a feature only if `settle_ts < known_at_ts` (`merge_asof`, `allow_exact_matches=False`, `02`).
4. Entry price = 1m open at `known_at_ts`; exit = 1m open at `known_at_ts + 480m` (`03`).
5. Training rows for month M: decision time in `[M − 12 months, M)` AND exit time `<= M` (purge) (`04`).
6. Holdout: with `RUN_HOLDOUT = False`, rows at/after 2026-03-01 are never predicted (`04`).
7. Truncation self-test: `02` recomputes features on truncated data for 3 symbols and fails loudly on any difference.
8. Book membership at time t uses the universe of t's month, formed at the start of that month (`04`, `in_book`).
9. Maker fills are judged on 1m bars strictly after the order time; unfilled orders fall back to taker (`06`).
10. `08` chooses month m's parameters from months `< m` only.
