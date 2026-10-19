# THRESHOLD — 8-hour cross-sectional long/short book on crypto perpetuals

A market-neutral, cross-sectional strategy on Binance USDT-M perpetual futures. Every 8 hours it ranks the 20 most liquid names by a pooled LightGBM forecast of the next 8-hour return (in ATR units, relative to the cross-sectional mean), goes long the top 5 and short the bottom 5 with inverse-ATR weights, holds for one period, and repeats. The model is refit monthly on a rolling 12-month window with purging; the universe is re-formed monthly from trailing liquidity only.

This repository is the **replication package**: eight numbered scripts that rebuild every reported number from raw 1-minute bars and funding history, with the lookahead safeguards built in and self-tested. It does not contain data or results; those are produced locally when you run it.

> **Design rule for this project: no lookahead, ever.** Every script is written so that nothing known after a decision time can influence that decision, and `02_build_features.py` refuses to write output if its truncation test finds otherwise. See [Lookahead safeguards](#lookahead-safeguards) for what to verify.

---

## Contents

```
01_universe.py            monthly top-20 / top-40 by trailing 30-day median daily quote volume
02_build_features.py      15m bars, 35 features, per-symbol feature files; lookahead truncation self-test
03_add_horizon_label.py   adds the 8h label (exit_ts_480, exit_px_480, y_480) to every feature file
04_walkforward.py         pooled LightGBM, rolling 12-month train / 1-month test, purged; writes predictions
05_backtest.py            book backtest: N per side, inverse-ATR weights, 2x gross, netting, 3 cost scenarios
06_maker_fills.py         passive-execution simulation on the 1-minute path; blended cost; per-year table
07_current_book.py        this month's 20 names and one rebalance worked step by step on real numbers
08_param_walkforward.py   rolling (trailing-6-month) re-selection of N / model variant, lookahead-free

features_lib.py           feature and label formulas (imported by 02 and 04; not run directly)
common.py                 parquet loaders with schema checks, UTC normalisation
config.py                 every parameter in one place, including the two RUN SWITCHES
requirements.txt          pandas, pyarrow, numpy, scipy, lightgbm, scikit-learn, matplotlib
REPLICATION.md            reference values to check your run against, and the run-A / run-B protocol
```

Scripts are numbered in the order they must run. Run them from the repository root (`07` and `08` import `05_backtest.py` by module name, and every script does `import config`).

## Strategy in one page

**Universe (monthly, `01`).** On the 1st of each month, rank every symbol by its *median* daily quote volume over the 30 days strictly before the 1st. A symbol qualifies if it has at least 180 days of history, at least 24 of the 30 days present, and a trailing median of at least 5M USDT. Ranks 1–20 are the tradeable book; ranks 21–40 are added to the training pool only and are never traded. Delisted names are included for the months they qualified (no survivorship filter).

**Bars and features (`02`, `features_lib.py`).** 1-minute bars are aggregated to 15-minute bars labelled by open time. A bar with open time *T* becomes usable at `known_at_ts = T + 15m`. Features fall into four groups, 35 in total:

| group | count | examples |
|---|---|---|
| 15m bar features | 18 | ATR%, returns over 1/2/4/8/16/96 bars in ATR units, short/long realised-vol ratio, VWAP distance, RSI, range and body in ATR, volume and trade-count z-scores, hour / weekday / minute-of-day |
| intra-bar (from the 15 one-minute rows inside the bar) | 9 | taker-buy ratio (full bar and last 5 min), fraction of up minutes, path length, net move, return skew, last-5-minute volume share, 1m return std, chop |
| funding | 5 | last settled rate, 6-settlement mean, 12-settlement std, minutes since settlement, last interval |
| cross-asset (BTC) | 3 | BTC 4-bar and 16-bar return in ATR, BTC vol ratio |

Plus, when `XS_USE_SYMCAT = True`, the symbol itself as a LightGBM categorical feature (`sym_cat`).

**Label (`03`).** `y_480 = (open_1m[known_at_ts + 480m] − open_1m[known_at_ts]) / ATR14[T]`: enter at the first 1-minute open after the decision, exit at the 1-minute open eight hours later.

**Model (`04`).** For each test month *M*, train one pooled LightGBM regressor (Huber loss, 400 trees, learning rate 0.03, 31 leaves, min 500 samples per leaf, L2 = 10, seed 7) on every pool row with decision time in `[M − 12 months, M)` **and** exit time `≤ M` (purge). The target is `y_480` minus the cross-sectional mean of `y_480` over the pool at the same timestamp, clipped to ±10 ATR. Predict every row in month *M*. The reported skill metric is the cross-sectional Spearman IC across the book names at each timestamp, averaged over the month.

**Book (`05`).** On the 8-hour UTC grid (00:00, 08:00, 16:00), among book names with a prediction (at least 10 are required to trade a period): demean `yhat` over the book, long the top N, short the bottom N (N = 5 reported; 3 and 7 tested), inverse-ATR weights within each leg, each leg equal to 1× equity (2× gross, dollar-neutral). Hold to the next grid point. A name staying in the same leg is topped up or trimmed, not re-traded; costs are charged on `|w_new − w_old|` only. Funding settled inside the hold is paid or received. Three flat cost scenarios: taker 4.0 bps/side (3 fee + 1 slippage), maker_mix 1.5, maker −1.0 (rebate). A 200-seed random-book baseline uses the same names, sizing and costs.

**Execution simulation (`06`).** Each weight change becomes a limit order at the 1-minute open of the grid time. It fills if, within the next W minutes (5/15/30), a 1-minute bar trades *through* the limit by 1 or 2 bps; otherwise it falls back to a taker fill at the open of minute W. Because filled orders are priced from real execution to real exit, adverse selection on resting orders is in the number. The script reports fill rate by notional, the blended all-in cost per side actually achieved, and net / Sharpe / drawdown per year.

**Rolling parameter choice (`08`).** Re-selects (N, model variant) every month from the trailing 6 months of realised out-of-sample P&L and applies it to the next month, as a check on whether adaptive selection beats the fixed reference setting.

## Reference results

Run A (`RUN_HOLDOUT = True`, `XS_USE_SYMCAT = True`), 2021-07-01 to 2026-09-15, 5,708 eight-hour periods, 136 distinct book names. Returns are fractions of allocated equity, additive (not compounded), at 2× gross.

| quantity | value |
|---|---|
| Book IC, mean / months > 0 / worst month | +0.052 / 97% / −0.046 |
| Gross (long leg / short leg), N = 5 | +4.94 (+2.32 / +2.62) |
| Turnover per period (of 2.0 gross) | 1.88 |
| Flat taker (4 bps): net / Sharpe / max DD | +0.46 / 0.21 / −0.74 |
| Random-book z-score | 8.8 |
| Maker sim, W = 15 min, 1 bp through: fill / cost / net / Sharpe / max DD | 95.4% / 1.09 bps / +3.58 / 1.58 / −0.58 |
| Maker sim, W = 15 min, 2 bp through | 93.1% / 2.04 bps / +2.56 / 1.13 / −0.64 |
| Maker sim per year (W15 / 1 bp) | 2021H2 +0.63, 2022 +1.09, 2023 +0.37, 2024 +0.54, 2025 +1.03, 2026 (to Sep 15) −0.085 |
| Holdout Mar–Sep 2026 (never seen during development) | book IC mean ≈ +0.014, 5 of 7 months > 0; edge 1.39 bps vs simulated cost 1.46 bps |
| Top features by gain | `sym_cat`, then `fr_std12`, `fr_mean6` |

Run B (identity ablation, `XS_USE_SYMCAT = False`): book IC +0.033, 86% of months > 0, gross +3.14, turnover 2.35, maker sim W15/1 bp net +1.78, Sharpe 0.76. The symbol-identity feature carries roughly 40% of the IC and more than half the P&L; it is trained on trailing data only and still helped in the holdout, but it is the slow per-name component most likely to decay, so it should be monitored with and without.

The honest reading of these numbers: the signal was strong from 2021 to 2025 and did not invert in the 2026 holdout, but it weakened to the point where edge per unit traded roughly equals the simulated passive cost. Whether it is tradeable depends on realised fill quality, which a backtest cannot settle.

`REPLICATION.md` lists these and several more values to the precision you should expect. LightGBM is seeded, so expect agreement to about three significant figures across machines, not bit-for-bit.

## Data

The raw data is **not** in this repository. You need two directories of per-symbol parquet files, with paths set in `config.py`:

`RAW_OHLCV_DIR/<SYMBOL>.parquet` — 1-minute bars. `common.load_ohlcv_1m` requires these columns and raises if any is missing:

```
timestamp, open, high, low, close, volume, quote_volume, trades,
taker_buy_volume, taker_buy_quote_volume, open_time, close_time
```

`RAW_FUNDING_DIR/<SYMBOL>.parquet` — funding history with at least `timestamp, funding_rate`.

Timestamps may be tz-naive (treated as UTC) or tz-aware; the loaders normalise everything to UTC nanoseconds and floor funding timestamps to the minute. The results above used Binance USDT-M perpetuals, 790 symbols including delisted ones, through 2026-09-15. The schema matches Binance's public futures kline and funding-rate exports, so any source that produces those fields will work.

## Setup and run

Python 3.10 or newer is recommended. The scripts were developed and run on Windows (note the `D:/...` defaults in `config.py` and the console-encoding guards), but nothing in them is platform-specific.

```bash
pip install -r requirements.txt
# edit RAW_OHLCV_DIR and RAW_FUNDING_DIR in config.py

python 01_universe.py             # minutes.    data_processed/xs/universe.parquet, build_symbols.txt; results/universe_report.txt
python 02_build_features.py       # 1–2 h for ~230 symbols.  data_processed/xs/{feat,slim,fund}/<SYMBOL>.parquet
python 03_add_horizon_label.py    # seconds.
python 04_walkforward.py          # 1–2 h.      results/xs_predictions_h480_tr12.parquet, xs_folds_h480_tr12.csv, xs_importance_h480_tr12.csv
python 05_backtest.py             #             results/xs_report_h480.md, xs_equity_h480.png, xs_periods_h480.parquet
python 06_maker_fills.py          #             results/xs_maker_report_h480.md, xs_maker_grid_h480.csv, xs_maker_equity_h480.png
python 07_current_book.py 100000  #             results/current_book.md  (argument = equity in USDT, default 100,000)
python 08_param_walkforward.py    #             results/xs_pwf_report_h480.md, xs_pwf_choices_h480.csv, xs_pwf_equity_h480.png
```

`02` drops the first 48 hours after each listing, skips any symbol with under 30 days of data after that trim, and always builds `BTCUSDT` because the cross-asset features need it. It is resumable: symbols that already have a feature file are skipped, so delete a file to rebuild it. `04` caches the training pool in `data_processed/xs/pool_h480.parquet`; delete it if you change the universe or features. `04` prints `symbol identity feature: ON` or `OFF` at the start — check it matches the run you intend.

### The two run switches

Both live in `config.py`; change them there, not in the scripts.

| run | `RUN_HOLDOUT` | `XS_USE_SYMCAT` | output suffix | purpose |
|---|---|---|---|---|
| Development | `False` | `True` | `_h480_tr12` | in-sample work; rows from 2026-03-01 onward are never predicted |
| **Run A** (reported) | `True` | `True` | `_h480_tr12` | the main tables above |
| Run B (ablation) | `True` | `False` | `_h480_tr12_nosym` | symbol-identity ablation |

Switching between runs only requires re-running `04`, `05` and `06`; steps `01`–`03` are unaffected. `08` needs both run A and run B prediction files present to have more than one model variant to choose between.

## Lookahead safeguards

These are the properties a reviewer should check, and where each is enforced.

1. **Universe.** Month *M*'s ranking uses daily volumes on days strictly before the 1st of *M* (`01`, `win = ... < M`). A name that was liquid then and delisted later is correctly included then.
2. **Bar timing.** 15-minute bars are labelled by open time and usable only at `known_at_ts = open + 15m`; all rolling features are trailing (`features_lib.compute_bar_features`).
3. **Funding.** A settlement enters a feature only if `settle_ts < known_at_ts`, enforced with `merge_asof(..., allow_exact_matches=False)` and an assertion (`02`, `join_funding`).
4. **Prices.** Entry is the 1-minute open at `known_at_ts`; exit is the 1-minute open at `known_at_ts + 480m` (`02`, `03`).
5. **Training window and purge.** Rows for test month *M* are trained on decision times in `[M − 12 months, M)` **and** exit times `≤ M`, so no training label overlaps the test period (`04`).
6. **Holdout.** With `RUN_HOLDOUT = False`, rows at or after 2026-03-01 are never predicted, so no development decision could have seen them (`04`).
7. **Truncation self-test.** For the first three symbols built, `02` recomputes every 15m feature on 25 random rows from data truncated at that row and aborts on any difference above 1e-8 (`features_lib.truncation_test`).
8. **Book membership.** Membership at time *t* comes from the universe of *t*'s month, which was formed at the start of that month (`04`, `in_book`).
9. **Fills.** Maker fills are judged on 1-minute bars strictly after the order time; unfilled orders fall back to a taker fill at the end of the window (`06`).
10. **Parameter selection.** `08` chooses month *m*'s parameters from months `< m` only.

If you change a feature, keep it trailing and re-run `02` on at least one symbol so the truncation test exercises it.

## Suggested `.gitignore`

Everything the pipeline writes is derived and large; keep it out of the repository.

```
data_processed/
results/
__pycache__/
*.pyc
```

## Known quirks

- Some docstrings and plot titles in `06_maker_fills.py` still say "4h book"; the horizon is driven entirely by `XS_HORIZON` in `config.py` (480 = this 8h book; 240 reproduces the earlier 4h book, with unsuffixed output names).
- `05_backtest.py` saves the equity chart as `xs_equity_h480.png` but the markdown report links to `xs_equity.png`; the file is there, the link is just unsuffixed.
- `common.py` requires `taker_buy_quote_volume`, `open_time` and `close_time` in the 1-minute files, but none of the three feeds a feature or an output (`02` sums `taker_buy_quote_volume` per bar and then drops it). They are schema checks against the source export, not inputs. Relax `OHLCV_REQUIRED` if your data lacks them.
- `features_lib.py` has a stand-alone `main()` from the single-asset phase of the project that reads `data_processed/<SYM>_15m_f.parquet`; it is not part of this pipeline and those files are not produced here.

## Units and conventions

All times are UTC. Returns are fractions of allocated equity, additive across periods, at 2× gross exposure; +1.0 over a year means 100% of allocated equity. Edge and cost are in basis points per unit of notional traded. Sharpe ratios are annualised from daily sums of per-period net (√365). ATR is Wilder's 14-period ATR on 15-minute bars.
