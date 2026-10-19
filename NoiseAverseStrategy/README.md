# Replication Package — Intraday Momentum (Noise-Area Breakout), Style C Maker

**Strategy:** Noise-averse intraday momentum on crypto perpetual futures, with
breakout-confirmation maker entry (Style C). Adapted from Zarattini, Aziz &
Barbon (2024).

**What this package reproduces:** the winning configuration of a multi-phase
pre-registered study — the Style C maker machine, its walk-forward, its sizing/
stop overlay, and the deployable equity curve — end to end from raw Binance data.

---

## 0. READ THIS FIRST — honest status of the result

This is a **research result, not a validated profitable system.** Before using
anything here for capital allocation, understand:

- The headline **walk-forward Sharpe (+1.17, t=2.87)** and the **deployable sized
  Sharpe (+0.65)** are computed on the **closed Binance dataset that trained the
  machine.** They are optimistic in-sample estimates, **not yet deflated** for
  multiple testing. Across this study, deflation has repeatedly cut raw t≈2.5
  toward 0.5–0.7.
- **2026 is negative in every version** (−1.16 to −1.56 Sharpe). The edge is a
  rare-high-volatility-day collector; the compressed 2026 regime starves it.
  This is confirmed across seven mechanisms and is the single most important
  caveat.
- **Style C is a *different strategy* from the plain taker system** — it adds a
  breakout-confirmation entry layer. Its numbers are not apples-to-apples with
  the §03A taker results.
- The **only uncontaminated test is a forward test.** Everything here is
  backtest on spent data. Recommendation remains: freeze and forward-test at
  dust/zero capital; do NOT allocate live risk yet.

Full write-up: `study_docs/QTRR_Noise-Averse_Intraday_Momentum_Trading_Strategy.docx`
(section 03B covers the maker execution finding; the append-only trial log is
`study_docs/trial_registry.jsonl`).

---

## 1. The winning configuration (frozen spec)

**Entry — Style C (confirm-then-passive-maker):**
- Session anchored 13:30 UTC; decisions every 30 min (14:00–23:30); forced flat 23:59 UTC.
- Noise area: σ(t) = mean over prior N sessions of |close(t)/open − 1| at the same
  minute-of-session; ≥10 valid priors. Gap-aware bands:
  Upper = max(O, prev_close)·(1+k·σ), Lower = min(O, prev_close)·(1−k·σ).
- On a breakout beyond the band at a mark: **arm** a confirmation trigger at
  signal ± `trigger_bps` (5 bps), then a **passive maker limit** a further
  `maker_offset_bps` (3 bps) beyond it. Fill (earning the +1 bps maker rebate)
  requires price to confirm through the trigger AND reach the passive limit
  before the next mark; else cancel. **All exits pay taker.**

**Exits:**
- Dynamic stop (`stop` ∈ {band, vwap}); opposite-band flip; forced 23:59 close.
- Continuous intrabar **catastrophic stop** at 400 bps from the *maker fill
  price*, conservative fills (worse of {stop, bar close} on gap-through).
- **No take-profit** (96% of P&L is in the top ~50 trades; a TP amputates the
  right tail that is the edge).

**Sizing — fixed-fractional risk:**
- notional = min(Equity · RISK_FRAC / stop_distance, MAX_LEV · Equity)
- RISK_FRAC = 0.0001 (0.01% risk/trade — a decimal fraction, NOT percent;
  0.01 would be 1%), stop_distance = 400 bps = 0.04, MAX_LEV = 20×.
- RISK_FRAC and MAX_LEV are **risk-budget choices set from limits, not tuned on
  P&L.**

**Filters:** volume confirmation (K, per-quarter selected), news blackout
(CPI/NFP/FOMC, 223 dates in `reference_data/macro_calendar_2020_2026.csv`).

**Walk-forward selection (this is what produces the headline):** parameters are
NOT globally fixed — they are re-selected each calendar quarter on the trailing
`train_days` window (plateau-smoothed Sharpe, flat if nothing qualifies). Grid:
`interval ∈ {30,60,120}`, `k ∈ {1.0,1.5,2.0}`, `stop ∈ {band,vwap}`,
`L ∈ {20,60,120}`, `K ∈ {0,2.0}`, style = C. `offset` is FROZEN (not swept) —
it is a potent free parameter and must be set from execution reality, not tuned.

**Most-recent-quarter (2026-07-01) selection**, for a single-combo deployment:
`interval=30, k=1.0, stop=band, L=20, K=0` (volume filter OFF that quarter).
Note this is only the *plurality* — the walk-forward genuinely spreads across
combos per asset (the modal full combo was chosen by ~5% of assets).

---

## 2. Directory layout

```
engine/          causal event-driven backtester + bias test suite
  engine.py            spec_v1 session/band/VWAP construction, causal loaders
  wf_engine.py         grid runner, causal sigma / leverage / prev-close helpers
  tests_bias.py        anti-bias suite (truncation, future-perturbation, shuffle)
walkforward/
  batch_run.py         asset discovery over a data directory
  walkforward_vconf.py volume-confirmation profile + walk-forward plumbing
  walkforward_maker.py MAKER walk-forward (styles A/B/C, fill accounting)  <-- winning run
sizing/
  sized_maker_C.py     Style C + catastrophic stop + fixed-risk sizing  <-- deployable curve
  catastrophic_stop.py continuous intrabar stop measurement
  fixed_risk_sizing.py fixed-fractional sizing measurement
  mfe_mae_analysis.py  trade excursion analysis (why stop=400bps, no TP)
training/
  train_recent_configs.py  train Style C on the most recent N days -> per-asset config JSONs
                           (the deploy-prep step; emits config_<SYMBOL>.json for the engine)
robustness/
  monte_carlo_robustness.py  block bootstrap, luck test, deflated Sharpe, concentration
data_tools/
  binance_convert.py   Binance Vision zip -> parquet, volume-continuity audit
  download_funding.py  funding-rate downloader + coverage audit
deploy/
  taquant_maker_C.py           TaQuant engine port of Style C (per-asset, manifest)
  taquant_intraday_momentum.py TaQuant port of the taker baseline (reference)
reference_data/
  macro_calendar_2020_2026.csv  news-blackout dates (223)
study_docs/
  QTRR_..._Strategy.docx        full research report (incl. 03B maker section)
  trial_registry.jsonl          append-only trial log
```

---

## 3. Environment

```
python >= 3.10
pip install numpy pandas pyarrow scipy
# multiprocessing is stdlib; all runners take --workers
```

No network needed to run the pipeline (data_tools/ fetch raw data if you don't
have it). All scripts have a CONFIG block at the top AND matching CLI flags;
CLI overrides CONFIG.

---

## 4. Reproduce, step by step

### Step 1 — get data (skip if you already have 1-minute parquet)
```
python data_tools/binance_convert.py   --help   # Binance Vision zips -> parquet
python data_tools/download_funding.py  --help   # 8h funding per symbol
```
Result: two directories, `<OHLCV_DIR>` and `<FUNDING_DIR>`, each holding
`<BASE>_USDT_USDT.parquet`. The converter runs a volume-continuity audit — a
units defect here silently breaks the study (it did once), so keep the audit on.

### Step 2 — (optional) verify the engine is causal
```
python engine/tests_bias.py --ohlcv-dir <OHLCV_DIR>
```
Anti-bias suite must pass BEFORE trusting any result.

### Step 3 — the winning walk-forward (Style C)
```
python walkforward/walkforward_maker.py \
    --ohlcv-dir <OHLCV_DIR> --funding-dir <FUNDING_DIR> \
    --out maker_out --news-calendar reference_data/macro_calendar_2020_2026.csv \
    --data-attestation "Binance-2020-2026" \
    --force-style C --trigger-bps 5 --maker-offset-bps 3 \
    --maker-bps 1 --taker-bps 2 --workers <N>
```
Produces `maker_oos_returns_C_off5.csv`, `maker_chosen_combos_C_off5.csv`,
`maker_fill_stats_C_off5.csv`. Expected pooled OOS Sharpe ≈ +1.17 (t≈2.87),
fill rate ≈ 89%. (Styles A and B for comparison: drop `--force-style`, or set
it to A / B; offsets 3/5/8 for robustness.)

### Step 4 — deployable sized curve
```
python sizing/sized_maker_C.py \
    --ohlcv-dir <OHLCV_DIR> --funding-dir <FUNDING_DIR> \
    --out sized_C_out --news-calendar reference_data/macro_calendar_2020_2026.csv \
    --data-attestation "Binance-2020-2026" \
    --risk-frac 0.0001 --cat-stop-bps 400 --max-lev 20 \
    --maker-bps 1 --taker-bps 2 --trigger-bps 5 --maker-offset-bps 3 --workers <N>
```
Produces `sized_C_trades.csv`, `sized_C_monthly_equity.csv`. Expected: Sharpe
+0.65, Sortino +1.68, Calmar +0.48, CAGR +7.0%, maxDD −14.5%, final 1.50×.
**The console prints `risk X%/trade | notional Y%` — verify it matches your
intent** (0.0001 → risk 0.010% / notional 0.250%; 0.01 → risk 1% / notional 25%).

### Step 5 — robustness (deflate the headline)
```
python robustness/monte_carlo_robustness.py \
    --returns sized_C_out/sized_C_trades.csv --out mc_C.json --n-trials 20
```
Block bootstrap, luck p-value, concentration (removing best ~10 days), and the
**deflated Sharpe** at the honest trial count. This is the number to trust over
the raw t-stat.

---

### Step 6 — train per-asset configs for deployment (the forward-test freeze)
```
python training/train_recent_configs.py \
    --ohlcv-dir <OHLCV_DIR> --funding-dir <FUNDING_DIR> \
    --out trained_configs --news-calendar reference_data/macro_calendar_2020_2026.csv \
    --train-days 180 --workers <N>
```
For each asset, trains Style C on the most recent `--train-days` and writes
`config_<SYMBOL>.json` (TaQuant import format) with that asset's selected combo.
Also writes `_INDEX.csv` (symbol, combo, plateau=conviction, train_sessions).
Add `--min-plateau 1.0` to emit only higher-conviction assets.

**HONEST STATUS:** these configs are TRAINED, not validated. The fit has no
out-of-sample check yet -- the genuine test is what happens AFTER deployment
(the forward test). Do not read the training-window plateau as forward edge.
Re-run each quarter to refresh the freeze.

---

## 5. Deploy (TaQuant engine)

`deploy/taquant_maker_C.py` is the per-asset engine port. One instance per
symbol, 1-minute bars, PERP/USD-M market. Manifest at the top holds every
parameter incl. the 223-date news calendar. Set UI Initial Capital = `capital_usd`
and commission to your maker-entry/taker-exit tier.

**Caveats for the port (read before trusting live numbers):**
1. It is a **single frozen combo**, not the per-quarter walk-forward machine —
   it will NOT reproduce the +0.65 curve. It is a fixed-parameter proxy; its
   defaults are set to the most-recent-quarter plurality combo.
2. **`risk_frac` is a decimal** (0.0001 = 0.01%, 0.01 = 1%). Match `capital_usd`
   to your UI Initial Capital or the % returns will be wrong.
3. Confirm the engine posts a real **limit** order for maker entries (the
   strategy tries `place_order(side,'limit',qty,price)` and falls back to
   market). If it falls back, you pay taker and lose the Style C rebate.
4. The volume-confirmation window is behaviorally correct but not guaranteed
   bit-identical to the research engine; reconcile one asset if you need exact
   trade-list parity.

---

## 6. Key numbers (all in-sample, Binance 2020–2026, closed data)

| stage | metric | value |
|---|---|---|
| Style C walk-forward | pooled OOS Sharpe | +1.17 (t 2.87) |
| Style C walk-forward | fill rate | 89% |
| Style A (rebate only) | Sharpe | +0.29 (fails) |
| Style B (confirm, taker) | Sharpe | +1.05 (offset-robust 3/5/8) |
| Deployable sized C | Sharpe / Sortino / Calmar | +0.65 / +1.68 / +0.48 |
| Deployable sized C | CAGR / maxDD | +7.0% / −14.5% |
| Deployable sized C | 2026 | −1.56 (still negative) |

**What the study concluded:** the maker *rebate* does not rescue the strategy
(Style A fails); what works is **breakout-confirmation entry timing** (a new
layer), which holds even at taker cost (Style B); the rebate then adds cleanly
on top (Style C). All of it remains regime-dependent and unconfirmed out of
sample. Freeze and forward-test; do not allocate.
```
```
