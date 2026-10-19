# Replication Package — Directional Crypto Regime Strategy (Option A)

Self-contained scripts + data to rebuild and verify the validated strategy.
Pairs with `REPLICATION_GUIDE.md` (the full write-up). Python 3 + numpy/pandas (+ requests for the pullers).

## Layout
```
scripts/   config.py  regime.py  execution_engine.py  run_replication.py  walkforward.py
           ingest_ohlcv.py  ingest_funding.py  ingest_supply_3coin.py  ingest_fng*.py
           execution_selftest.py
Data/      price/  funding/  supply/  fear_greed/      (parquets ship pre-populated)
strategy_manifest.json                                  (validated states + provenance = ground truth)
```

## Quick start (verify it reproduces)
```
cd scripts
python run_replication.py      # -> ETH exec Sharpe ~1.12 / -23% ; hyst d=5 ~1.17 / -19% ; 95% manifest agreement
python execution_selftest.py   # -> 7/7
python walkforward.py          # -> stateless OOS 1.12 ; hyst d=5 1.16
```

## To refresh data from scratch (run locally; exchange/CM APIs need network)
```
python ingest_ohlcv.py         # edit SYMBOL for BTCUSDT/ETHUSDT; pull Coinbase ETH 1h separately
python ingest_funding.py       # per-asset Binance perp funding
python ingest_supply_3coin.py  # CRITICAL: 3-coin basket -> full_basket_supply.parquet
python ingest_fng.py           # alternative.me Fear & Greed
```

## The 3 things that must be right (see guide §1)
1. **3-coin supply** (`full_basket_supply.parquet`, with USDT-Tron) — not the 2-coin basket.
2. **Option A** = greed brake OFF, `require_both=True` (it's the default `OPTION_A` in `regime.py`).
3. **T+2** entry lag.
Canary: `run_replication.py` prints manifest agreement — must be **>=95%**.

## What's validated
- ETH execution Sharpe **1.12** (−23% DD), two-sided, lag-robust. Hysteresis d=5 -> ~1.16–1.17 (better DD), modest.
- BTC is a weak leg (~0.55 at T+2). XRP is excluded (no edge).
- Headline is risk reduction + two-sided edge, not raw return. ~1.5 cycles / one path — treat Sharpe as directional.

## Native (platform) engine — second, independent path
`regime_backtest_engine.py` is the company's two-file engine: it reads ONLY the manifest's
embedded daily states (`strategy_manifest.json` -> `regime.states`) plus in-memory bars — no
signal recompute, no external file paths — and runs the same intrabar execution.

```
cd scripts
python run_native_engine.py    # ETH ~1.14, BTC ~0.77, PORTFOLIO ~1.09; ledger reconciles: YES
```

This is an **independent cross-check** of `run_replication.py`: two separate engines, the same
strategy, the same answer (ETH 1.12 reconstruction vs 1.14 native). Differences are methodology,
not disagreement:
- Native annualizes Sharpe **per-bar** (sqrt 8760); the reconstruction resamples to **daily** (sqrt 365).
- Native arms a market order and fills the **next** bar's open; the reconstruction fills the current bar.
Both reconcile their trade ledger to the per-bar equity stream to the penny.

Manifest regen (when states need extending past 2026-06-03): use the engine's offline mode —
`python regime_backtest_engine.py --build-regime --manifest m.json --panel daily_panel.csv` —
which recomputes `regime.states` from a daily panel using the recorded `regime.provenance` recipe
(3-coin basket, Option A, T+2). The recipe in the shipped manifest is already the validated one.
