# binance_um_siphon

Pulls the full 1-minute OHLCV and funding-rate history for every Binance
USDⓈ-M perpetual that has ever existed — currently listed **and** delisted —
and writes one parquet per symbol in the *Data Layout & File Format Reference*
schema.

## Requirements

- Python 3.10+
- `pip install pandas pyarrow requests`
- Network access to `data.binance.vision` (the public archive, not geo-restricted)
  and optionally `fapi.binance.com` (REST API, geo-restricted in some regions)

## Quick start

```bash
python binance_um_siphon.py --out /data/binance_um
```

That discovers every USDT-quoted perp in the archive (~900 symbols), downloads
their history and writes:

```
/data/binance_um/
├── ohlcv/            one <SYMBOL>.parquet per asset, 1-minute bars
├── funding/          one <SYMBOL>.parquet per asset, one row per settlement
├── _meta/
│   ├── symbols.csv   coverage summary per symbol (see below)
│   └── failed.txt    only written if some symbols failed
└── _cache/           raw archive zips; keeps re-runs cheap, safe to delete
```

Re-running the same command is incremental: zips already in `_cache` are not
downloaded again, and only new months/days plus the REST tail are fetched.

## Options

| flag | default | meaning |
|---|---|---|
| `--out PATH` | required | output root |
| `--symbols A B ...` | all | restrict to these symbols |
| `--quote USDT ...` | `USDT` | quote-asset suffixes to keep; `--quote ''` keeps everything (USDC, BUSD, BTC…) |
| `--include-delivery` | off | also keep quarterly contracts such as `BTCUSDT_240329` |
| `--workers N` | 8 | symbols processed in parallel |
| `--no-rest` | off | archive only; skip the REST API tail |
| `--no-verify` | off | skip sha256 verification of downloaded zips |
| `--cache PATH` | `<out>/_cache` | where to keep raw zips |
| `-v` | off | debug logging |

Examples:

```bash
# just a few symbols
python binance_um_siphon.py --out /data/binance_um --symbols BTCUSDT ETHUSDT SOLUSDT

# everything, including USDC/BUSD pairs and quarterlies, archive only
python binance_um_siphon.py --out /data/binance_um --quote '' --include-delivery --no-rest

# more parallelism, less logging noise
python binance_um_siphon.py --out /data/binance_um --workers 16 2> run.log
```

## How it sources data

1. **Binance public archive** (`data.binance.vision`). This is the only place
   delisted symbols survive. Monthly zips are used first; daily zips fill the
   current month. Checksums are verified. The archive lags real time by about
   one day for klines and up to one month for funding (funding has monthly
   files only).
2. **REST API** (`fapi.binance.com`, no key needed). For symbols whose status is
   `TRADING` the script fills the remaining tail: klines up to the last fully
   closed minute, funding up to now. The still-open bar is never written.
   If the API is unreachable (e.g. HTTP 451 geo-block) a warning is logged and
   the run continues archive-only.

Delisted symbols therefore end at their last archived bar; nothing more exists
for them anywhere.

## Output schema

### `ohlcv/<SYMBOL>.parquet` (12 columns, RangeIndex)

| column | dtype | notes |
|---|---|---|
| `timestamp` | `datetime64[ms, UTC]` | bar **open** time, identical to `open_time` |
| `open, high, low, close` | `float64` | |
| `volume` | `float64` | base asset |
| `quote_volume` | `float64` | quote (USDT) |
| `trades` | `Int64` | |
| `taker_buy_volume` | `float64` | base; taker-sell = `volume - taker_buy_volume` |
| `taker_buy_quote_volume` | `float64` | quote |
| `open_time` | `int64` | epoch ms |
| `close_time` | `int64` | epoch ms, always `open_time + 59999` |

Sorted ascending, no duplicate timestamps, no NaNs. The first bar is the
listing minute. Gaps are *not* filled — check `missing_minutes` in
`_meta/symbols.csv` per asset.

### `funding/<SYMBOL>.parquet` (4 columns, RangeIndex)

| column | dtype | notes |
|---|---|---|
| `timestamp` | `datetime64[ms, UTC]` | settlement time; ms jitter preserved (`:00.003`) — floor/round before joining |
| `funding_rate` | `float64` | per-interval rate, not annualised |
| `funding_interval_hours` | `Int64` (nullable) | from the archive where available; REST-tail rows use the symbol's *current* interval. Unreliable — derive the real interval from consecutive `timestamp` diffs |
| `calc_time` | `int64` | epoch ms, equal to `timestamp` |

Funding coverage usually starts earlier and ends later than the price series.
Intersect explicitly.

### `_meta/symbols.csv`

One row per symbol: `status` (`TRADING` / `DELISTED` / other exchangeInfo
status, or `UNKNOWN` when the REST API was unreachable), `ohlcv_rows`,
`first_bar`, `last_bar`, `missing_minutes`, `rest_tail_rows`, `funding_rows`,
`first_funding`, `last_funding`, `funding_intervals_seen` (hours, from
timestamp diffs), `seconds`.

## Sizing

A full-history symbol takes roughly 45 s at 8 workers (network bound), so the
first complete pull of ~900 symbols runs for a few hours. Budget on the order
of 20 GB for the raw zip cache plus about half that again for the parquets.
Delete `_cache/` once you are happy with the parquets if space matters; the
next run will simply re-download what it needs.

## Loading

```python
import pandas as pd
bars = pd.read_parquet("/data/binance_um/ohlcv/BTCUSDT.parquet")
fund = pd.read_parquet("/data/binance_um/funding/BTCUSDT.parquet")

# join funding onto the minute grid as an event series
fund["minute"] = fund["timestamp"].dt.floor("min")
bars = bars.merge(fund[["minute", "funding_rate"]], left_on="timestamp",
                  right_on="minute", how="left").drop(columns="minute")
bars["funding_rate"] = bars["funding_rate"].fillna(0.0)
```

## Troubleshooting

- `REST API unreachable (451 ...)` — the API is geo-blocked where you run;
  everything still works from the archive, just one day behind.
- `rate limited, sleeping` — the API returned 429/418; the script backs off
  automatically. Lower `--workers` if it keeps happening.
- `checksum mismatch` — a transient CDN problem; the file is re-downloaded.
  If it persists for one key, run with `--no-verify` for that symbol.
- Symbols listed in `_meta/failed.txt` can simply be re-run with `--symbols`.
