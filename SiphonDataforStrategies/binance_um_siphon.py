#!/usr/bin/env python3
"""
binance_um_siphon.py
====================

Pull the complete 1-minute OHLCV + funding-rate history for EVERY Binance
USDⓈ-M perpetual that has ever existed (listed *and* delisted) and write
it out in the "Data Layout & File Format Reference" layout:

    <out>/ohlcv/<SYMBOL>.parquet      12 cols, 1-minute bars, RangeIndex
    <out>/funding/<SYMBOL>.parquet     4 cols, one row per settlement
    <out>/_meta/symbols.csv            coverage / status summary per symbol
    <out>/_cache/...                   raw archive zips (makes re-runs cheap)

Sources
-------
1. Binance public data archive (data.binance.vision, S3 bucket).  This is
   the only place delisted symbols survive, and it is not geo-restricted.
   Monthly zips are used first, daily zips fill the current month, so the
   archive lags real time by ~1 day for klines and up to ~1 month for
   funding.
2. Binance Futures REST API (fapi.binance.com, no key needed) fills the
   remaining tail for symbols that are still trading.  If the API is
   unreachable (e.g. geo-blocked) the script just skips the tail.

Usage
-----
    pip install pandas pyarrow requests
    python binance_um_siphon.py --out /data/binance_um
    python binance_um_siphon.py --out /data/binance_um --symbols BTCUSDT ETHUSDT
    python binance_um_siphon.py --out /data/binance_um --workers 12 --no-rest

Re-running is incremental: archive files already in _cache are not
re-downloaded, and only the tail is fetched from the API.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import logging
import re
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd
import requests

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
S3_LIST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
S3_DL = "https://data.binance.vision"
FAPI = "https://fapi.binance.com"

KLINE_PREFIX_M = "data/futures/um/monthly/klines/"
KLINE_PREFIX_D = "data/futures/um/daily/klines/"
FUND_PREFIX_M = "data/futures/um/monthly/fundingRate/"

ARCHIVE_KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume",
    "ignore",
]
ARCHIVE_FUND_COLS = ["calc_time", "funding_interval_hours", "last_funding_rate"]

OHLCV_COLS = [
    "timestamp", "open", "high", "low", "close", "volume", "quote_volume",
    "trades", "taker_buy_volume", "taker_buy_quote_volume", "open_time",
    "close_time",
]
FUNDING_COLS = ["timestamp", "funding_rate", "funding_interval_hours", "calc_time"]

MINUTE_MS = 60_000
REST_TAIL_MAX_DAYS = 45          # only try the API when archive ends recently
REST_KLINE_LIMIT = 1500
REST_FUND_LIMIT = 1000

log = logging.getLogger("siphon")


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #
class Http:
    """Thread-safe requests session with retry/backoff."""

    def __init__(self, timeout: int = 60, retries: int = 5):
        self.timeout = timeout
        self.retries = retries
        self._local = threading.local()
        self._rest_lock = threading.Lock()
        self._rest_next_ok = 0.0

    @property
    def s(self) -> requests.Session:
        if not hasattr(self._local, "s"):
            s = requests.Session()
            s.headers["User-Agent"] = "binance-um-siphon/1.0"
            self._local.s = s
        return self._local.s

    def get(self, url: str, params: dict | None = None, ok404: bool = False):
        delay = 1.0
        for attempt in range(self.retries):
            try:
                r = self.s.get(url, params=params, timeout=self.timeout)
                if r.status_code == 404 and ok404:
                    return None
                if r.status_code in (418, 429):          # Binance rate limit
                    wait = float(r.headers.get("Retry-After", 30))
                    log.warning("rate limited, sleeping %.0fs", wait)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                return r
            except requests.RequestException as e:
                if attempt == self.retries - 1:
                    raise
                log.debug("retry %s (%s)", url, e)
                time.sleep(delay)
                delay = min(delay * 2, 30)
        raise RuntimeError(f"unreachable: {url}")

    def rest(self, path: str, params: dict | None = None):
        """GET on fapi with a crude global pacing (~10 req/s)."""
        with self._rest_lock:
            now = time.time()
            if now < self._rest_next_ok:
                time.sleep(self._rest_next_ok - now)
            self._rest_next_ok = time.time() + 0.1
        r = self.get(FAPI + path, params=params)
        return r.json()


HTTP = Http()


# --------------------------------------------------------------------------- #
# S3 listing
# --------------------------------------------------------------------------- #
_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def s3_list(prefix: str, delimiter: str = "/") -> tuple[list[str], list[str]]:
    """Return (common_prefixes, keys) under *prefix*, following pagination."""
    prefixes: list[str] = []
    keys: list[str] = []
    marker = None
    while True:
        params = {"prefix": prefix, "delimiter": delimiter}
        if marker:
            params["marker"] = marker
        r = HTTP.get(S3_LIST, params=params)
        root = ET.fromstring(r.content)
        for cp in root.findall("s3:CommonPrefixes/s3:Prefix", _NS):
            prefixes.append(cp.text)
        for k in root.findall("s3:Contents/s3:Key", _NS):
            keys.append(k.text)
        truncated = root.findtext("s3:IsTruncated", "false", _NS) == "true"
        if not truncated:
            break
        marker = root.findtext("s3:NextMarker", None, _NS) or (keys[-1] if keys else prefixes[-1])
    return prefixes, keys


def list_symbols(prefix: str) -> set[str]:
    prefixes, _ = s3_list(prefix)
    return {p[len(prefix):].strip("/") for p in prefixes}


def list_zip_keys(prefix: str) -> list[str]:
    _, keys = s3_list(prefix)
    return sorted(k for k in keys if k.endswith(".zip"))


# --------------------------------------------------------------------------- #
# archive download + parse
# --------------------------------------------------------------------------- #
def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch_zip(key: str, cache_dir: Path, verify: bool = True) -> Path | None:
    """Download an archive zip into the cache (if missing); return its path."""
    dest = cache_dir / key
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        r = HTTP.get(f"{S3_DL}/{key}", ok404=True)
        if r is None:
            return None
        data = r.content
        if verify:
            c = HTTP.get(f"{S3_DL}/{key}.CHECKSUM", ok404=True)
            if c is not None:
                expected = c.text.split()[0].strip().lower()
                if _sha256(data) != expected:
                    log.warning("checksum mismatch %s (attempt %d)", key, attempt + 1)
                    continue
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(dest)
        return dest
    raise RuntimeError(f"checksum kept failing for {key}")


def _read_csv_from_zip(path: Path, cols: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        with z.open(name) as f:
            head = f.readline()
        first = head.decode().split(",")[0].strip()
        header = not first.lstrip("-").isdigit()        # old files have no header row
        with z.open(name) as f:
            return pd.read_csv(
                io.BytesIO(f.read()),
                header=0 if header else None,
                names=cols,
                dtype=str,          # parse numerics ourselves (avoid float/int surprises)
            )


def _to_ms(series: pd.Series) -> pd.Series:
    """Archive timestamps are ms; newer spot files are µs.  Normalise to ms."""
    s = pd.to_numeric(series, errors="coerce").astype("int64")
    if len(s) and s.abs().max() > 10**14:          # µs
        s = s // 1000
    return s


def parse_kline_zip(path: Path) -> pd.DataFrame:
    raw = _read_csv_from_zip(path, ARCHIVE_KLINE_COLS)
    return _klines_to_frame(raw)


def _klines_to_frame(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({
        "open_time": _to_ms(raw["open_time"]),
        "open": pd.to_numeric(raw["open"], errors="coerce").astype("float64"),
        "high": pd.to_numeric(raw["high"], errors="coerce").astype("float64"),
        "low": pd.to_numeric(raw["low"], errors="coerce").astype("float64"),
        "close": pd.to_numeric(raw["close"], errors="coerce").astype("float64"),
        "volume": pd.to_numeric(raw["volume"], errors="coerce").astype("float64"),
        "quote_volume": pd.to_numeric(raw["quote_volume"], errors="coerce").astype("float64"),
        "trades": pd.to_numeric(raw["count"], errors="coerce").astype("Int64"),
        "taker_buy_volume": pd.to_numeric(raw["taker_buy_volume"], errors="coerce").astype("float64"),
        "taker_buy_quote_volume": pd.to_numeric(raw["taker_buy_quote_volume"], errors="coerce").astype("float64"),
    })
    # close_time is always open_time + 59999 for 1m bars; derive it rather
    # than trusting the file (µs files / API would otherwise disagree).
    df["close_time"] = df["open_time"] + (MINUTE_MS - 1)
    return df


def parse_funding_zip(path: Path) -> pd.DataFrame:
    raw = _read_csv_from_zip(path, ARCHIVE_FUND_COLS)
    return pd.DataFrame({
        "calc_time": _to_ms(raw["calc_time"]),
        "funding_rate": pd.to_numeric(raw["last_funding_rate"], errors="coerce").astype("float64"),
        "funding_interval_hours": pd.to_numeric(raw["funding_interval_hours"], errors="coerce").astype("Int64"),
    })


# --------------------------------------------------------------------------- #
# REST tail (live symbols only)
# --------------------------------------------------------------------------- #
class Rest:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.status: dict[str, str] = {}
        self.interval_hours: dict[str, int] = {}
        if not enabled:
            return
        try:
            info = HTTP.rest("/fapi/v1/exchangeInfo")
            self.status = {s["symbol"]: s["status"] for s in info["symbols"]}
            try:
                for fi in HTTP.rest("/fapi/v1/fundingInfo"):
                    self.interval_hours[fi["symbol"]] = int(fi["fundingIntervalHours"])
            except Exception as e:                      # noqa: BLE001
                log.warning("fundingInfo unavailable: %s", e)
            log.info("REST API reachable: %d symbols in exchangeInfo", len(self.status))
        except Exception as e:                          # noqa: BLE001
            log.warning("REST API unreachable (%s) -> archive only", str(e)[:200])
            self.enabled = False

    def klines_since(self, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        rows = []
        cur = start_ms
        while cur < end_ms:
            batch = HTTP.rest("/fapi/v1/klines", {
                "symbol": symbol, "interval": "1m", "startTime": cur,
                "endTime": end_ms, "limit": REST_KLINE_LIMIT,
            })
            if not batch:
                break
            rows.extend(batch)
            last_open = int(batch[-1][0])
            if len(batch) < REST_KLINE_LIMIT or last_open <= cur:
                break
            cur = last_open + MINUTE_MS
        if not rows:
            return pd.DataFrame(columns=OHLCV_COLS[1:])
        raw = pd.DataFrame(rows, columns=ARCHIVE_KLINE_COLS).astype(str)
        return _klines_to_frame(raw)

    def funding_since(self, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        rows = []
        cur = start_ms
        while cur < end_ms:
            batch = HTTP.rest("/fapi/v1/fundingRate", {
                "symbol": symbol, "startTime": cur, "endTime": end_ms,
                "limit": REST_FUND_LIMIT,
            })
            if not batch:
                break
            rows.extend(batch)
            last = int(batch[-1]["fundingTime"])
            if len(batch) < REST_FUND_LIMIT or last <= cur:
                break
            cur = last + 1
        if not rows:
            return pd.DataFrame(columns=["calc_time", "funding_rate", "funding_interval_hours"])
        df = pd.DataFrame({
            "calc_time": [int(r["fundingTime"]) for r in rows],
            "funding_rate": [float(r["fundingRate"]) for r in rows],
        })
        # The API does not return the interval; use the current interval
        # from fundingInfo (8h default) for these recent rows, else <NA>.
        ih = self.interval_hours.get(symbol, 8)
        df["funding_interval_hours"] = pd.Series([ih] * len(df), dtype="Int64")
        return df


# --------------------------------------------------------------------------- #
# per-symbol pipeline
# --------------------------------------------------------------------------- #
def _ms_to_ts(ms: pd.Series) -> pd.Series:
    return pd.to_datetime(ms, unit="ms", utc=True).astype("datetime64[ms, UTC]")


def finalize_ohlcv(parts: list[pd.DataFrame]) -> pd.DataFrame:
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=OHLCV_COLS)
    df = pd.concat(parts, ignore_index=True)
    df = df.dropna(subset=["open_time"])
    df = df.drop_duplicates("open_time", keep="last").sort_values("open_time")
    df["timestamp"] = _ms_to_ts(df["open_time"])
    df = df[OHLCV_COLS].reset_index(drop=True)
    df["trades"] = df["trades"].astype("Int64")
    df["open_time"] = df["open_time"].astype("int64")
    df["close_time"] = df["close_time"].astype("int64")
    return df


def finalize_funding(parts: list[pd.DataFrame]) -> pd.DataFrame:
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=FUNDING_COLS)
    df = pd.concat(parts, ignore_index=True)
    df = df.dropna(subset=["calc_time"])
    df = df.drop_duplicates("calc_time", keep="first").sort_values("calc_time")
    df["timestamp"] = _ms_to_ts(df["calc_time"])      # keep the ms jitter, as in the spec
    df = df[FUNDING_COLS].reset_index(drop=True)
    df["funding_interval_hours"] = df["funding_interval_hours"].astype("Int64")
    df["calc_time"] = df["calc_time"].astype("int64")
    return df


_MONTH_RE = re.compile(r"-(\d{4}-\d{2})\.zip$")
_DAY_RE = re.compile(r"-(\d{4}-\d{2}-\d{2})\.zip$")


def process_symbol(symbol: str, out: Path, cache: Path, rest: Rest, verify: bool) -> dict:
    t0 = time.time()
    now_ms = int(time.time() * 1000)
    meta: dict = {"symbol": symbol, "status": rest.status.get(symbol, "DELISTED" if rest.enabled else "UNKNOWN")}

    # ---------------- OHLCV ----------------
    monthly = list_zip_keys(f"{KLINE_PREFIX_M}{symbol}/1m/")
    months_covered = {m.group(1) for k in monthly if (m := _MONTH_RE.search(k))}
    daily = [k for k in list_zip_keys(f"{KLINE_PREFIX_D}{symbol}/1m/")
             if (d := _DAY_RE.search(k)) and d.group(1)[:7] not in months_covered]

    parts = []
    for key in monthly + daily:
        p = fetch_zip(key, cache, verify)
        if p is not None:
            try:
                parts.append(parse_kline_zip(p))
            except Exception as e:                      # noqa: BLE001
                log.error("%s: bad zip %s (%s) -> deleting from cache", symbol, key, e)
                p.unlink(missing_ok=True)
    ohlcv = finalize_ohlcv(parts)

    # REST tail: only if the symbol is trading and the archive ends recently
    last_ms = int(ohlcv["open_time"].iloc[-1]) if len(ohlcv) else None
    rest_rows = 0
    if rest.enabled and rest.status.get(symbol) == "TRADING":
        start = last_ms + MINUTE_MS if last_ms else None
        recent = last_ms is None or (now_ms - last_ms) < REST_TAIL_MAX_DAYS * 86_400_000
        if recent:
            end = (now_ms // MINUTE_MS) * MINUTE_MS - 1       # exclude the still-open bar
            try:
                tail = rest.klines_since(symbol, start or 0, end)
                tail = tail[tail["close_time"] <= end]
                rest_rows = len(tail)
                if rest_rows:
                    ohlcv = finalize_ohlcv([ohlcv, tail])
            except Exception as e:                          # noqa: BLE001
                log.warning("%s: REST kline tail failed: %s", symbol, str(e)[:200])

    # ---------------- funding ----------------
    fparts = []
    for key in list_zip_keys(f"{FUND_PREFIX_M}{symbol}/"):
        p = fetch_zip(key, cache, verify)
        if p is not None:
            try:
                fparts.append(parse_funding_zip(p))
            except Exception as e:                      # noqa: BLE001
                log.error("%s: bad funding zip %s (%s)", symbol, key, e)
                p.unlink(missing_ok=True)
    funding = finalize_funding(fparts)
    if rest.enabled and rest.status.get(symbol) == "TRADING":
        flast = int(funding["calc_time"].iloc[-1]) if len(funding) else 0
        if (now_ms - flast) < 62 * 86_400_000 or flast == 0:
            try:
                ftail = rest.funding_since(symbol, flast + 1, now_ms)
                if len(ftail):
                    funding = finalize_funding([funding, ftail])
            except Exception as e:                      # noqa: BLE001
                log.warning("%s: REST funding tail failed: %s", symbol, str(e)[:200])

    # ---------------- write ----------------
    (out / "ohlcv").mkdir(parents=True, exist_ok=True)
    (out / "funding").mkdir(parents=True, exist_ok=True)
    if len(ohlcv):
        _atomic_parquet(ohlcv, out / "ohlcv" / f"{symbol}.parquet")
    if len(funding):
        _atomic_parquet(funding, out / "funding" / f"{symbol}.parquet")

    # ---------------- meta ----------------
    if len(ohlcv):
        span = (ohlcv["open_time"].iloc[-1] - ohlcv["open_time"].iloc[0]) // MINUTE_MS + 1
        meta.update(
            ohlcv_rows=len(ohlcv),
            first_bar=str(ohlcv["timestamp"].iloc[0]),
            last_bar=str(ohlcv["timestamp"].iloc[-1]),
            missing_minutes=int(span - len(ohlcv)),
            rest_tail_rows=rest_rows,
        )
    else:
        meta.update(ohlcv_rows=0, first_bar=None, last_bar=None, missing_minutes=None, rest_tail_rows=0)
    if len(funding):
        gaps = funding["calc_time"].diff().dropna() / 3_600_000
        meta.update(
            funding_rows=len(funding),
            first_funding=str(funding["timestamp"].iloc[0]),
            last_funding=str(funding["timestamp"].iloc[-1]),
            funding_intervals_seen=",".join(str(int(round(g))) for g in sorted(gaps.round().unique())),
        )
    else:
        meta.update(funding_rows=0, first_funding=None, last_funding=None, funding_intervals_seen="")
    meta["seconds"] = round(time.time() - t0, 1)
    return meta


def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, engine="pyarrow", index=False, compression="zstd")
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def discover_symbols(quotes: list[str], include_delivery: bool) -> list[str]:
    syms = list_symbols(KLINE_PREFIX_M) | list_symbols(KLINE_PREFIX_D)
    keep = []
    for s in sorted(syms):
        if not re.fullmatch(r"[A-Z0-9_]+", s):          # a few junk folders with non-ASCII names exist
            log.debug("skipping odd symbol folder %r", s)
            continue
        if "_" in s and not include_delivery:          # e.g. BTCUSDT_230331 quarterlies
            continue
        if quotes and not any(s.endswith(q) for q in quotes):
            continue
        keep.append(s)
    return keep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path, help="output root folder")
    ap.add_argument("--symbols", nargs="*", help="only these symbols (default: every symbol in the archive)")
    ap.add_argument("--quote", nargs="*", default=["USDT"],
                    help="quote-asset suffixes to keep (default USDT). Use --quote '' for all")
    ap.add_argument("--include-delivery", action="store_true", help="also keep quarterly contracts (BTCUSDT_240329 ...)")
    ap.add_argument("--workers", type=int, default=8, help="parallel symbols (default 8)")
    ap.add_argument("--no-rest", action="store_true", help="archive only, skip the REST API tail")
    ap.add_argument("--no-verify", action="store_true", help="skip sha256 checksum verification of zips")
    ap.add_argument("--cache", type=Path, help="zip cache dir (default <out>/_cache)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
    )
    out: Path = args.out
    cache: Path = args.cache or out / "_cache"
    out.mkdir(parents=True, exist_ok=True)
    (out / "_meta").mkdir(exist_ok=True)

    quotes = [q for q in (args.quote or []) if q]
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        log.info("discovering symbols in the archive ...")
        symbols = discover_symbols(quotes, args.include_delivery)
    log.info("%d symbols to process", len(symbols))

    rest = Rest(enabled=not args.no_rest)
    if rest.enabled:
        live = sum(1 for s in symbols if rest.status.get(s) == "TRADING")
        log.info("%d currently TRADING, %d delisted/other", live, len(symbols) - live)

    metas: list[dict] = []
    failed: list[str] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_symbol, s, out, cache, rest, not args.no_verify): s for s in symbols}
        for fut in as_completed(futs):
            s = futs[fut]
            done += 1
            try:
                m = fut.result()
                metas.append(m)
                log.info("[%d/%d] %-16s %8s bars  %6s funding  %s -> %s  (%.0fs)",
                         done, len(symbols), s, m["ohlcv_rows"], m["funding_rows"],
                         (m["first_bar"] or "")[:10], (m["last_bar"] or "")[:16], m["seconds"])
            except Exception as e:                      # noqa: BLE001
                failed.append(s)
                log.error("[%d/%d] %s FAILED: %s", done, len(symbols), s, e)

    if metas:
        pd.DataFrame(metas).sort_values("symbol").to_csv(out / "_meta" / "symbols.csv", index=False)
    if failed:
        (out / "_meta" / "failed.txt").write_text("\n".join(failed) + "\n")
        log.error("%d symbols failed: %s", len(failed), " ".join(failed[:20]))
    log.info("done. %d ok, %d failed. output in %s", len(metas), len(failed), out)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
