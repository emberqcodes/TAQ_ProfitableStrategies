"""Shared helpers: loading raw parquet files with schema checks."""
import pandas as pd
import config as C

OHLCV_REQUIRED = ["timestamp", "open", "high", "low", "close", "volume",
                  "quote_volume", "trades", "taker_buy_volume",
                  "taker_buy_quote_volume", "open_time", "close_time"]
FUNDING_REQUIRED = ["timestamp", "funding_rate"]


def _ensure_utc(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    if s.dt.tz is None:
        s = s.dt.tz_localize("UTC")
    else:
        s = s.dt.tz_convert("UTC")
    # force one resolution everywhere; mixed ms/us/ns units break merge_asof / reindex
    return s.dt.as_unit("ns")


def load_ohlcv_1m(symbol: str) -> pd.DataFrame:
    path = C.RAW_OHLCV_DIR / f"{symbol}.parquet"
    df = pd.read_parquet(path)
    missing = [c for c in OHLCV_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{symbol}: OHLCV missing columns {missing}")
    df["timestamp"] = _ensure_utc(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="first")
    df = df.reset_index(drop=True)
    return df


def load_funding(symbol: str) -> pd.DataFrame:
    path = C.RAW_FUNDING_DIR / f"{symbol}.parquet"
    df = pd.read_parquet(path)
    missing = [c for c in FUNDING_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{symbol}: funding missing columns {missing}")
    df["timestamp"] = _ensure_utc(df["timestamp"])
    # ms jitter documented (:00.003) -> floor to the minute BEFORE any join
    df["settle_ts"] = df["timestamp"].dt.floor("min")
    df = df.sort_values("settle_ts").drop_duplicates("settle_ts", keep="last")
    df = df.reset_index(drop=True)
    return df
