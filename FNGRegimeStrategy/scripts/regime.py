"""
regime.py -- daily Option-A regime (stateless + hysteresis), self-contained.

Consolidates the regime calculation. The ONLY config that reproduces the validated
result is OPTION_A on the 3-coin supply basket (see README). Functions take explicit
file paths so nothing is hard-wired to one asset.
"""
import numpy as np, pandas as pd

LONG, FLAT, SHORT = 1, 0, -1
TREND_N = 30
PCT_WINDOW = 365

# >>> THE VALIDATED CONFIG. greed brake OFF (long_brake=False); fear brake needs BOTH. <<<
OPTION_A = dict(supply_split=50, fear_extreme=20, greed_extreme=95,
                require_both=True, long_brake=False, greed_brake=False)

# ----------------------------- pure signal funcs -----------------------------
def breakout_state(close, n=TREND_N):
    prior_high = close.rolling(n, min_periods=n).max().shift(1)
    prior_low  = close.rolling(n, min_periods=n).min().shift(1)
    st = pd.Series(np.nan, index=close.index)
    st[close > prior_high] = 1.0
    st[close < prior_low]  = -1.0
    return st.ffill().fillna(0.0)

def trailing_pct_rank(s, window=PCT_WINDOW, min_valid=None):
    """Percentile (0-100) of today within trailing `window` days. Look-ahead-free, NaN-tolerant."""
    if min_valid is None:
        min_valid = max(2, window - 30)
    def _rank(w):
        if len(w) < window: return np.nan
        cur = w[-1]
        if not np.isfinite(cur): return np.nan
        valid = w[np.isfinite(w)]
        if len(valid) < min_valid: return np.nan
        return float((valid <= cur).mean() * 100.0)
    return s.rolling(window, min_periods=1).apply(_rank, raw=True)

# ----------------------------- decision rules --------------------------------
def decide(trend, supply_pct, funding_pct, fng_pct, cfg=OPTION_A):
    """One day -> LONG/FLAT/SHORT (stateless)."""
    if any(pd.isna(x) for x in (trend, supply_pct, funding_pct, fng_pct)): return FLAT
    if trend == 0: return FLAT
    split, fear, greed, rb = cfg["supply_split"], cfg["fear_extreme"], cfg["greed_extreme"], cfg["require_both"]
    supply_expanding = supply_pct > split
    f_hi, g_hi = funding_pct >= greed, fng_pct >= greed
    f_lo, g_lo = funding_pct <= fear, fng_pct <= fear
    greed_x = (f_hi and g_hi) if rb else (f_hi or g_hi)
    fear_x  = (f_lo and g_lo) if rb else (f_lo or g_lo)
    if trend > 0:
        if not supply_expanding: return FLAT
        if cfg["long_brake"] and cfg["greed_brake"] and greed_x: return FLAT
        return LONG
    else:
        if supply_expanding: return FLAT
        if fear_x: return FLAT
        return SHORT

def _brakes(fu, fg, cfg):
    rb, fear, greed = cfg["require_both"], cfg["fear_extreme"], cfg["greed_extreme"]
    f_hi, g_hi = fu >= greed, fg >= greed; f_lo, g_lo = fu <= fear, fg <= fear
    return ((f_hi and g_hi) if rb else (f_hi or g_hi)), ((f_lo and g_lo) if rb else (f_lo or g_lo))

def decide_hyst(prev, trend, sp, fu, fg, cfg=OPTION_A, delta=5):
    """Sticky-supply hysteresis: enter outside split+/-delta, hold inside the band."""
    if any(pd.isna(x) for x in (trend, sp, fu, fg)) or trend == 0: return FLAT
    hi, lo = cfg["supply_split"] + delta, cfg["supply_split"] - delta
    greed_x, fear_x = _brakes(fu, fg, cfg)
    brake_long = cfg["long_brake"] and cfg["greed_brake"] and greed_x
    if prev == LONG  and trend > 0 and sp >= lo and not brake_long: return LONG
    if prev == SHORT and trend < 0 and sp <= hi and not fear_x:     return SHORT
    if trend > 0: return LONG  if (sp > hi and not brake_long) else FLAT
    return SHORT if (sp < lo and not fear_x) else FLAT

# ----------------------------- loaders / panel -------------------------------
def _ms_day(s): return pd.to_datetime(s, unit="ms", utc=True).dt.tz_localize(None).dt.normalize()

def load_price_daily(price_file):
    df = pd.read_parquet(price_file); df["date"] = _ms_day(df["open_time"])
    return df.set_index("date")["close"].sort_index()

def load_supply_roc(supply_file):
    df = pd.read_parquet(supply_file); df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df.set_index("date")["roc_28d"].sort_index()

def load_funding_daily(funding_file):
    df = pd.read_parquet(funding_file)
    d = pd.to_datetime(df["funding_time"], unit="ms", utc=True).dt.tz_localize(None).dt.normalize()
    return df.groupby(d)["funding_rate"].sum().sort_index()

def load_fng(fng_file):
    df = pd.read_parquet(fng_file); df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df.set_index("date")["value"].astype(float).sort_index()

def build_panel(price_file, funding_file, supply_file, fng_file, window=PCT_WINDOW):
    close = load_price_daily(price_file)
    grid = pd.date_range(close.index.min(), close.index.max(), freq="D")
    close = close.reindex(grid).ffill(limit=3)
    sp = trailing_pct_rank(load_supply_roc(supply_file), window).reindex(grid).ffill(limit=3)
    fu = trailing_pct_rank(load_funding_daily(funding_file), window).reindex(grid).ffill(limit=3)
    fg = trailing_pct_rank(load_fng(fng_file), window).reindex(grid).ffill(limit=3)
    tr = breakout_state(close, TREND_N)
    pan = pd.DataFrame({"close": close, "trend": tr, "sp": sp, "fu": fu, "fg": fg}, index=grid)
    pan.index.name = "date"; return pan

def run_stateless(pan, cfg=OPTION_A):
    return pd.Series([decide(t, s, f, g, cfg) for t, s, f, g in zip(pan["trend"], pan["sp"], pan["fu"], pan["fg"])], index=pan.index)

def run_hyst(pan, cfg=OPTION_A, delta=5):
    out = np.empty(len(pan), int); prev = FLAT
    tr, sp, fu, fg = pan["trend"].values, pan["sp"].values, pan["fu"].values, pan["fg"].values
    for i in range(len(pan)):
        prev = decide_hyst(prev, tr[i], sp[i], fu[i], fg[i], cfg, delta); out[i] = prev
    return pd.Series(out, index=pan.index)
