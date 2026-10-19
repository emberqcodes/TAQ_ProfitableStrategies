"""
TaQuant Python Strategy -- Intraday Momentum (Noise-Area Breakout) -- Crypto Perp -- v1

manifest = {
  "params": {
    "type": "object",
    "properties": {
      "risk_frac":        {"type": "number", "default": 0.0001, "minimum": 0.0, "title": "Risk per trade (fraction of equity, 0.0001 = 0.01%)"},
      "max_leverage":     {"type": "number", "default": 20,     "minimum": 1,   "title": "Max leverage (x capital)"},
      "capital_usd":      {"type": "number", "default": 100000, "minimum": 1,   "title": "Sizing capital (USD) -- set UI Initial Capital to match"},
      "cat_stop_bps":     {"type": "number", "default": 400,    "minimum": 1,   "title": "Catastrophic stop distance (bps from entry)"},
      "anchor_utc_min":   {"type": "number", "default": 810,    "minimum": 0,   "title": "Session anchor, minutes past 00:00 UTC (810 = 13:30)"},
      "close_utc_min":    {"type": "number", "default": 1439,   "minimum": 1,   "title": "Forced-flat minute past 00:00 UTC (1439 = 23:59)"},
      "mark_interval_min":{"type": "number", "default": 30,     "minimum": 1,   "title": "Decision-mark interval (minutes)"},
      "first_mark_min":   {"type": "number", "default": 840,    "minimum": 0,   "title": "First decision mark, minutes past 00:00 UTC (840 = 14:00)"},
      "noise_lookback":   {"type": "number", "default": 14,     "minimum": 2,   "title": "Noise-area lookback (sessions)"},
      "min_obs":          {"type": "number", "default": 10,     "minimum": 1,   "title": "Min valid prior sessions for a band"},
      "band_k":           {"type": "number", "default": 1.0,    "minimum": 0.1, "title": "Band width multiplier k"},
      "vol_conf_k":       {"type": "number", "default": 2.0,    "minimum": 0.0, "title": "Volume-confirmation multiple K (0 = off)"},
      "vol_window_min":   {"type": "number", "default": 30,     "minimum": 1,   "title": "Volume-confirmation window (minutes)"},
      "stop_style":       {"type": "string", "default": "band", "enum": ["band", "vwap"], "title": "Dynamic stop style"},
      "taker_bps":        {"type": "number", "default": 2.0,    "minimum": 0.0, "title": "Taker fee per side (bps) -- informational; engine applies commission"},
      "news_dates":       {"type": "string", "default": "2020-01-10,2020-01-14,2020-01-29,2020-02-07,2020-02-13,2020-03-03,2020-03-06,2020-03-11,2020-03-15,2020-04-03,2020-04-10,2020-04-29,2020-05-08,2020-05-12,2020-06-05,2020-06-10,2020-07-02,2020-07-14,2020-07-29,2020-08-07,2020-08-12,2020-09-04,2020-09-11,2020-09-16,2020-10-02,2020-10-13,2020-11-05,2020-11-06,2020-11-12,2020-12-04,2020-12-10,2020-12-16,2021-01-08,2021-01-13,2021-01-27,2021-02-05,2021-02-10,2021-03-05,2021-03-10,2021-03-17,2021-04-02,2021-04-13,2021-04-28,2021-05-07,2021-05-12,2021-06-04,2021-06-10,2021-06-16,2021-07-02,2021-07-13,2021-07-28,2021-08-06,2021-08-11,2021-09-03,2021-09-14,2021-09-22,2021-10-08,2021-10-13,2021-11-03,2021-11-05,2021-11-10,2021-12-03,2021-12-10,2021-12-15,2022-01-07,2022-01-12,2022-01-26,2022-02-04,2022-02-10,2022-03-04,2022-03-10,2022-03-16,2022-04-01,2022-04-12,2022-05-04,2022-05-06,2022-05-11,2022-06-03,2022-06-10,2022-06-15,2022-07-08,2022-07-13,2022-07-27,2022-08-05,2022-08-10,2022-09-02,2022-09-13,2022-09-21,2022-10-07,2022-10-13,2022-11-02,2022-11-04,2022-11-10,2022-12-02,2022-12-13,2022-12-14,2023-01-06,2023-01-12,2023-02-01,2023-02-03,2023-02-14,2023-03-10,2023-03-14,2023-03-22,2023-04-07,2023-04-12,2023-05-03,2023-05-05,2023-05-10,2023-06-02,2023-06-13,2023-06-14,2023-07-07,2023-07-12,2023-07-26,2023-08-04,2023-08-10,2023-09-01,2023-09-13,2023-09-20,2023-10-06,2023-10-12,2023-11-01,2023-11-03,2023-11-14,2023-12-08,2023-12-12,2023-12-13,2024-01-05,2024-01-11,2024-01-31,2024-02-02,2024-02-13,2024-03-08,2024-03-12,2024-03-20,2024-04-05,2024-04-10,2024-05-01,2024-05-03,2024-05-15,2024-06-07,2024-06-12,2024-07-05,2024-07-11,2024-07-31,2024-08-02,2024-08-14,2024-09-06,2024-09-11,2024-09-18,2024-10-04,2024-10-10,2024-11-01,2024-11-07,2024-11-13,2024-12-06,2024-12-11,2024-12-18,2025-01-10,2025-01-15,2025-01-29,2025-02-07,2025-02-12,2025-03-07,2025-03-12,2025-03-19,2025-04-04,2025-04-10,2025-05-02,2025-05-07,2025-05-13,2025-06-06,2025-06-11,2025-06-18,2025-07-03,2025-07-15,2025-07-30,2025-08-01,2025-08-12,2025-09-05,2025-09-11,2025-09-17,2025-10-03,2025-10-15,2025-10-24,2025-10-29,2025-11-20,2025-12-10,2025-12-16,2025-12-18,2026-01-09,2026-01-13,2026-01-28,2026-02-11,2026-02-13,2026-03-06,2026-03-11,2026-03-18,2026-04-03,2026-04-10,2026-04-29,2026-05-08,2026-05-12,2026-06-05,2026-06-10,2026-06-17,2026-07-02,2026-07-14,2026-07-29,2026-08-07,2026-08-12,2026-09-04,2026-09-11,2026-09-16,2026-10-02,2026-10-14,2026-10-28,2026-11-06,2026-11-10,2026-12-04,2026-12-09,2026-12-10", "title": "News blackout dates (comma-separated YYYY-MM-DD; CPI/NFP/FOMC). No entries on these UTC dates."}
    }
  }
}
"""

# ---------------------------------------------------------------------------------------
# Per-asset port of the pre-registered intraday noise-area momentum strategy.
#
# RUN ON: 1-MINUTE timeframe, a single PERP / USD-M futures market (two-sided; the short
#   leg will not fill on Spot). One instance per symbol. Set UI Initial Capital = capital_usd.
#   Use the longest available period. Commission: set the engine to your taker tier.
#
# Pipeline (causal, matches the validated research engine):
#   session anchored at 13:30 UTC -> noise-area bands from the prior N sessions (frozen at
#   the open) -> half-hour decision marks -> breakout entry beyond band, optionally
#   volume-confirmed -> next-bar-open fill (engine market order on the mark bar) -> dynamic
#   VWAP/band stop + continuous intrabar catastrophic stop -> opposite-band flip ->
#   forced flat at 23:59 UTC. No entries on news-blackout dates.
#
# SIZING (fixed-fractional on stop distance):
#   notional = min( capital_usd * risk_frac / (cat_stop_bps/1e4), max_leverage * capital_usd )
#   qty = notional / price. risk_frac and max_leverage are RISK-BUDGET choices, not tuned.
#
# CAUSALITY: bands and volume baselines for session D use only sessions < D (rolled at each
#   session close). Nothing uses the current or a future session. Marks act on the closing
#   1-minute bar and the position changes on that bar (engine fills at market).
# ---------------------------------------------------------------------------------------

import datetime
from taquant_sdk import StrategyContext


def _f(bar, *names):
    for nm in names:
        if isinstance(bar, dict) and nm in bar:
            return float(bar[nm])
    raise KeyError(names)


def _bar_dt(ts):
    t = ts / 1000.0 if ts > 1e12 else float(ts)
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc)


def _p(ctx, name, dflt):
    try:
        v = float(ctx.config.get(name, dflt))
        return v
    except Exception:
        return dflt


def _ps(ctx, name, dflt):
    try:
        v = ctx.config.get(name, dflt)
        return str(v) if v is not None else dflt
    except Exception:
        return dflt


def initialize(ctx: StrategyContext):
    st = ctx.state
    st.update(dict(
        # session bookkeeping
        cur_date=None, open_px=None, prev_close=None,
        cum_pv=0.0, cum_v=0.0,                     # for running VWAP
        # per-mark abs-move profile of the CURRENT session (minute-of-session -> |ret|)
        sess_absmove={},
        # rolling history of prior sessions: list of dicts {mod: absmove}, and dollar-vol windows
        hist_absmove=[],                           # each: {mos_minute: abs_move_from_open}
        hist_volwin=[],                            # each: {mark_minute: dollar_vol_in_window}
        cur_volwin={},                             # current session window dollar vols at marks
        last_close_px=None,
        # position
        pos=0, entry=None, qty=0.0, cat_stop=None, ext=None,
    ))
    # cache params
    st['risk_frac']    = _p(ctx, 'risk_frac', 0.0001)
    st['max_leverage'] = _p(ctx, 'max_leverage', 20.0)
    st['capital_usd']  = _p(ctx, 'capital_usd', 100000.0)
    st['cat_stop_bps'] = _p(ctx, 'cat_stop_bps', 400.0)
    st['anchor']       = int(_p(ctx, 'anchor_utc_min', 810))
    st['close_min']    = int(_p(ctx, 'close_utc_min', 1439))
    st['mark_int']     = int(_p(ctx, 'mark_interval_min', 30))
    st['first_mark']   = int(_p(ctx, 'first_mark_min', 840))
    st['lookback']     = int(_p(ctx, 'noise_lookback', 14))
    st['min_obs']      = int(_p(ctx, 'min_obs', 10))
    st['band_k']       = _p(ctx, 'band_k', 1.0)
    st['vol_k']        = _p(ctx, 'vol_conf_k', 2.0)
    st['vol_win']      = int(_p(ctx, 'vol_window_min', 30))
    st['stop_style']   = _ps(ctx, 'stop_style', 'band')
    # marks (minutes past midnight UTC)
    st['marks'] = list(range(st['first_mark'], st['close_min'] + 1, st['mark_int']))
    # news blackout set
    raw = _ps(ctx, 'news_dates', '')
    st['news'] = set(x.strip() for x in raw.replace(';', ',').split(',') if x.strip())


def _sigma_at(st, mos):
    """Mean |move| at this minute-of-session over prior sessions (>=min_obs)."""
    vals = [h[mos] for h in st['hist_absmove'][-st['lookback']:] if mos in h]
    if len(vals) < st['min_obs']:
        return None
    return sum(vals) / len(vals)


def _volbase_at(st, mark):
    vals = [h[mark] for h in st['hist_volwin'][-st['lookback']:] if mark in h]
    if len(vals) < st['min_obs']:
        return None
    return sum(vals) / len(vals)


def _roll_session(st):
    """At session end: push current session's profiles into history, reset."""
    if st['sess_absmove']:
        st['hist_absmove'].append(dict(st['sess_absmove']))
    if st['cur_volwin']:
        st['hist_volwin'].append(dict(st['cur_volwin']))
    if st['last_close_px'] is not None:
        st['prev_close'] = st['last_close_px']
    st['open_px'] = None
    st['cum_pv'] = 0.0
    st['cum_v'] = 0.0
    st['sess_absmove'] = {}
    st['cur_volwin'] = {}
    st['_dvol_ring'] = []


def _flatten(ctx, st, tag):
    if st['pos'] != 0 and st['qty'] > 0:
        ctx.place_order('sell' if st['pos'] > 0 else 'buy', 'market', st['qty'])
    st['pos'] = 0; st['entry'] = None; st['qty'] = 0.0
    st['cat_stop'] = None; st['ext'] = None


def on_bar(ctx: StrategyContext, bar):
    st = ctx.state
    o = _f(bar, 'o', 'open'); high = _f(bar, 'h', 'high')
    low = _f(bar, 'l', 'low'); close = _f(bar, 'c', 'close')
    vol = 0.0
    for nm in ('v', 'volume', 'baseVolume'):
        if nm in bar:
            vol = float(bar[nm]); break
    qv = None
    for nm in ('qv', 'quoteVolume', 'quote_volume'):
        if nm in bar:
            qv = float(bar[nm]); break
    dvol = qv if qv is not None else (high + low + close) / 3.0 * vol

    dt = _bar_dt(bar['ts'] if 'ts' in bar else bar.get('timestamp', bar.get('time')))
    mod = dt.hour * 60 + dt.minute
    dstr = dt.date().isoformat()

    # ---- session rollover: a new UTC date starts a new session at/after the anchor ----
    if st['cur_date'] is not None and dstr != st['cur_date']:
        # previous session ended; if still in a position it should have been closed at close_min,
        # but force-flat defensively at the first bar of a new day.
        if st['pos'] != 0:
            _flatten(ctx, st, 'newday')
        _roll_session(st)
    st['cur_date'] = dstr

    # ---- anchor: set session open at the first bar at/after the anchor minute ----
    if st['open_px'] is None and mod >= st['anchor']:
        st['open_px'] = o                          # session open price
    if st['open_px'] is None:
        st['last_close_px'] = close
        return                                     # pre-session bars: nothing to do

    # ---- running VWAP + record abs-move profile at this minute-of-session ----
    st['cum_pv'] += (high + low + close) / 3.0 * vol
    st['cum_v'] += vol
    mos = mod                                       # minute-of-day is our session key
    st['sess_absmove'][mos] = abs(close / st['open_px'] - 1.0)
    st['last_close_px'] = close

    # accumulate the trailing volume window ending at this minute (simple: sum last vol_win bars
    # is approximated by tracking per-mark below; here we store the window dollar-vol at marks)

    op = st['open_px']
    pc = st['prev_close'] if st['prev_close'] is not None else op
    hi_a, lo_a = max(op, pc), min(op, pc)
    vwap = (st['cum_pv'] / st['cum_v']) if st['cum_v'] > 0 else close

    # ---- true trailing volume window: accumulate EVERY minute (not just at marks) ----
    ring = st.setdefault('_dvol_ring', [])
    ring.append((mod, dvol))
    win_lo = mod - st['vol_win']
    while ring and ring[0][0] <= win_lo:
        ring.pop(0)
    win_dvol = sum(v for _, v in ring)

    # ---- continuous intrabar catastrophic stop (checked EVERY minute) ----
    if st['pos'] != 0 and st['cat_stop'] is not None:
        if (st['pos'] > 0 and low <= st['cat_stop']) or (st['pos'] < 0 and high >= st['cat_stop']):
            _flatten(ctx, st, 'catstop')
            return

    # ---- forced flat at/after close minute ----
    if mod >= st['close_min']:
        if st['pos'] != 0:
            _flatten(ctx, st, 'eod')
        return

    # ---- decision only on marks ----
    if mod not in st['marks']:
        return

    # record this mark's trailing-window dollar-vol for the causal baseline
    st['cur_volwin'][mod] = win_dvol

    sig = _sigma_at(st, mos)
    has_band = sig is not None
    ub = hi_a * (1 + st['band_k'] * sig) if has_band else None
    lb = lo_a * (1 - st['band_k'] * sig) if has_band else None

    # volume confirmation gate for ENTRIES
    ok_entry = True
    if st['vol_k'] > 0:
        base = _volbase_at(st, mod)
        ok_entry = (base is not None) and (win_dvol >= st['vol_k'] * base)
    # news blackout
    if dstr in st['news']:
        ok_entry = False

    # ---- manage open position (stops / flips) ----
    if st['pos'] != 0 and has_band:
        # dynamic stop
        if st['stop_style'] == 'vwap':
            if st['pos'] > 0 and close < max(vwap, lb):
                _flatten(ctx, st, 'vwapstop'); return
            if st['pos'] < 0 and close > min(vwap, ub):
                _flatten(ctx, st, 'vwapstop'); return
        # opposite-band flip
        if st['pos'] > 0 and close < lb:
            _flatten(ctx, st, 'flip')
            if ok_entry:
                _enter(ctx, st, -1, close)
            return
        if st['pos'] < 0 and close > ub:
            _flatten(ctx, st, 'flip')
            if ok_entry:
                _enter(ctx, st, 1, close)
            return

    # ---- entry when flat ----
    if st['pos'] == 0 and has_band and ok_entry:
        if close > ub:
            _enter(ctx, st, 1, close)
        elif close < lb:
            _enter(ctx, st, -1, close)


def _enter(ctx, st, side, price):
    stop_dist = st['cat_stop_bps'] / 1e4
    notional = min(st['capital_usd'] * st['risk_frac'] / stop_dist,
                   st['max_leverage'] * st['capital_usd'])
    qty = round(notional / price, 3)
    if qty <= 0:
        return
    ctx.place_order('buy' if side > 0 else 'sell', 'market', qty)
    st['pos'] = side; st['entry'] = price; st['qty'] = qty; st['ext'] = price
    st['cat_stop'] = price * (1 - stop_dist) if side > 0 else price * (1 + stop_dist)


def on_stop(ctx: StrategyContext):
    pass
