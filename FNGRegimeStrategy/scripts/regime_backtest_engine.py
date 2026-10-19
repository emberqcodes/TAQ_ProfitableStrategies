#!/usr/bin/env python3
"""
regime_backtest_engine.py
=========================
Internal backtest engine for the Directional Crypto Regime Strategy family.

TWO-FILE ENGINE (script + JSON manifest) with one hard rule:
  * The script holds MECHANICS only -- no strategy constant is hardcoded.
  * The manifest holds the STRATEGY, INCLUDING the decided regime itself.

WHY THE REGIME LIVES IN THE MANIFEST
  The runtime engine cannot read files from external paths. The daily regime
  (LONG / FLAT / SHORT per asset per day) is therefore carried INSIDE the
  manifest as data -- `regime.states` -- not recomputed from a signal file at
  run time. The recipe that produced those states is recorded in
  `regime.provenance` so they stay auditable and regenerable.

RUNTIME (what the harness calls) -- reads ONLY the manifest, no file paths:
    import regime_backtest_engine as E
    result = E.run_backtest(manifest_dict, ohlcv_by_asset, funding_by_asset=None)
  where ohlcv_by_asset = {"btc": df, "eth": df} are bar DataFrames the harness
  supplies in memory (columns open/high/low/close[/volume], UTC DatetimeIndex).
  Pipeline: embedded states -> shift to the entry lag (T+N) -> broadcast to bars
            -> intrabar execution -> reconcile ledger -> performance/trades/equity.

OFFLINE (run where the daily signals ARE reachable) -- generates the states:
    python regime_backtest_engine.py --build-regime --manifest m.json --panel daily_panel.csv --out m.json
  computes the regime from the panel using the manifest's `regime.provenance`
  recipe and writes the result into `regime.states`.

OTHER MODES:
    python regime_backtest_engine.py --manifest m.json --data-dir data/   # local file-based run (testing)
    python regime_backtest_engine.py --selftest                            # mechanics, no files
    python regime_backtest_engine.py --emit-manifest m.json                # write the locked template
"""

import argparse
import datetime as _dt
import json
import os
import sys
import numpy as np
import pandas as pd

LONG, FLAT, SHORT = 1, 0, -1
BARS_PER_YEAR = {"5m": 105120, "15m": 35040, "30m": 17520, "1h": 8760, "2h": 4380, "4h": 2190, "1d": 365}
_FREQ = {"5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "2h": "2h", "4h": "4h", "1d": "1D"}


# ======================================================================================
# MANIFEST  (load + strict validation)
# ======================================================================================
def load_manifest(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _walk(d, dotted):
    cur = d
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None, False
        cur = cur[key]
    return cur, True


def validate_manifest(m, *, require_states=True, require_provenance=False):
    """Return a list of problems (empty == valid). `require_states` for a RUN;
    `require_provenance` for --build-regime (the recipe that generates the states)."""
    problems = []

    def need(path, types, *, rng=None, choices=None, nonempty=False):
        val, ok = _walk(m, path)
        if not ok:
            problems.append(f"missing required field: '{path}'")
            return None
        if not isinstance(val, types):
            tn = types.__name__ if isinstance(types, type) else "/".join(t.__name__ for t in types)
            problems.append(f"'{path}' must be {tn}, got {type(val).__name__}")
            return None
        if rng is not None and isinstance(val, (int, float)) and not isinstance(val, bool):
            if not (rng[0] <= val <= rng[1]):
                problems.append(f"'{path}' = {val} out of range [{rng[0]}, {rng[1]}]")
        if choices is not None and val not in choices:
            problems.append(f"'{path}' = {val!r} not one of {choices}")
        if nonempty and hasattr(val, "__len__") and len(val) == 0:
            problems.append(f"'{path}' must be non-empty")
        return val

    need("strategy.name", str)
    assets = need("strategy.assets", list, nonempty=True) or []
    need("data.timeframe", str, choices=list(BARS_PER_YEAR.keys()))

    # ---- the regime, embedded as decided states ----
    if require_states:
        st, ok = _walk(m, "regime.states")
        if not ok:
            problems.append("missing required field: 'regime.states' (the engine cannot read an "
                            "external regime file; embed the decided states in the manifest)")
        elif not isinstance(st, dict):
            problems.append("'regime.states' must be an object keyed by asset")
        else:
            for a in assets:
                if a not in st:
                    problems.append(f"'regime.states' has no entry for asset '{a}'")
                    continue
                seq = st[a]
                if not isinstance(seq, list) or len(seq) == 0:
                    problems.append(f"'regime.states.{a}' must be a non-empty list of [date, state] pairs")
                    continue
                bad = [p for p in seq[:5000]
                       if not (isinstance(p, list) and len(p) == 2 and isinstance(p[0], str)
                               and isinstance(p[1], int) and p[1] in (-1, 0, 1))]
                if bad:
                    problems.append(f"'regime.states.{a}' has malformed pairs (need [\"YYYY-MM-DD\", -1|0|1]); "
                                    f"first bad: {bad[0]!r}")

    # ---- provenance (recipe) ----
    if require_provenance:
        need("regime.provenance.logic", str, choices=["option_a"])
        need("regime.provenance.signals.trend.window_days", int, rng=(2, 365))
        need("regime.provenance.signals.supply.roc_days", int, rng=(1, 365))
        need("regime.provenance.percentile.window_days", int, rng=(30, 2000))
        need("regime.provenance.percentile.ffill_limit", int, rng=(0, 30))
        for c in ("price_col", "supply_col", "funding_col", "fng_col"):
            need(f"regime.provenance.panel_columns.{c}", str)
        need("regime.provenance.decision.supply_split_pctile", (int, float), rng=(0, 100))
        need("regime.provenance.decision.fear_extreme_pctile", (int, float), rng=(0, 100))
        need("regime.provenance.decision.greed_extreme_pctile", (int, float), rng=(0, 100))
        need("regime.provenance.decision.brake_requires_both", bool)
        need("regime.provenance.decision.long_brake_enabled", bool)

    # ---- execution ----
    need("execution.entry.type", str, choices=["immediate"])
    need("execution.entry_lag_days", int, rng=(1, 10))
    need("execution.stop.trailing.enabled", bool)
    need("execution.stop.trailing.atr_period", int, rng=(2, 200))
    need("execution.stop.trailing.atr_mult", (int, float), rng=(0.1, 100))
    need("execution.stop.hard.enabled", bool)
    need("execution.stop.take_profit.enabled", bool)
    need("execution.regime_flip_exit", bool)
    need("execution.reentry_lockout.enabled", bool)

    # ---- sizing ----
    method = need("sizing.method", str, choices=["risk_based", "fixed_notional"])
    need("sizing.capital_usd", (int, float), rng=(1, 1e12))
    need("sizing.max_leverage", (int, float), rng=(0.01, 1000))
    need("sizing.compounding", bool)
    if method == "risk_based":
        need("sizing.risk_usd", (int, float), rng=(0.01, 1e9))
    elif method == "fixed_notional":
        need("sizing.notional_usd", (int, float), rng=(0.01, 1e12))

    # ---- costs ----
    need("costs.taker_bps", (int, float), rng=(0, 1000))
    need("costs.slippage_bps", (int, float), rng=(0, 1000))
    need("costs.funding.enabled", bool)

    # ---- consistency ----
    if (_walk(m, "execution.stop.trailing.enabled")[0] is False
            and _walk(m, "execution.stop.hard.enabled")[0] is False
            and _walk(m, "execution.regime_flip_exit")[0] is False):
        problems.append("no exit enabled (trailing / hard / regime-flip all off) -- positions could never close")
    if require_provenance and _walk(m, "regime.provenance.logic")[0] == "option_a" \
            and _walk(m, "regime.provenance.decision.long_brake_enabled")[0] is True:
        problems.append("provenance.decision.long_brake_enabled=true contradicts Option A (must be off)")
    return problems


# ======================================================================================
# REGIME -- runtime: read embedded states ;  offline: generate them (validated logic)
# ======================================================================================
def regime_states_from_manifest(m, asset):
    """Embedded [date, state] pairs -> daily state Series (reindexed to a contiguous day grid)."""
    pairs = m["regime"]["states"][asset]
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d, _ in pairs])
    s = pd.Series([int(v) for _, v in pairs], index=idx, name="state").sort_index()
    full = pd.date_range(s.index.min(), s.index.max(), freq="D")
    if not s.index.equals(full):
        miss = len(full) - len(s)
        print(f"  (!) regime.states.{asset} not daily-contiguous ({miss} gap-day(s)); forward-filling.")
        s = s.reindex(full).ffill()
    return s


def trailing_pct_rank(s, window, min_valid=None):
    """Percentile (0-100) of today's value within the trailing `window` days. Look-ahead-free.
    NaN-tolerant: ranks against the finite values in the window, requiring >= min_valid of them."""
    if min_valid is None:
        min_valid = max(2, window - 30)

    def _rank(w):
        if len(w) < window:
            return np.nan
        cur = w[-1]
        if not np.isfinite(cur):
            return np.nan
        valid = w[np.isfinite(w)]
        if len(valid) < min_valid:
            return np.nan
        return float((valid <= cur).mean() * 100.0)

    return s.rolling(window, min_periods=1).apply(_rank, raw=True)


def breakout_state(close, n):
    """+1 on a new n-day high, -1 on a new n-day low, hold otherwise. Look-ahead-free."""
    prior_high = close.rolling(n, min_periods=n).max().shift(1)
    prior_low = close.rolling(n, min_periods=n).min().shift(1)
    st = pd.Series(np.nan, index=close.index)
    st[close > prior_high] = 1.0
    st[close < prior_low] = -1.0
    return st.ffill().fillna(0.0)


def decide(trend, supply_pct, funding_pct, fng_pct, dcfg):
    """One day's signals -> {LONG, FLAT, SHORT}. Option-A asymmetric brake. Thresholds from dcfg."""
    if any(pd.isna(x) for x in (trend, supply_pct, funding_pct, fng_pct)):
        return FLAT
    if trend == 0:
        return FLAT
    split = dcfg["supply_split_pctile"]; fear_lo = dcfg["fear_extreme_pctile"]
    greed_hi = dcfg["greed_extreme_pctile"]; require_both = dcfg["brake_requires_both"]
    long_brake = dcfg["long_brake_enabled"]
    supply_expanding = supply_pct > split
    f_hi, g_hi = funding_pct >= greed_hi, fng_pct >= greed_hi
    f_lo, g_lo = funding_pct <= fear_lo, fng_pct <= fear_lo
    greed_extreme = (f_hi and g_hi) if require_both else (f_hi or g_hi)
    fear_extreme = (f_lo and g_lo) if require_both else (f_lo or g_lo)
    if trend > 0:
        if not supply_expanding:
            return FLAT
        if long_brake and greed_extreme:
            return FLAT
        return LONG
    else:
        if supply_expanding:
            return FLAT
        if fear_extreme:
            return FLAT
        return SHORT


def build_states_from_panel(asset, panel, prov):
    """OFFLINE: daily panel + provenance recipe -> decided daily state Series (the validated logic)."""
    pc = prov["panel_columns"]
    price = panel[pc["price_col"].format(asset=asset)]
    supply = panel[pc["supply_col"].format(asset=asset)]
    funding = panel[pc["funding_col"].format(asset=asset)]
    fng = panel[pc["fng_col"].format(asset=asset)]
    win = prov["percentile"]["window_days"]; mv = prov["percentile"].get("min_valid")
    ffl = prov["percentile"]["ffill_limit"]
    if ffl > 0:
        supply = supply.ffill(limit=ffl); funding = funding.ffill(limit=ffl); fng = fng.ffill(limit=ffl)
    trend = breakout_state(price, prov["signals"]["trend"]["window_days"])
    supply_growth = supply.pct_change(prov["signals"]["supply"]["roc_days"]) * 100
    sig = pd.DataFrame({"trend": trend,
                        "supply_pct": trailing_pct_rank(supply_growth, win, mv),
                        "funding_pct": trailing_pct_rank(funding, win, mv),
                        "fng_pct": trailing_pct_rank(fng, win, mv)})
    dcfg = prov["decision"]
    states = [decide(r.trend, r.supply_pct, r.funding_pct, r.fng_pct, dcfg)
              for r in sig.itertuples(index=False)]
    return pd.Series(states, index=sig.index, name="state")


# ======================================================================================
# PERMISSION  (daily state -> per-bar, with the entry lag)
# ======================================================================================
def map_permission(daily_state, bar_index, lag_days):
    """A state decided at day D's close is actionable from D + lag_days. lag_days=2 leaves a day
    for data availability. Causal for lag >= 1 (a future state never reaches an earlier day)."""
    gov = daily_state.shift(lag_days)
    gov.index = pd.DatetimeIndex(gov.index).tz_localize(None).normalize()
    days = pd.DatetimeIndex(bar_index).tz_convert("UTC").tz_localize(None).normalize()
    perm = gov.reindex(pd.DatetimeIndex(days).unique()).ffill()
    return pd.Series(perm.reindex(days).fillna(0).values, index=bar_index, name="perm")


# ======================================================================================
# EXECUTION ENGINE  (intrabar; ported verbatim from the validated layer)
# ======================================================================================
def atr(high, low, close, n):
    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def run_execution(df, perm, a, *, trailing_on, trail, hard_on, sl_mult, tp_on, tp_mult,
                  sizing, risk, notional, max_leverage, capital, cpt, fund, reentry_lockout):
    """Immediate entry (market, fills next bar open); trailing/hard ATR stop; ATR take-profit;
    regime-flip exit always on; optional after-stop re-entry lockout. Returns (trades, net$, pos)."""
    o = df["open"].values; h = df["high"].values
    lo = df["low"].values; c = df["close"].values
    perm = np.asarray(perm, float); a = np.asarray(a, float)
    n = len(df)
    fund = np.zeros(n) if fund is None else np.asarray(fund, float)
    day_num = df.index.normalize().asi8 if reentry_lockout else None
    _active = ([sl_mult] if hard_on else []) + ([trail] if trailing_on else [])
    risk_mult = min(_active) if _active else sl_mult

    cash = 0.0; equity_mark = np.full(n, np.nan); pos_arr = np.zeros(n)
    pos = 0; entry_px = entry_i = None; ext = stop = sl = tp = None
    cur_notional = float(notional); working = None; lock_day = -1; trades = []

    def book(ei, xi, d, epx, xpx, reason, notion):
        # funding accrues on bars HELD after entry (ei+1..xi); the entry bar fills at its open and
        # pays no funding -- this matches the loop, so the ledger reconciles to the stream exactly.
        fseg = float(d * notion * fund[ei + 1:xi + 1].sum())
        gross = float(d * notion * (xpx / epx - 1.0))
        cost = 2.0 * notion * cpt
        net = gross - cost - fseg
        trades.append(dict(dir=int(d), entry_i=ei, exit_i=xi,
                           entry_time=df.index[ei], exit_time=df.index[xi],
                           entry_px=round(epx, 2), exit_px=round(xpx, 2),
                           hold_bars=int(xi - ei), exit_reason=reason,
                           notional_usd=round(notion, 2), leverage=round(notion / capital, 2),
                           gross_usd=round(gross, 2), funding_usd=round(fseg, 2),
                           cost_usd=round(cost, 2), net_usd=round(net, 2), net=net))

    def enter(d, fill_px, i):
        nonlocal pos, entry_px, entry_i, ext, stop, sl, tp, cash, cur_notional
        pos = d; entry_px = fill_px; entry_i = i
        if sizing == "risk_based" and np.isfinite(a[i]) and a[i] > 0 and risk_mult > 0:
            cur_notional = min(risk * fill_px / (risk_mult * a[i]), max_leverage * capital)
        else:
            cur_notional = float(notional)
        cash -= cur_notional * cpt
        ext = fill_px
        stop = (ext - trail * a[i]) if d > 0 else (ext + trail * a[i])
        sl = (entry_px - sl_mult * a[i]) if d > 0 else (entry_px + sl_mult * a[i])
        tp = (entry_px + tp_mult * a[i]) if d > 0 else (entry_px - tp_mult * a[i])

    for i in range(n):
        if pos == 0 and working is not None:                        # (A) fill armed market order at this open
            d = working["dir"]
            enter(d, o[i], i); working = None
            pos_arr[i] = pos
            equity_mark[i] = capital + cash + pos * cur_notional * (c[i] / entry_px - 1.0)
            continue
        if pos != 0:                                                # (B) manage open position
            cash -= pos * cur_notional * fund[i]
            exit_px = None; reason = None
            cands = ([stop] if trailing_on else []) + ([sl] if hard_on else [])
            if pos > 0:
                eff = max(cands) if cands else -np.inf
                if eff > -np.inf and lo[i] <= eff:
                    exit_px = o[i] if o[i] <= eff else eff; reason = "stop"
                elif tp_on and h[i] >= tp:
                    exit_px = o[i] if o[i] >= tp else tp; reason = "take_profit"
            else:
                eff = min(cands) if cands else np.inf
                if eff < np.inf and h[i] >= eff:
                    exit_px = o[i] if o[i] >= eff else eff; reason = "stop"
                elif tp_on and lo[i] <= tp:
                    exit_px = o[i] if o[i] <= tp else tp; reason = "take_profit"
            if exit_px is None and perm[i] != pos:
                exit_px = o[i]; reason = "regime_flip"
            if exit_px is not None:
                cash += pos * cur_notional * (exit_px / entry_px - 1.0) - cur_notional * cpt
                book(entry_i, i, pos, entry_px, exit_px, reason, cur_notional)
                if reentry_lockout and reason in ("stop", "take_profit"):
                    lock_day = day_num[i]
                pos = 0; entry_px = entry_i = ext = stop = None
                pos_arr[i] = 0; equity_mark[i] = capital + cash
            else:
                if trailing_on:
                    if pos > 0:
                        ext = max(ext, h[i]); stop = max(stop, ext - trail * a[i])
                    else:
                        ext = min(ext, lo[i]); stop = min(stop, ext + trail * a[i])
                pos_arr[i] = pos
                equity_mark[i] = capital + cash + pos * cur_notional * (c[i] / entry_px - 1.0)
                continue
        if pos == 0:                                                # (C) flat -> arm immediate order
            locked = reentry_lockout and day_num[i] <= lock_day
            if working is None and not locked:
                d = perm[i]
                if d != 0 and np.isfinite(a[i]):
                    working = {"dir": int(d), "kind": "market"}
            equity_mark[i] = capital + cash

    if pos != 0:
        i = n - 1; exit_px = c[i]
        cash += pos * cur_notional * (exit_px / entry_px - 1.0) - cur_notional * cpt
        book(entry_i, i, pos, entry_px, exit_px, "data_end", cur_notional)
        equity_mark[i] = capital + cash; pos = 0

    eq = pd.Series(equity_mark, index=df.index).ffill().fillna(capital)
    bar_net = eq.diff().fillna(eq.iloc[0] - capital)
    return pd.DataFrame(trades), bar_net.rename("net"), pd.Series(pos_arr, index=df.index, name="pos")


def buy_hold(df, capital, cpt, fund):
    o = df["open"].values; c = df["close"].values; n = len(df)
    notion = float(capital); entry = o[0]
    fund_cum = np.cumsum(notion * (np.zeros(n) if fund is None else np.asarray(fund, float)))
    eq = capital + notion * (c / entry - 1.0) - fund_cum - notion * cpt
    eq = eq.copy(); eq[-1] -= notion * cpt
    s = pd.Series(eq, index=df.index)
    return s.diff().fillna(s.iloc[0] - capital).rename("net")


# ======================================================================================
# ACCOUNTING / STATS
# ======================================================================================
def perf(net, capital, bpy):
    r = net.dropna()
    if len(r) == 0:
        return dict(pnl=0.0, ret=0.0, sharpe=np.nan, maxdd_usd=0.0, maxdd_pct=0.0, n=0)
    eq = capital + r.cumsum(); peak = eq.cummax(); sd = r.std()
    return dict(pnl=float(r.sum()), ret=float(r.sum() / capital),
                sharpe=float(r.mean() / sd * np.sqrt(bpy)) if sd and sd > 0 else np.nan,
                maxdd_usd=float((eq - peak).min()), maxdd_pct=float((eq / peak - 1).min()), n=len(r))


def stat_row(scope, net, trades, pos, capital, bpy):
    s = perf(net, capital, bpy)
    n_tr = len(trades)
    wins = float((trades["net"] > 0).mean() * 100) if n_tr else np.nan
    hold = float(trades["hold_bars"].mean()) if n_tr else np.nan
    rc = trades["exit_reason"].value_counts().to_dict() if n_tr else {}
    return dict(scope=scope, n_trades=n_tr,
                net_usd=round(s["pnl"], 2), ret_on_capital_pct=round(s["ret"] * 100, 2),
                final_equity_usd=round(capital + s["pnl"], 2),
                sharpe=(round(s["sharpe"], 3) if np.isfinite(s["sharpe"]) else ""),
                maxdd_pct=round(s["maxdd_pct"] * 100, 2), maxdd_usd=round(s["maxdd_usd"], 2),
                win_pct=(round(wins, 1) if np.isfinite(wins) else ""),
                avg_hold_bars=(round(hold, 1) if np.isfinite(hold) else ""),
                stops=int(rc.get("stop", 0)), flips=int(rc.get("regime_flip", 0)),
                take_profits=int(rc.get("take_profit", 0)),
                fees_usd=round(float(trades["cost_usd"].sum()) if n_tr else 0.0, 2),
                time_in_mkt_pct=round(float((pos != 0).mean()) * 100, 1) if len(pos) else 0.0)


_EMPTY_TR = pd.DataFrame(columns=["net", "hold_bars", "exit_reason", "cost_usd"])


# ======================================================================================
# RUNTIME  (no file reads -- OHLCV/funding supplied in memory by the caller)
# ======================================================================================
def run_one_asset(m, asset, df, fund=None):
    e = m["execution"]; szc = m["sizing"]; cc = m["costs"]
    cpt = (cc["taker_bps"] + cc["slippage_bps"]) / 1e4
    states = regime_states_from_manifest(m, asset)
    a = atr(df["high"], df["low"], df["close"], e["stop"]["trailing"]["atr_period"]).values
    perm = map_permission(states, df.index, e["entry_lag_days"]).values
    trades, net, pos = run_execution(
        df, perm, a,
        trailing_on=e["stop"]["trailing"]["enabled"], trail=e["stop"]["trailing"]["atr_mult"],
        hard_on=e["stop"]["hard"]["enabled"], sl_mult=e["stop"]["hard"].get("atr_mult", 2.0),
        tp_on=e["stop"]["take_profit"]["enabled"], tp_mult=e["stop"]["take_profit"].get("atr_mult", 4.0),
        sizing=szc["method"], risk=szc.get("risk_usd", 0.0), notional=szc.get("notional_usd", 0.0),
        max_leverage=szc["max_leverage"], capital=szc["capital_usd"], cpt=cpt,
        fund=fund, reentry_lockout=e["reentry_lockout"]["enabled"])
    bh = buy_hold(df, szc["capital_usd"], cpt, fund)
    return dict(asset=asset, trades=trades, net=net, pos=pos, bh=bh)


def run_backtest(m, ohlcv_by_asset, funding_by_asset=None, outdir=None, verbose=True):
    """HARNESS ENTRY POINT. ohlcv_by_asset = {asset: DataFrame(open,high,low,close[,volume])} with a
    UTC DatetimeIndex. funding_by_asset optional {asset: per-bar funding array/Series}. Reads no files.
    Returns dict(performance=DataFrame, trades=DataFrame, equity=DataFrame, reconciled=bool)."""
    problems = validate_manifest(m, require_states=True)
    if problems:
        raise ValueError("invalid manifest:\n  - " + "\n  - ".join(problems))
    tf = m["data"]["timeframe"]; bpy = BARS_PER_YEAR[tf]; cap = m["sizing"]["capital_usd"]
    funding_by_asset = funding_by_asset or {}

    rows, eq_cols, all_trades = [], {}, []
    port_net = port_bh = port_pos = None
    reconciled = True
    for a in m["strategy"]["assets"]:
        if a not in ohlcv_by_asset:
            raise KeyError(f"no OHLCV supplied for asset '{a}'")
        df = ohlcv_by_asset[a]
        if df.index.tz is None:
            df = df.copy(); df.index = df.index.tz_localize("UTC")
        r = run_one_asset(m, a, df, fund=funding_by_asset.get(a))
        if len(r["trades"]) and abs(r["trades"]["net"].sum() - r["net"].sum()) > 1e-6:
            reconciled = False
            if verbose:
                print(f"  (!) reconciliation gap on {a}: {abs(r['trades']['net'].sum() - r['net'].sum()):.6g}")
        rows.append(stat_row(f"{a.upper()} strategy", r["net"], r["trades"], r["pos"], cap, bpy))
        rows.append(stat_row(f"{a.upper()} buy & hold", r["bh"], _EMPTY_TR, pd.Series(dtype=float), cap, bpy))
        eq_cols[f"{a}_strategy"] = cap + r["net"].cumsum()
        eq_cols[f"{a}_buyhold"] = cap + r["bh"].cumsum()
        if len(r["trades"]):
            tr = r["trades"].copy(); tr["asset"] = a; all_trades.append(tr)
        port_net = r["net"] if port_net is None else port_net.add(r["net"], fill_value=0.0)
        port_bh = r["bh"] if port_bh is None else port_bh.add(r["bh"], fill_value=0.0)
        pf = (r["pos"] != 0).astype(float)
        port_pos = pf if port_pos is None else port_pos.add(pf, fill_value=0.0)

    port_trades = pd.concat(all_trades, ignore_index=True) if all_trades else _EMPTY_TR
    rows.append(stat_row("PORTFOLIO strategy", port_net, port_trades, (port_pos > 0).astype(float), cap, bpy))
    rows.append(stat_row("PORTFOLIO buy & hold", port_bh, _EMPTY_TR, pd.Series(dtype=float), cap, bpy))
    eq_cols["PORTFOLIO_strategy"] = cap + port_net.cumsum()
    eq_cols["PORTFOLIO_buyhold"] = cap + port_bh.cumsum()

    perf_df = pd.DataFrame(rows)
    equity_df = pd.DataFrame(eq_cols).sort_index()
    trades_df = port_trades if len(port_trades) else pd.DataFrame()

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        perf_df.to_csv(os.path.join(outdir, "engine_performance.csv"), index=False)
        trades_df.to_csv(os.path.join(outdir, "engine_trades.csv"), index=False)
        equity_df.to_csv(os.path.join(outdir, "engine_equity.csv"))
    if verbose:
        print("=" * 96)
        print(f"  {m['strategy']['name']}")
        print(f"  timeframe={tf}  entry lag=T+{m['execution']['entry_lag_days']}  "
              f"trail={m['execution']['stop']['trailing']['atr_mult']:g}xATR{m['execution']['stop']['trailing']['atr_period']}  "
              f"risk=${m['sizing'].get('risk_usd', 0):,.0f}  "
              f"cost={(m['costs']['taker_bps'] + m['costs']['slippage_bps']):g}bps/turn  "
              f"funding={'ON' if m['costs']['funding']['enabled'] else 'OFF'}")
        print("=" * 96)
        with pd.option_context("display.width", 200, "display.max_columns", 30):
            print(perf_df.to_string(index=False))
        print(f"\nLedger reconciles to per-bar stream: {'YES' if reconciled else 'NO -- see warnings'}")
        if outdir:
            print(f"Outputs -> {outdir}/engine_performance.csv, engine_trades.csv, engine_equity.csv")
    return dict(performance=perf_df, trades=trades_df, equity=equity_df, reconciled=reconciled)


# ======================================================================================
# LOCAL FILE LOADERS  (testing convenience only -- NOT used by the harness)
# ======================================================================================
def _load_ohlcv_file(m, asset, data_dir):
    rel = m["data"].get("ohlcv_path", "").format(asset=asset, timeframe=m["data"]["timeframe"])
    path = os.path.join(data_dir, os.path.basename(rel)) if data_dir else rel
    df = pd.read_parquet(path)
    df.index = pd.DatetimeIndex(df.index)
    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    cols = {c.lower(): c for c in df.columns}
    out = pd.DataFrame({k: df[cols[k]] for k in ("open", "high", "low", "close") if k in cols})
    if "volume" in cols:
        out["volume"] = df[cols["volume"]]
    return out.sort_index()


def _load_funding_file(m, asset, index, data_dir):
    fc = m["costs"]["funding"]
    if not fc.get("enabled", False):
        return None
    rel = fc.get("path", "").format(asset=asset, timeframe=m["data"]["timeframe"])
    path = os.path.join(data_dir, os.path.basename(rel)) if data_dir else rel
    if not rel or not os.path.exists(path):
        print(f"  (!) funding enabled but source missing for {asset}; using zero funding.")
        return None
    f = pd.read_parquet(path)[fc.get("rate_col", "funding_rate")]
    f.index = pd.DatetimeIndex(f.index)
    f.index = (f.index.tz_localize("UTC") if f.index.tz is None else f.index.tz_convert("UTC")).floor(_FREQ[m["data"]["timeframe"]])
    f = f[~f.index.duplicated(keep="last")]
    return f.reindex(index).fillna(0.0).values


# ======================================================================================
# OFFLINE  --build-regime  (turns a daily panel into the embedded states block)
# ======================================================================================
def build_regime(m, panel):
    prov = m["regime"]["provenance"]
    out = {}
    for a in m["strategy"]["assets"]:
        s = build_states_from_panel(a, panel, prov).dropna().astype(int)
        out[a] = [[d.strftime("%Y-%m-%d"), int(v)] for d, v in s.items()]
    return out


def import_states_from_csvs(m, csv_map, state_col="state"):
    """Embed ALREADY-DECIDED states straight from per-asset CSVs (a `state` column of -1/0/1 indexed
    by date) into regime.states. Use this when the regime has been decided elsewhere and you just
    need it inside the manifest -- no recompute, the uploaded states are authoritative."""
    m.setdefault("regime", {}).setdefault("states", {})
    summary = {}
    for asset, path in csv_map.items():
        df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
        if state_col not in df.columns:
            raise KeyError(f"{path}: no '{state_col}' column (has {list(df.columns)})")
        s = df[state_col].round().astype(int)
        bad = sorted(set(s.unique()) - {-1, 0, 1})
        if bad:
            raise ValueError(f"{path}: state column has values outside -1/0/1: {bad}")
        full = pd.date_range(s.index.min(), s.index.max(), freq="D")
        gaps = len(full) - len(s.index.unique())
        m["regime"]["states"][asset] = [[d.strftime("%Y-%m-%d"), int(v)] for d, v in s.items()]
        summary[asset] = dict(n=len(s), start=str(s.index.min().date()), end=str(s.index.max().date()),
                              gap_days=gaps, longs=int((s == 1).sum()), flats=int((s == 0).sum()),
                              shorts=int((s == -1).sum()))
    return summary


# ======================================================================================
# SELF-TEST
# ======================================================================================
def _synth_bars(seed, n=600):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="h", tz="UTC")
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    o = price; c = price * (1 + rng.normal(0, 0.001, n))
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.001, n)))
    loo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.001, n)))
    return pd.DataFrame({"open": o, "high": hi, "low": loo, "close": c, "volume": rng.uniform(1, 5, n)}, index=idx)


def _selftest():
    fails = []
    dcfg = dict(supply_split_pctile=50, fear_extreme_pctile=20, greed_extreme_pctile=95,
                brake_requires_both=True, long_brake_enabled=False)

    # (1) decide(): Option-A truth table
    NEU, FEAR, GREED, EXP, CON = 50, 10, 97, 70, 30
    for args, exp in [((1, EXP, NEU, NEU), LONG), ((1, EXP, GREED, GREED), LONG), ((1, CON, NEU, NEU), FLAT),
                      ((-1, CON, NEU, NEU), SHORT), ((-1, CON, FEAR, FEAR), FLAT), ((-1, EXP, NEU, NEU), FLAT),
                      ((0, EXP, NEU, NEU), FLAT), ((1, EXP, np.nan, NEU), FLAT)]:
        if decide(*args, dcfg) != exp:
            fails.append(f"decide{args} -> {decide(*args, dcfg)}, expected {exp}")

    # (2) trailing_pct_rank: look-ahead-free + NaN tolerance
    s = pd.Series(np.arange(400.0)); base = trailing_pct_rank(s, 365)
    s2 = s.copy(); s2.iloc[380] = 1e9
    if not base.iloc[:380].equals(trailing_pct_rank(s2, 365).iloc[:380]):
        fails.append("trailing_pct_rank: future value changed a past percentile")
    s3 = s.copy(); s3.iloc[200] = np.nan
    if trailing_pct_rank(s3, 365).iloc[370:].isna().all():
        fails.append("trailing_pct_rank: one NaN blanked the tail")

    # (3) breakout_state causal
    if breakout_state(pd.Series(np.arange(100.0)), 30).iloc[40] != 1.0:
        fails.append("breakout_state: rising series should be +1 after warmup")

    # (4) map_permission lag + causality
    sti = pd.date_range("2024-07-01", periods=6, freq="D")
    sst = pd.Series([1.0, -1.0, 1.0, 0.0, -1.0, 1.0], index=sti)
    hb = pd.date_range("2024-07-01", periods=6 * 24, freq="h", tz="UTC")
    dn = pd.DatetimeIndex(hb).tz_convert("UTC").tz_localize(None).normalize()
    dp = lambda p, day: sorted(np.unique(p.values[dn == pd.Timestamp(day)]).tolist())
    p1 = map_permission(sst, hb, 1); p2 = map_permission(sst, hb, 2)
    if not (dp(p1, "2024-07-03") == [-1.0] and dp(p2, "2024-07-03") == [1.0] and dp(p2, "2024-07-02") == [0.0]):
        fails.append("map_permission: entry-lag mapping wrong")
    sc = sst.copy(); sc.iloc[5] = 7.0
    if dp(map_permission(sc, hb, 2), "2024-07-03") != [1.0]:
        fails.append("map_permission: not causal")

    # (5) execution causality + (6) reconciliation (funding ON) + (7) sizing
    df = _synth_bars(1)
    perm = np.where(np.arange(len(df)) % 120 < 60, 1.0, -1.0)
    a = atr(df["high"], df["low"], df["close"], 14).values
    fundv = np.full(len(df), 1e-5)
    kw = dict(trailing_on=True, trail=15.0, hard_on=False, sl_mult=2.0, tp_on=False, tp_mult=4.0,
              sizing="risk_based", risk=2000.0, notional=0.0, max_leverage=25.0, capital=30000.0,
              cpt=0.001, fund=fundv, reentry_lockout=True)
    tr_a, net_a, _ = run_execution(df, perm, a, **kw)
    df2 = df.copy(); df2.iloc[-1, df2.columns.get_loc("close")] *= 1.5
    a2 = atr(df2["high"], df2["low"], df2["close"], 14).values
    tr_b, _, _ = run_execution(df2, perm, a2, **kw)
    k = min(len(tr_a), len(tr_b)) - 1
    if k > 0 and not np.allclose(tr_a["net_usd"].values[:k], tr_b["net_usd"].values[:k]):
        fails.append("run_execution: corrupting the last bar changed earlier trades")
    if len(tr_a) and abs(tr_a["net"].sum() - net_a.sum()) > 1e-6:
        fails.append(f"run_execution: ledger does not reconcile (gap {abs(tr_a['net'].sum() - net_a.sum()):.3g})")
    if len(tr_a):
        st = tr_a[(tr_a.exit_reason == "stop") & (tr_a.hold_bars <= 2)]
        if len(st) and not (0.5 * 2000 <= abs(st.iloc[0]["gross_usd"]) <= 1.8 * 2000):
            fails.append("sizing: initial-stop loss not near the $2000 risk budget")

    # (8) buy_hold closed-form
    bh = buy_hold(df, 30000.0, 0.0, None)
    if abs(bh.sum() - 30000.0 * (df["close"].iloc[-1] / df["open"].iloc[0] - 1.0)) > 1e-6:
        fails.append("buy_hold: closed-form mismatch")

    # (9) embedded-states round-trip: states in manifest -> Series -> permission
    ex = _example_manifest()
    if validate_manifest(ex, require_states=True):
        fails.append(f"validate rejected the example manifest: {validate_manifest(ex)}")
    ser = regime_states_from_manifest(ex, "btc")
    pairs = ex["regime"]["states"]["btc"]
    if int(ser.loc[pd.Timestamp(pairs[1][0])]) != pairs[1][1]:
        fails.append("regime_states_from_manifest: state value did not round-trip")

    # (10) build-regime offline reproduces the locked logic, and validation catches a broken manifest
    days = pd.date_range("2021-01-01", periods=500, freq="D")
    rng = np.random.default_rng(3)
    panel = pd.DataFrame({"btc_close": 30000 * np.exp(np.cumsum(rng.normal(0, 0.03, 500))),
                          "eth_close": 2000 * np.exp(np.cumsum(rng.normal(0, 0.04, 500))),
                          "supply_total": 8e10 * np.exp(np.cumsum(rng.normal(0, 0.004, 500))),
                          "funding_btc": rng.normal(0.0001, 0.0003, 500),
                          "funding_eth": rng.normal(0.0001, 0.0004, 500),
                          "fng": np.clip(rng.normal(50, 18, 500), 1, 99)}, index=days)
    prov = ex["regime"]["provenance"]
    direct = build_states_from_panel("btc", panel, prov).dropna().astype(int)
    mb = json.loads(json.dumps(ex)); mb["strategy"]["assets"] = ["btc", "eth"]
    built = build_regime(mb, panel)
    if [v for _, v in built["btc"]] != list(direct.values):
        fails.append("build_regime: did not match build_states_from_panel")
    bad = json.loads(json.dumps(ex)); bad["execution"]["entry_lag_days"] = 0
    del bad["regime"]["states"]["eth"]
    if len(validate_manifest(bad, require_states=True)) < 2:
        fails.append("validate_manifest missed problems in a broken manifest")

    print(f"engine self-test: {10 - len(fails)}/10 checks passed")
    for f in fails:
        print("  FAIL:", f)
    return 1 if fails else 0


def _example_manifest():
    """Locked configuration template. `regime.states` holds a SHORT illustrative stub; real states
    are filled by --build-regime from a daily panel. `regime.provenance` is the generation recipe."""
    stub = {"btc": [["2020-10-08", -1], ["2020-10-09", 0], ["2020-11-13", 1], ["2020-11-20", 1], ["2020-12-01", 0]],
            "eth": [["2020-10-08", -1], ["2020-10-09", 0], ["2020-11-13", 1], ["2020-11-20", 1], ["2020-12-01", 0]]}
    return {
        "manifest_version": "1.1",
        "strategy": {"name": "Directional Crypto Regime Strategy (Option A)",
                     "codename": "Ember", "assets": ["btc", "eth"]},
        "data": {"timeframe": "1h",
                 "ohlcv_path": "data/binance_{asset}_perp_{timeframe}_ohlcv.parquet"},
        "regime": {
            "states": stub,
            "provenance": {
                "logic": "option_a",
                "note": "Daily LONG/FLAT/SHORT per asset. Generated offline by --build-regime from a "
                        "daily panel; embedded here because the engine cannot read external files.",
                "signals": {"trend": {"method": "breakout", "window_days": 30},
                            "supply": {"method": "roc_percentile", "roc_days": 28},
                            "funding": {"method": "percentile"},
                            "fng": {"method": "percentile"}},
                "percentile": {"window_days": 365, "min_valid": 335, "ffill_limit": 3},
                "decision": {"supply_split_pctile": 50, "fear_extreme_pctile": 20,
                             "greed_extreme_pctile": 95, "brake_requires_both": True,
                             "long_brake_enabled": False},
                "panel_columns": {"price_col": "{asset}_close", "supply_col": "supply_total",
                                  "funding_col": "funding_{asset}", "fng_col": "fng"},
                "basket": ["usdt_ethereum", "usdt_tron", "usdc_ethereum"]}},
        "execution": {
            "entry": {"type": "immediate"}, "entry_lag_days": 2, "fill": "next_bar_open",
            "stop": {"trailing": {"enabled": True, "atr_period": 14, "atr_mult": 15.0},
                     "hard": {"enabled": False, "atr_mult": 2.0},
                     "take_profit": {"enabled": False, "atr_mult": 4.0}},
            "regime_flip_exit": True, "reentry_lockout": {"enabled": True, "unit": "calendar_day"}},
        "sizing": {"method": "risk_based", "risk_usd": 2000, "capital_usd": 30000,
                   "max_leverage": 25, "compounding": False},
        "costs": {"taker_bps": 5, "slippage_bps": 5,
                  "funding": {"enabled": True, "path": "data/binance_{asset}_perp_funding.parquet",
                              "rate_col": "funding_rate"}},
    }


def main(argv):
    ap = argparse.ArgumentParser(description="Manifest-driven regime backtest engine (regime embedded in manifest).")
    ap.add_argument("--manifest", help="path to the strategy manifest JSON")
    ap.add_argument("--data-dir", help="local folder with OHLCV/funding parquet (standalone testing only)")
    ap.add_argument("--outdir", default="data/engineoutputs", help="where to write results")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--emit-manifest", metavar="PATH", help="write the locked example manifest and exit")
    ap.add_argument("--build-regime", action="store_true", help="compute regime.states from --panel and write the manifest")
    ap.add_argument("--panel", help="daily panel CSV (for --build-regime)")
    ap.add_argument("--import-states", action="store_true", help="embed decided-state CSVs into regime.states (no recompute)")
    ap.add_argument("--states", help="comma-separated asset=path.csv pairs (for --import-states), e.g. btc=a.csv,eth=b.csv")
    ap.add_argument("--state-col", default="state", help="name of the state column in the states CSV (default: state)")
    ap.add_argument("--out", help="output manifest path for --build-regime/--import-states (default: overwrite --manifest)")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()
    if args.emit_manifest:
        json.dump(_example_manifest(), open(args.emit_manifest, "w", encoding="utf-8"), indent=2)
        print(f"wrote example manifest -> {args.emit_manifest}")
        return 0

    if args.import_states:
        if not (args.manifest and args.states):
            ap.error("--import-states needs --manifest and --states (asset=path.csv,...)")
        m = load_manifest(args.manifest)
        csv_map = dict(kv.split("=", 1) for kv in args.states.split(","))
        summary = import_states_from_csvs(m, csv_map, args.state_col)
        probs = validate_manifest(m, require_states=True)
        if probs:
            print("MANIFEST INVALID after import -- not writing:")
            for p in probs:
                print("  -", p)
            return 2
        outp = args.out or args.manifest
        json.dump(m, open(outp, "w", encoding="utf-8"), indent=2)
        for a, info in summary.items():
            print(f"  {a}: {info['n']} days {info['start']}..{info['end']} "
                  f"(L={info['longs']} F={info['flats']} S={info['shorts']}, gap-days={info['gap_days']})")
        print(f"embedded regime.states -> {outp}")
        return 0

    if args.build_regime:
        if not (args.manifest and args.panel):
            ap.error("--build-regime needs --manifest and --panel")
        m = load_manifest(args.manifest)
        probs = validate_manifest(m, require_states=False, require_provenance=True)
        if probs:
            print("MANIFEST PROVENANCE INVALID -- cannot build regime:")
            for p in probs:
                print("  -", p)
            return 2
        panel = pd.read_csv(args.panel, index_col=0, parse_dates=True).sort_index()
        m.setdefault("regime", {})["states"] = build_regime(m, panel)
        m["regime"].setdefault("provenance", {})["generated_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
        m["regime"]["provenance"]["source"] = os.path.basename(args.panel)
        outp = args.out or args.manifest
        json.dump(m, open(outp, "w", encoding="utf-8"), indent=2)
        n = {a: len(v) for a, v in m["regime"]["states"].items()}
        print(f"built regime.states {n} -> {outp}")
        return 0

    if not args.manifest:
        ap.error("--manifest is required (or use --selftest / --emit-manifest / --build-regime)")
    m = load_manifest(args.manifest)
    problems = validate_manifest(m, require_states=True)
    if problems:
        print(f"MANIFEST INVALID ({len(problems)} problem(s)) -- refusing to run:")
        for p in problems:
            print("  -", p)
        return 2
    if not args.data_dir:
        ap.error("standalone run needs --data-dir with OHLCV parquet; in a harness, call "
                 "run_backtest(manifest, ohlcv_by_asset) directly with in-memory bars.")
    ohlcv, funding = {}, {}
    for a in m["strategy"]["assets"]:
        df = _load_ohlcv_file(m, a, args.data_dir)
        ohlcv[a] = df
        funding[a] = _load_funding_file(m, a, df.index, args.data_dir)
    run_backtest(m, ohlcv, funding, outdir=args.outdir, verbose=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
