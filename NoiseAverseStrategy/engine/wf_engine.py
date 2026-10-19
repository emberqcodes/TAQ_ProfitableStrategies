"""Walk-forward variant engine (wf spec v1, frozen grid).

Separate from frozen engine.py. Computes per-session RETURNS (per unit
equity) for any grid combo, so walk-forward selection is a windowed
statistic over precomputed arrays. Causality: sigma/lev for session d come
from sessions < d (reuses the engine's construction); decisions signal on
mark closes and fill on next bar opens, identical to spec_v1.
"""
import numpy as np
import pandas as pd
from engine import MARKS, P, build_sessions, load_bars, attach_funding

COST_PER_SIDE = 4.0 / 1e4          # 2 bps taker + 2 bps slippage (frozen)
GRID = dict(interval=[30, 60, 120], k=[1.0, 1.5, 2.0],
            cap=[1, 10 ** 9], stop=['vwap', 'band'], L=[20, 60, 120])
AXES = list(GRID.keys())


def all_combos():
    out = []
    for iv in GRID['interval']:
        for k in GRID['k']:
            for cap in GRID['cap']:
                for st in GRID['stop']:
                    for L in GRID['L']:
                        out.append(dict(interval=iv, k=k, cap=cap, stop=st, L=L))
    return out


def neighbors(idx_tuple):
    """Grid-neighbor index tuples (±1 along each axis), for plateau scoring."""
    out = []
    for ax in range(len(AXES)):
        for d in (-1, 1):
            t = list(idx_tuple)
            t[ax] += d
            if 0 <= t[ax] < len(GRID[AXES[ax]]):
                out.append(tuple(t))
    return out


def causal_sigma(sessions, L, min_obs=None):
    """sigma[d, mark] from the L strictly prior sessions (same rule as engine)."""
    min_obs = min_obs if min_obs is not None else P['min_band_obs']
    n, nmk = len(sessions), len(MARKS)
    absmove = np.full((n, nmk), np.nan)
    for d, s in enumerate(sessions):
        if s.get('valid'):
            absmove[d] = np.abs(s['mark_close'] / s['open'] - 1.0)
    sigma = np.full((n, nmk), np.nan)
    for d in range(n):
        w = absmove[max(0, d - L):d]
        if len(w):
            cnt = np.isfinite(w).sum(axis=0)
            mu = np.nansum(w, axis=0) / np.maximum(cnt, 1)
            sigma[d] = np.where(cnt >= min_obs, mu, np.nan)
    return sigma


def causal_lev_prevclose(sessions):
    """Leverage (14-session vol, frozen sizing) and prior close, from engine rules."""
    n = len(sessions)
    oc = np.full(n, np.nan)
    prev_close = np.full(n, np.nan)
    last = np.nan
    for d, s in enumerate(sessions):
        if s.get('valid'):
            oc[d] = s['exit_close'] / s['open'] - 1.0
            prev_close[d] = last
            last = s['exit_close']
        else:
            prev_close[d] = last
    lev = np.full(n, np.nan)
    VL = P['vol_lookback']
    for d in range(n):
        r = oc[max(0, d - VL):d]
        r = r[np.isfinite(r)]
        if len(r) >= VL:
            v = r.std(ddof=1)
            if v > 0:
                lev[d] = min(P['vol_target'] / v, P['max_leverage'])
    return lev, prev_close


def run_combo(sessions, sigma, lev, prev_close, interval, k, cap, stop,
              cost=COST_PER_SIDE, collect_trades=False, conf=None):
    """Per-session returns per unit equity for one combo.

    interval: minutes between decision checks (30/60/120; subset of MARKS)
    k: band width multiplier on sigma
    cap: max entries per session (a flip consumes an entry; if exhausted,
         a flip signal becomes exit-to-flat)
    stop: 'vwap' -> spec trailing stop max(VWAP,LB)/min(VWAP,UB);
          'band' -> exit only on opposite-band flip or at session close
    conf: optional bool matrix [n_sessions, len(MARKS)] gating ENTRIES only
          (new entries, re-entries, and the entering leg of flips). Exits
          and stops are never gated; an unconfirmed flip exits to flat.
    Returns (ret[d], ntrades[d], trades_list or None).
    """
    n = len(sessions)
    ret = np.full(n, np.nan)
    ntr = np.zeros(n, dtype=int)
    step = interval // 30
    mark_ids = list(range(step - 1, len(MARKS), step))
    trades = [] if collect_trades else None

    for d, s in enumerate(sessions):
        if not s.get('valid') or not np.isfinite(lev[d]):
            continue
        op = s['open']
        pc = prev_close[d] if np.isfinite(prev_close[d]) else op
        hi_a, lo_a = max(op, pc), min(op, pc)
        L_frac = lev[d]
        pos, entry_px, entry_m = 0, np.nan, None
        r = 0.0
        entries = 0
        funded = False

        def leg_return(px_out, side, e_px):
            gross = side * (px_out / e_px - 1.0)
            fee = cost * (1.0 + px_out / e_px)
            return (gross - fee) * L_frac, gross * L_frac

        for i in mark_ids:
            m = MARKS[i]
            if not funded and m >= 16 * 60:
                funded = True
                if pos != 0 and np.isfinite(s['px16']):
                    fr = s.get('funding_rate', np.nan)
                    if np.isfinite(fr):
                        r += -pos * L_frac * (s['px16'] / entry_px) * fr
            p = s['mark_close'][i]
            if not np.isfinite(p):
                continue
            sg = sigma[d, i]
            has_band = np.isfinite(sg)
            ub = hi_a * (1 + k * sg) if has_band else np.nan
            lb = lo_a * (1 - k * sg) if has_band else np.nan
            vw = s['mark_vwap'][i]
            fpx = s['mark_fill'][i]
            ok_entry = True if conf is None else bool(conf[d, i])
            if pos != 0:
                flip_hit = has_band and ((pos > 0 and p < lb) or (pos < 0 and p > ub))
                if stop == 'vwap':
                    if pos > 0:
                        sl = max(vw, lb) if has_band else vw
                        stop_hit = np.isfinite(vw) and p < sl
                    else:
                        sl = min(vw, ub) if has_band else vw
                        stop_hit = np.isfinite(vw) and p > sl
                else:
                    stop_hit = False
                if (flip_hit or stop_hit) and np.isfinite(fpx):
                    dr, dg = leg_return(fpx, pos, entry_px)
                    r += dr
                    ntr[d] += 1
                    if collect_trades:
                        trades.append(dict(date=s['date'], side=pos, entry=entry_px,
                                           exit=fpx, entry_mark=entry_m, exit_mark=m,
                                           ret=dr, gross_ret=dg))
                    old = pos
                    pos = 0
                    if flip_hit and entries < cap and ok_entry:
                        pos, entry_px, entry_m = -old, fpx, m
                        entries += 1
            elif has_band and np.isfinite(fpx) and entries < cap and ok_entry:
                if p > ub:
                    pos, entry_px, entry_m, entries = 1, fpx, m, entries + 1
                elif p < lb:
                    pos, entry_px, entry_m, entries = -1, fpx, m, entries + 1

        if pos != 0:
            dr, dg = leg_return(s['exit_close'], pos, entry_px)
            r += dr
            ntr[d] += 1
            if collect_trades:
                trades.append(dict(date=s['date'], side=pos, entry=entry_px,
                                   exit=s['exit_close'], entry_mark=entry_m,
                                   exit_mark='EOD', ret=dr, gross_ret=dg))
        ret[d] = r
    return ret, ntr, trades
