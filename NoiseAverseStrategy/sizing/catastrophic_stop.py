"""Continuous (intrabar) catastrophic stop — measurement overlay.

Adds a resting catastrophic stop at -X% from entry, checked at 1-minute
resolution (between the 30-min decision marks). Conservative fill model:
when a bar's low (long) / high (short) pierces the stop, fill at the WORSE
of {stop level, that bar's close} -- never better than the trigger. This
biases the stop to look LESS protective than an idealized touch, which is
the safe direction of error.

Scope (bias protocol): this instruments the frozen profitable combo and
reports, per candidate stop level, how the tail and the mean change vs the
no-stop baseline. It is measurement to SEED a forward pre-registration, not
an optimizer. The stop's stated purpose is tail insurance (bound the -600
to -1000 bps gap), explicitly NOT mean improvement. Judge it on the tail
columns, not the mean column.

Reuses the trade tracer from mfe_mae_analysis (entry/exit logic identical to
the deployed combo); layers the intrabar stop on top and recomputes P&L.

Usage (paths from CONFIG block or CLI, same as mfe_mae_analysis):
  python catastrophic_stop.py --ohlcv-dir X --funding-dir Y --out cstop_out \
      --book majors7 --news-calendar macro_calendar_2020_2026.csv \
      --levels 300,400,500,600
"""
import argparse, json, os
import numpy as np
import pandas as pd
from engine import load_bars, build_sessions, attach_funding, MARKS, P
import wf_engine as W
from walkforward_vconf import vol_profile_conf

OHLCV_DIR = '/path/to/ohlcv'
FUNDING_DIR = '/path/to/funding'
OUTPUT_DIR = 'cstop_out'
NEWS_CALENDAR = 'macro_calendar_2020_2026.csv'
BOOK = 'majors7'
COMBO = '{"interval":120,"k":1.0,"stop":"band","L":60,"K":2.0}'
LEVELS = '300,400,500,600'

M7 = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE', 'ADA']
BOOKS = {'all': None, 'majors7': M7, 'trio': ['BTC', 'ETH', 'SOL'], 'ETH': ['ETH']}
COST = 4.0 / 1e4


def trace_with_paths(df, sessions, combo, conf, news_dates):
    """Like the MFE/MAE tracer but retains the intrabar path (bars) per trade
    so a continuous stop can be applied at any level in one pass."""
    interval, k, stop, L = combo['interval'], combo['k'], combo['stop'], combo['L']
    sigma = W.causal_sigma(sessions, L)
    lev, pc = W.causal_lev_prevclose(sessions)
    step = interval // 30
    mark_ids = list(range(step - 1, len(MARKS), step))
    df = df.copy()
    df['mod'] = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute
    df['date'] = df['datetime'].dt.date
    bars_by_day = {d: g.set_index('mod') for d, g in df.groupby('date')}

    trades = []

    def open_trade(side, epx, emod, s, g):
        trades.append(dict(_s=s, _g=g, side=side, entry=epx, emod=emod))

    for d, s in enumerate(sessions):
        if not s.get('valid') or not np.isfinite(lev[d]):
            continue
        if news_dates and str(s['date']) in news_dates:
            continue
        op = s['open']; pcl = pc[d] if np.isfinite(pc[d]) else op
        hi_a, lo_a = max(op, pcl), min(op, pcl)
        g = bars_by_day.get(s['date'])
        if g is None:
            continue
        pos, epx, emod = 0, np.nan, None
        for i in mark_ids:
            m = MARKS[i]; p_mark = s['mark_close'][i]
            if not np.isfinite(p_mark):
                continue
            sg = sigma[d, i]; hb = np.isfinite(sg)
            ub = hi_a * (1 + k * sg) if hb else np.nan
            lb = lo_a * (1 - k * sg) if hb else np.nan
            vw = s['mark_vwap'][i]; fpx = s['mark_fill'][i]
            ok = True if conf is None else bool(conf[d, i])
            if pos != 0:
                flip = hb and ((pos > 0 and p_mark < lb) or (pos < 0 and p_mark > ub))
                if stop == 'vwap':
                    sl = (max(vw, lb) if pos > 0 else min(vw, ub)) if hb else vw
                    hit = np.isfinite(vw) and ((pos > 0 and p_mark < sl) or (pos < 0 and p_mark > sl))
                else:
                    hit = False
                if (flip or hit) and np.isfinite(fpx):
                    trades[-1].update(exit=fpx, xmod=m, reason='flip' if (flip and ok) else 'exit')
                    pos = 0
                    if flip and ok:
                        pos, epx, emod = -trades[-1]['side'], fpx, m
                        open_trade(pos, fpx, m, s, g)
            if pos == 0 and hb and np.isfinite(fpx) and ok:
                if p_mark > ub: pos, epx, emod = 1, fpx, m; open_trade(1, fpx, m, s, g)
                elif p_mark < lb: pos, epx, emod = -1, fpx, m; open_trade(-1, fpx, m, s, g)
        if pos != 0:
            trades[-1].update(exit=s['exit_close'], xmod=MARKS[-1] + 30, reason='EOD')
    return [t for t in trades if 'exit' in t and np.isfinite(t['entry']) and t['entry'] > 0]


def apply_stop(t, level_bps):
    """Return (pnl_bps, stopped_bool) with a continuous intrabar stop at
    -level_bps from entry. Conservative fill: worse of {stop, bar close}."""
    side, epx, emod, xmod = t['side'], t['entry'], t['emod'], t['xmod']
    if level_bps is None:
        return side * (t['exit'] / epx - 1) * 1e4, False
    stop_px = epx * (1 - level_bps / 1e4) if side > 0 else epx * (1 + level_bps / 1e4)
    seg = t['_g'][(t['_g'].index >= emod) & (t['_g'].index < xmod)]
    for mod, bar in seg.iterrows():
        pierced = (bar['low'] <= stop_px) if side > 0 else (bar['high'] >= stop_px)
        if pierced:
            fill = min(stop_px, bar['close']) if side > 0 else max(stop_px, bar['close'])
            return side * (fill / epx - 1) * 1e4, True
    return side * (t['exit'] / epx - 1) * 1e4, False


def _trace_one(args):
    """Worker: trace one asset, return trades WITHOUT the heavy bar frames
    (re-derives excursion-relevant fields the main process needs)."""
    u, combo, news = args
    df = load_bars(u['bars'])
    sessions = build_sessions(df)
    if u['funding'] and os.path.exists(u['funding']):
        sessions = attach_funding(sessions, u['funding'])
    else:
        for s in sessions:
            s['funding_rate'] = np.nan
    conf = None if combo['K'] == 0 else vol_profile_conf(sessions, combo['K'])
    tt = trace_with_paths(df, sessions, combo, conf, news)
    # apply all stop levels here (needs _g); return only light per-(trade,level) rows
    return [dict(base=u['base'], date=t['_s']['date'],
                 base_pnl=apply_stop(t, None)[0],
                 **{f'pnl_{lv}': apply_stop(t, lv)[0] for lv in _trace_one.levels},
                 **{f'stop_{lv}': apply_stop(t, lv)[1] for lv in _trace_one.levels})
            for t in tt]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ohlcv-dir', default=OHLCV_DIR)
    ap.add_argument('--funding-dir', default=FUNDING_DIR)
    ap.add_argument('--out', default=OUTPUT_DIR)
    ap.add_argument('--book', default=BOOK, choices=list(BOOKS))
    ap.add_argument('--news-calendar', default=NEWS_CALENDAR)
    ap.add_argument('--combo', default=COMBO)
    ap.add_argument('--levels', default=LEVELS)
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    if not os.path.isdir(a.ohlcv_dir):
        raise SystemExit(f'ohlcv dir not found: {a.ohlcv_dir} (edit CONFIG or --ohlcv-dir)')
    news = None
    if a.news_calendar and str(a.news_calendar).lower() != 'none':
        news = set(pd.read_csv(a.news_calendar)['date'].astype(str))
    levels = [int(x) for x in a.levels.split(',')]
    from batch_run import discover
    fdir = a.funding_dir if os.path.isdir(a.funding_dir) else os.devnull
    universe, _ = discover(a.ohlcv_dir, fdir)
    keep = BOOKS[a.book]
    if keep is not None:
        universe = [u for u in universe if u['base'] in keep]
    combo = json.loads(a.combo)
    os.makedirs(a.out, exist_ok=True)

    _trace_one.levels = levels
    jobs = [(u, combo, news) for u in universe]
    print(f"processing {len(jobs)} assets on {a.workers} workers...")
    from multiprocessing import Pool
    if a.workers > 1:
        with Pool(a.workers) as pool:
            parts = pool.map(_trace_one, jobs)
    else:
        parts = [_trace_one(j) for j in jobs]
    rowsflat = [r for part in parts for r in part]
    T = pd.DataFrame(rowsflat)
    base = T['base_pnl'].values
    big_thresh = np.percentile(base[base > 0], 90)
    tot0 = base.sum()
    print(f"\n{len(T)} trades | combo={combo}")
    print(f"baseline: mean {base.mean():+.2f} bps | worst {base.min():.0f} | "
          f"1st pct {np.percentile(base,1):.0f} | 5th pct {np.percentile(base,5):.0f}")
    print("\nlevel  mean_pnl  worst  1st_pct  5th_pct  %stopped  big_winners_hit  tot_pnl_chg")
    outrows = [dict(level='none', mean=round(base.mean(),2), worst=round(base.min()),
                    p1=round(np.percentile(base,1)), p5=round(np.percentile(base,5)),
                    stopped=0.0, big_hit=0.0, tot_chg=0.0)]
    for lv in levels:
        pnl = T[f'pnl_{lv}'].values; stp = T[f'stop_{lv}'].values.astype(bool)
        big = base >= big_thresh
        big_hit = (stp & big).sum() / max(1, big.sum()) * 100
        print(f"{lv:>5}  {pnl.mean():+7.2f}  {pnl.min():5.0f}  {np.percentile(pnl,1):6.0f}  "
              f"{np.percentile(pnl,5):6.0f}  {stp.mean()*100:6.1f}%  {big_hit:6.0f}%   "
              f"{(pnl.sum()-tot0)/abs(tot0)*100:+5.1f}%")
        outrows.append(dict(level=lv, mean=round(pnl.mean(),2), worst=round(pnl.min()),
                            p1=round(np.percentile(pnl,1)), p5=round(np.percentile(pnl,5)),
                            stopped=round(stp.mean()*100,1), big_hit=round(big_hit,1),
                            tot_chg=round((pnl.sum()-tot0)/abs(tot0)*100,1)))
    pd.DataFrame(outrows).to_csv(os.path.join(a.out, f'cstop_{a.book}.csv'), index=False)
    print("\nRead: pick the level that MOST reduces the 1st/5th-pct tail while")
    print("keeping big_winners_hit low and mean roughly flat. Tail insurance:")
    print("a near-flat mean with a shrunk tail is SUCCESS. Do NOT pick max-mean.")
    print(f"saved -> {os.path.join(a.out, f'cstop_{a.book}.csv')}")
    return


if __name__ == '__main__':
    main()
