"""Trade-level MFE/MAE excursion analysis for the frozen profitable combo.

Scope (bias protocol): this is MEASUREMENT, not tuning. It runs the frozen
walk-forward-selected combo once, recording each trade's maximum favorable
excursion (MFE) and maximum adverse excursion (MAE) over its life, using
1-minute bars. It changes no parameters and makes no strategy decision.
The hypothetical SL/TP curves it prints are DIAGNOSTIC ONLY: they show what
fixed stops/targets WOULD have done in-sample, which is exactly the number
you must NOT optimize on — see the printed warning. Their purpose is to
decide whether a stop has any job to do, and to seed a FORWARD pre-reg.

What "MFE/MAE" means here: for each trade, from entry to actual exit, at
1-minute resolution, track the running favorable and adverse move (signed by
position side), in basis points of entry price. Also record realized P&L,
exit reason, and holding time.

Paths and options are set in the CONFIG block near the top of this file, or
overridden per-run via CLI flags:
  python mfe_mae_analysis.py --ohlcv-dir <dir> --funding-dir <dir> \
      --out mfe_out [--book majors7] [--news-calendar macro_calendar_2020_2026.csv]
      [--combo '{"interval":120,"k":1.0,"stop":"band","L":60,"K":2.0}']

If --combo is omitted, the CONFIG default combo is used. Set --news-calendar
to 'none' to disable the blackout.
"""
import argparse, json, os
import numpy as np
import pandas as pd
from engine import load_bars, build_sessions, attach_funding, MARKS, SESSION_OPEN_MIN, P
import wf_engine as W
from walkforward_vconf import vol_profile_conf, GRID, all_combos, quarter_starts, sharpe

# ============================ CONFIG (edit me) ============================
# CLI flags override these defaults when provided.
OHLCV_DIR = '/path/to/ohlcv'          # folder of <BASE>_USDT_USDT.parquet 1m bars
FUNDING_DIR = '/path/to/funding'      # folder of <BASE>_USDT_USDT.parquet funding
OUTPUT_DIR = 'mfe_out'                # where excursions_<book>.csv is written
NEWS_CALENDAR = 'macro_calendar_2020_2026.csv'  # or None to disable the blackout
BOOK = 'majors7'                      # all | majors7 | trio | ETH
COMBO = '{"interval":120,"k":1.0,"stop":"band","L":60,"K":2.0}'
TRAIN_DAYS = 180
# =========================================================================

M7 = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE', 'ADA']
BOOKS = {'all': None, 'majors7': M7, 'trio': ['BTC', 'ETH', 'SOL'], 'ETH': ['ETH']}


def trace_trades(df, sessions, combo, conf, news_dates):
    """Re-simulate the combo and, for each trade, walk 1-min bars entry->exit
    capturing MFE/MAE in bps. Returns a DataFrame of trades with excursions."""
    interval, k, stop, L = combo['interval'], combo['k'], combo['stop'], combo['L']
    sigma = W.causal_sigma(sessions, L)
    lev, pc = W.causal_lev_prevclose(sessions)
    step = interval // 30
    mark_ids = list(range(step - 1, len(MARKS), step))

    # index minute bars by (date, minute-of-day) for path lookup
    df = df.copy()
    df['mod'] = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute
    df['date'] = df['datetime'].dt.date
    bars_by_day = {d: g.set_index('mod') for d, g in df.groupby('date')}

    rows = []
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
        pos, entry_px, entry_mod = 0, np.nan, None
        for i in mark_ids:
            m = MARKS[i]
            p_mark = s['mark_close'][i]
            if not np.isfinite(p_mark):
                continue
            sg = sigma[d, i]; hb = np.isfinite(sg)
            ub = hi_a * (1 + k * sg) if hb else np.nan
            lb = lo_a * (1 - k * sg) if hb else np.nan
            vw = s['mark_vwap'][i]; fpx = s['mark_fill'][i]
            ok_entry = True if conf is None else bool(conf[d, i])
            close_now, flip_to = False, 0
            if pos != 0:
                flip = hb and ((pos > 0 and p_mark < lb) or (pos < 0 and p_mark > ub))
                if stop == 'vwap':
                    sl = (max(vw, lb) if pos > 0 else min(vw, ub)) if hb else vw
                    hit = np.isfinite(vw) and ((pos > 0 and p_mark < sl) or (pos < 0 and p_mark > sl))
                else:
                    hit = False
                if flip or hit:
                    close_now = True
                    if flip and ok_entry:
                        flip_to = -pos
            if close_now and np.isfinite(fpx):
                _record(rows, g, s, pos, entry_px, entry_mod, m, fpx, 'flip' if flip_to else 'stop')
                pos = flip_to; entry_px = fpx if flip_to else np.nan; entry_mod = m if flip_to else None
            elif pos == 0 and hb and np.isfinite(fpx) and ok_entry:
                if p_mark > ub: pos, entry_px, entry_mod = 1, fpx, m
                elif p_mark < lb: pos, entry_px, entry_mod = -1, fpx, m
        if pos != 0:
            _record(rows, g, s, pos, entry_px, entry_mod, MARKS[-1] + 30, s['exit_close'], 'EOD')
    return pd.DataFrame(rows)


def _record(rows, g, s, side, entry_px, entry_mod, exit_mod, exit_px, reason):
    if not np.isfinite(entry_px) or entry_px <= 0:
        return
    seg = g[(g.index >= entry_mod) & (g.index <= exit_mod)]
    if seg.empty:
        mfe = mae = 0.0
    else:
        hi = seg['high'].max(); lo = seg['low'].min()
        if side > 0:
            mfe = (hi / entry_px - 1) * 1e4; mae = (lo / entry_px - 1) * 1e4
        else:
            mfe = (entry_px / lo - 1) * 1e4; mae = (entry_px / hi - 1) * 1e4
    pnl = side * (exit_px / entry_px - 1) * 1e4
    rows.append(dict(date=str(s['date']), side=side, reason=reason,
                     hold_min=int(exit_mod - entry_mod), pnl_bps=pnl,
                     mfe_bps=mfe, mae_bps=mae))


def hypothetical_sl_tp(tr):
    """DIAGNOSTIC ONLY: what fixed SL/TP would have done in-sample."""
    print("\n--- HYPOTHETICAL fixed SL/TP (IN-SAMPLE, DIAGNOSTIC ONLY) ---")
    print("!! Do NOT pick the max of this table. It is the number the forward")
    print("!! test exists to check. Use it only to seed a pre-registered range.")
    base = tr['pnl_bps'].mean()
    print(f"baseline mean P&L/trade: {base:+.2f} bps (no SL/TP)")
    print("  SL(bps)   mean_pnl   %stopped     TP(bps)   mean_pnl   %capped")
    for lvl in [50, 100, 150, 200, 300]:
        capped_pnl = np.where(tr['mae_bps'] <= -lvl, -lvl, tr['pnl_bps'])  # SL
        tp_pnl = np.where(tr['mfe_bps'] >= lvl, lvl, tr['pnl_bps'])        # TP
        print(f"  {lvl:>5}   {capped_pnl.mean():+7.2f}   {(tr['mae_bps']<=-lvl).mean()*100:5.1f}%     "
              f"{lvl:>5}   {tp_pnl.mean():+7.2f}   {(tr['mfe_bps']>=lvl).mean()*100:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ohlcv-dir', default=OHLCV_DIR)
    ap.add_argument('--funding-dir', default=FUNDING_DIR)
    ap.add_argument('--out', default=OUTPUT_DIR)
    ap.add_argument('--book', default=BOOK, choices=list(BOOKS))
    ap.add_argument('--news-calendar', default=NEWS_CALENDAR)
    ap.add_argument('--combo', default=COMBO,
                    help='fixed combo to instrument (default from CONFIG block)')
    ap.add_argument('--train-days', type=int, default=TRAIN_DAYS)
    a = ap.parse_args()
    if not os.path.isdir(a.ohlcv_dir):
        raise SystemExit(f'ohlcv dir not found: {a.ohlcv_dir} '
                         f'(edit OHLCV_DIR in the CONFIG block or pass --ohlcv-dir)')
    news = None
    if a.news_calendar and str(a.news_calendar).lower() != 'none':
        if not os.path.exists(a.news_calendar):
            raise SystemExit(f'news calendar not found: {a.news_calendar}')
        news = set(pd.read_csv(a.news_calendar)['date'].astype(str))
    from batch_run import discover
    fdir = a.funding_dir if os.path.isdir(a.funding_dir) else os.devnull
    universe, _ = discover(a.ohlcv_dir, fdir)
    keep = BOOKS[a.book]
    if keep is not None:
        universe = [u for u in universe if u['base'] in keep]
    combo = json.loads(a.combo)
    os.makedirs(a.out, exist_ok=True)

    alltr = []
    for u in universe:
        df = load_bars(u['bars'])
        sessions = build_sessions(df)
        if u['funding'] and os.path.exists(u['funding']):
            sessions = attach_funding(sessions, u['funding'])
        else:
            for s in sessions: s['funding_rate'] = np.nan
        conf = None if combo['K'] == 0 else vol_profile_conf(sessions, combo['K'])
        tr = trace_trades(df, sessions, combo, conf, news)
        tr['base'] = u['base']
        alltr.append(tr)
    tr = pd.concat(alltr, ignore_index=True)
    tr.to_csv(os.path.join(a.out, f'excursions_{a.book}.csv'), index=False)

    win = tr[tr.pnl_bps > 0]; los = tr[tr.pnl_bps <= 0]
    def q(s, col): return {p: round(float(np.percentile(s[col], p)), 1) for p in [10,25,50,75,90]}
    print(f"book={a.book} combo={combo} | {len(tr)} trades | mean P&L {tr.pnl_bps.mean():+.2f} bps")
    print(f"\nMFE (max favorable, bps) all: {q(tr,'mfe_bps')}")
    print(f"MAE (max adverse,   bps) all: {q(tr,'mae_bps')}")
    print(f"\nWINNERS ({len(win)}, {len(win)/len(tr)*100:.0f}%): MFE {q(win,'mfe_bps')} | MAE {q(win,'mae_bps')}")
    print(f"LOSERS  ({len(los)}, {len(los)/len(tr)*100:.0f}%): MFE {q(los,'mfe_bps')} | MAE {q(los,'mae_bps')}")
    print(f"\nedge-giveback: winners' median MFE {np.median(win.mfe_bps):.0f} vs median exit {np.median(win.pnl_bps):.0f} bps")
    print(f"loser MAE tail (5th pct): {np.percentile(tr.mae_bps,5):.0f} bps  <- gap-risk indicator")
    hypothetical_sl_tp(tr)
    print(f"\nsaved -> {os.path.join(a.out, f'excursions_{a.book}.csv')}")


if __name__ == '__main__':
    main()
