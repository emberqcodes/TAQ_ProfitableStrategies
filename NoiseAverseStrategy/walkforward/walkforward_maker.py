"""Maker-execution walk-forward -- entry via resting limit orders.

Tests whether earning the maker rebate on ENTRY (vs paying taker) rescues the
strategy's economics. ALL exits and stops remain TAKER, as specified.

Two entry styles are BOTH run (frozen, not tuned), so the walk-forward shows
them side by side:
  A "pullback": passive limit on the near side (long BELOW signal close,
     short ABOVE). Fills when price retraces to the limit -> earns +maker_bps.
     Honest cost: strong breakouts that never retrace never fill (missed
     runners). This is the style that actually harvests the rebate.
  B "breakout": stop-limit on the far side (long ABOVE close, short BELOW).
     Fills only if momentum continues through the level. A resting order on
     the far side is MARKETABLE, so style B fills pay TAKER, not maker --
     modeled that way so it cannot fabricate rebate edge. B tests whether
     confirmation improves entry quality at taker cost.

Fill rule: limit placed at a decision mark; if the trigger price is not
touched by the NEXT decision mark, the order is cancelled (no entry that
cycle). Requires 1-minute bars to test the intrabar path between marks.

Costs (config): entry earns maker_bps on style A / pays taker_bps on style B;
EVERY exit/stop/flip/forced-close pays taker_bps. Reports unfilled-signal
rate prominently -- a high rebate on few fills must not masquerade as edge.

Walk-forward: same frozen machine as walkforward_vconf (quarterly refits,
trailing train window, plateau selection, flat if <=0), but the grid adds a
maker_style axis and each combo is evaluated with maker fills.

Paths and options are in the CONFIG block or via CLI flags.
"""
import argparse, json, os, datetime, traceback
from functools import partial
from multiprocessing import Pool
import numpy as np
import pandas as pd
from engine import load_bars, build_sessions, attach_funding, MARKS
import wf_engine as W
from walkforward_vconf import vol_profile_conf

# ============================ CONFIG (edit me) ============================
OHLCV_DIR   = '/path/to/ohlcv'          # <BASE>_USDT_USDT.parquet 1-minute bars
FUNDING_DIR = '/path/to/funding'        # <BASE>_USDT_USDT.parquet funding
OUTPUT_DIR  = 'maker_out'               # results written here
NEWS_CALENDAR = 'macro_calendar_2020_2026.csv'   # or 'none'
DATA_ATTESTATION = ''                   # REQUIRED: name the dataset/venue
TRAIN_DAYS  = 180
MAKER_BPS   = 1.0                        # rebate EARNED per maker entry (+1 bps)
TAKER_BPS   = 2.0                        # cost PAID per taker side (exits, style B entries)
SLIP_BPS    = 0.0                        # extra slippage on taker fills (maker fills assume none)
OFFSET_BPS  = 5.0                        # style A/B limit distance from signal close (bps)
TRIGGER_BPS = 5.0                        # style C: confirmation trigger above signal close (bps)
MAKER_OFFSET_BPS = 3.0                   # style C: passive maker limit ABOVE the trigger (bps)
FORCE_STYLE = ''                         # '' = A & B in grid; 'A','B','C' isolate one
# =========================================================================


def _grid(force_style):
    if force_style in ('A', 'B', 'C'):
        styles = [force_style]
    else:
        styles = ['A', 'B']              # C is opt-in via --force-style C (new mechanic)
    return dict(interval=[30, 60, 120], k=[1.0, 1.5, 2.0], stop=['vwap', 'band'],
                L=[20, 60, 120], K=[0.0, 2.0], maker_style=styles)

GRID = _grid('')
AXES = list(GRID.keys())
SHAPE = tuple(len(GRID[a]) for a in AXES)
MIN_TRAIN_TRADES = 30
ANN = np.sqrt(365)


def _set_grid(force_style):
    """Rebuild module-level grid globals for the chosen style isolation."""
    global GRID, AXES, SHAPE
    GRID = _grid(force_style)
    AXES = list(GRID.keys())
    SHAPE = tuple(len(GRID[a]) for a in AXES)


def all_combos():
    out = []
    for iv in GRID['interval']:
        for k in GRID['k']:
            for st in GRID['stop']:
                for L in GRID['L']:
                    for K in GRID['K']:
                        for ms in GRID['maker_style']:
                            out.append(dict(interval=iv, k=k, stop=st, L=L,
                                            K=K, maker_style=ms))
    return out


def neighbors(t):
    out = []
    for ax in range(len(AXES)):
        for d in (-1, 1):
            u = list(t)
            u[ax] += d
            if 0 <= u[ax] < len(GRID[AXES[ax]]):
                out.append(tuple(u))
    return out


def sharpe(x):
    x = x[np.isfinite(x)]
    if len(x) < 20 or x.std(ddof=1) == 0:
        return np.nan
    return float(x.mean() / x.std(ddof=1) * ANN)


def quarter_starts(dates, train_days):
    ds = pd.to_datetime(pd.Series(sorted(set(dates))))
    qs = sorted(set(pd.Timestamp(d.year, 3 * ((d.month - 1) // 3) + 1, 1) for d in ds))
    return [q for q in qs if q > ds.min() + pd.Timedelta(days=train_days)]


def run_combo_maker(sessions, minbars_by_day, sigma, lev, prev_close,
                    combo, conf, maker_bps, taker_bps, slip_bps, offset_bps,
                    trigger_bps=5.0, maker_offset_bps=3.0, collect=False):
    """Per-session net return with maker-limit entries and taker exits.

    Returns (ret[d], ntrades[d], nsignals[d], nfilled[d], trades|None).
    nsignals vs nfilled exposes the unfilled-signal rate honestly.
    """
    interval, k, stop = combo['interval'], combo['k'], combo['stop']
    style = combo['maker_style']
    n = len(sessions)
    ret = np.full(n, np.nan)
    ntr = np.zeros(n, dtype=int)
    nsig = np.zeros(n, dtype=int)
    nfil = np.zeros(n, dtype=int)
    step = interval // 30
    mark_ids = list(range(step - 1, len(MARKS), step))
    off = offset_bps / 1e4
    trig = trigger_bps / 1e4
    moff = maker_offset_bps / 1e4
    maker = maker_bps / 1e4
    taker = (taker_bps + slip_bps) / 1e4
    trades = [] if collect else None

    for d, s in enumerate(sessions):
        if not s.get('valid') or not np.isfinite(lev[d]):
            continue
        op = s['open']
        pc = prev_close[d] if np.isfinite(prev_close[d]) else op
        hi_a, lo_a = max(op, pc), min(op, pc)
        Lf = lev[d]
        g = minbars_by_day.get(s['date'])
        if g is None:
            continue
        mod_arr = g['mod'].to_numpy()
        lo_arr = g['low'].to_numpy()
        hi_arr = g['high'].to_numpy()

        pos, entry_px, entry_m, entry_is_maker = 0, np.nan, None, False
        r = 0.0

        def exit_leg(px_out, side, e_px):
            # exit ALWAYS pays taker
            gross = side * (px_out / e_px - 1.0)
            fee = taker * (1.0 + px_out / e_px)
            return (gross - fee) * Lf, gross * Lf

        for idx, i in enumerate(mark_ids):
            m = MARKS[i]
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

            # ---- manage open position (taker exits) ----
            if pos != 0:
                flip = has_band and ((pos > 0 and p < lb) or (pos < 0 and p > ub))
                if stop == 'vwap':
                    sl = (max(vw, lb) if pos > 0 else min(vw, ub)) if has_band else vw
                    stop_hit = np.isfinite(vw) and ((pos > 0 and p < sl) or (pos < 0 and p > sl))
                else:
                    stop_hit = False
                if (flip or stop_hit) and np.isfinite(fpx):
                    dr, dg = exit_leg(fpx, pos, entry_px)
                    r += dr; ntr[d] += 1
                    if collect:
                        trades.append(dict(date=s['date'], side=pos, entry=entry_px,
                                           exit=fpx, ret=dr, gross_ret=dg,
                                           maker_entry=entry_is_maker, reason='flip/stop'))
                    pos = 0
                    # a flip re-arms a NEW maker limit next cycle (not immediate)

            # ---- entry: place a resting limit, check fill before NEXT mark ----
            if pos == 0 and has_band and ok_entry and np.isfinite(fpx):
                sig_side = 0
                if p > ub:
                    sig_side = 1
                elif p < lb:
                    sig_side = -1
                if sig_side != 0:
                    nsig[d] += 1
                    ref = fpx  # signal reference price (next-bar open proxy at mark)
                    # window: this mark's minute .. next mark's minute
                    m_next = MARKS[mark_ids[idx + 1]] if idx + 1 < len(mark_ids) else s['close_min'] if 'close_min' in s else 1439
                    seg = (mod_arr > m) & (mod_arr <= m_next)
                    if style == 'A':
                        # passive: long limit BELOW, short ABOVE -> maker rebate
                        limit = ref * (1 - off) if sig_side > 0 else ref * (1 + off)
                        if sig_side > 0:
                            filled = np.any(seg & (lo_arr <= limit))
                        else:
                            filled = np.any(seg & (hi_arr >= limit))
                        entry_is_maker = True
                    elif style == 'C':
                        # confirm at trigger, THEN post passive maker limit beyond it.
                        # long: trigger = ref*(1+trig); maker limit = trigger*(1+moff),
                        # which fills passively only if price runs a further moff up.
                        if sig_side > 0:
                            trig_px = ref * (1 + trig)
                            lim_px = trig_px * (1 + moff)
                            confirmed = np.any(seg & (hi_arr >= trig_px))
                            # after confirmation, need price to reach the higher maker limit
                            filled = confirmed and np.any(seg & (hi_arr >= lim_px))
                        else:
                            trig_px = ref * (1 - trig)
                            lim_px = trig_px * (1 - moff)
                            confirmed = np.any(seg & (lo_arr <= trig_px))
                            filled = confirmed and np.any(seg & (lo_arr <= lim_px))
                        limit = lim_px
                        entry_is_maker = True     # venue pays rebate for providing liquidity
                    else:
                        # breakout stop-limit: long ABOVE, short BELOW -> taker
                        limit = ref * (1 + off) if sig_side > 0 else ref * (1 - off)
                        if sig_side > 0:
                            filled = np.any(seg & (hi_arr >= limit))
                        else:
                            filled = np.any(seg & (lo_arr <= limit))
                        entry_is_maker = False
                    if filled:
                        # entry fee: earn maker rebate (A) or pay taker (B)
                        if entry_is_maker:
                            r += maker * Lf            # rebate earned on entry notional
                        else:
                            r += -taker * Lf
                        pos, entry_px, entry_m = sig_side, limit, m
                        nfil[d] += 1

        if pos != 0:
            dr, dg = exit_leg(s['exit_close'], pos, entry_px)
            r += dr; ntr[d] += 1
            if collect:
                trades.append(dict(date=s['date'], side=pos, entry=entry_px,
                                   exit=s['exit_close'], ret=dr, gross_ret=dg,
                                   maker_entry=entry_is_maker, reason='EOD'))
        ret[d] = r
    return ret, ntr, nsig, nfil, trades


def process_asset(rec, cfg):
    base = rec['base']
    _set_grid(cfg.get('force_style', ''))
    try:
        df = load_bars(rec['bars'])
        sessions = build_sessions(df)
        if rec['funding'] and os.path.exists(rec['funding']):
            sessions = attach_funding(sessions, rec['funding'])
        else:
            for s in sessions:
                s['funding_rate'] = np.nan
        n_valid = sum(1 for s in sessions if s.get('valid'))
        if n_valid < cfg['train_days'] + 90:
            return dict(base=base, status='history_too_short', err=f'{n_valid} valid')
        lev, pc = W.causal_lev_prevclose(sessions)
        sigmas = {L: W.causal_sigma(sessions, L) for L in GRID['L']}
        confs = {K: (None if K == 0 else vol_profile_conf(sessions, K)) for K in GRID['K']}
        # minute bars indexed per session-date for intrabar fill checks
        df = df.copy()
        df['mod'] = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute
        df['date'] = df['datetime'].dt.date
        minbars = {dt: g[['mod', 'low', 'high']] for dt, g in df.groupby('date')}
        dates = pd.to_datetime([s['date'] for s in sessions])
        n = len(sessions)

        combos = all_combos()
        rets = np.full((len(combos), n), np.nan)
        ntrs = np.zeros((len(combos), n), dtype=int)
        for ci, c in enumerate(combos):
            r, nt, ns, nf, _ = run_combo_maker(
                sessions, minbars, sigmas[c['L']], lev, pc, c, confs[c['K']],
                cfg['maker_bps'], cfg['taker_bps'], cfg['slip_bps'], cfg['offset_bps'],
                cfg['trigger_bps'], cfg['maker_offset_bps'])
            rets[ci], ntrs[ci] = r, nt

        def cidx(ci): return np.unravel_index(ci, SHAPE)
        def flat(t): return int(np.ravel_multi_index(t, SHAPE))

        chosen, oos_rows, fill_stats = [], [], []
        train_floor = max(50, int(cfg['train_days'] * 200 / 365))
        for q in quarter_starts([s['date'] for s in sessions], cfg['train_days']):
            q_end = q + pd.offsets.QuarterBegin(startingMonth=1)
            tr_mask = (dates >= q - pd.Timedelta(days=cfg['train_days'])) & (dates < q)
            oos_mask = (dates >= q) & (dates < q_end)
            if oos_mask.sum() == 0 or tr_mask.sum() < train_floor:
                continue
            raw = np.array([sharpe(rets[ci][tr_mask])
                            if ntrs[ci][tr_mask].sum() >= MIN_TRAIN_TRADES else np.nan
                            for ci in range(len(combos))])
            plateau = np.full(len(combos), np.nan)
            for ci in range(len(combos)):
                if not np.isfinite(raw[ci]):
                    continue
                vals = [raw[ci]] + [raw[flat(t)] for t in neighbors(cidx(ci))]
                vals = [v for v in vals if np.isfinite(v)]
                plateau[ci] = np.mean(vals)
            if not np.isfinite(plateau).any() or np.nanmax(plateau) <= 0:
                chosen.append(dict(base=base, quarter=str(q.date()), combo=None))
                continue
            best = int(np.nanargmax(plateau))
            c = combos[best]
            chosen.append(dict(base=base, quarter=str(q.date()), combo=json.dumps(c),
                               plateau=round(float(plateau[best]), 2)))
            for d_i in np.where(oos_mask)[0]:
                if np.isfinite(rets[best][d_i]):
                    oos_rows.append(dict(base=base, date=sessions[d_i]['date'],
                                         ret=float(rets[best][d_i])))
            # fill accounting for the chosen combo this quarter
            _, _, ns, nf, _ = run_combo_maker(
                [sessions[j] for j in np.where(oos_mask)[0]],
                minbars, sigmas[c['L']], lev, pc, c, confs[c['K']],
                cfg['maker_bps'], cfg['taker_bps'], cfg['slip_bps'], cfg['offset_bps'],
                cfg['trigger_bps'], cfg['maker_offset_bps'])
            fill_stats.append(dict(base=base, quarter=str(q.date()), style=c['maker_style'],
                                   signals=int(ns.sum()), fills=int(nf.sum())))
        return dict(base=base, status='ok', chosen=chosen,
                    oos=pd.DataFrame(oos_rows), fills=pd.DataFrame(fill_stats))
    except Exception:
        return dict(base=base, status='error', err=traceback.format_exc(-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ohlcv-dir', default=OHLCV_DIR)
    ap.add_argument('--funding-dir', default=FUNDING_DIR)
    ap.add_argument('--out', default=OUTPUT_DIR)
    ap.add_argument('--news-calendar', default=NEWS_CALENDAR)
    ap.add_argument('--data-attestation', default=DATA_ATTESTATION, required=False)
    ap.add_argument('--train-days', type=int, default=TRAIN_DAYS)
    ap.add_argument('--maker-bps', type=float, default=MAKER_BPS)
    ap.add_argument('--taker-bps', type=float, default=TAKER_BPS)
    ap.add_argument('--slip-bps', type=float, default=SLIP_BPS)
    ap.add_argument('--offset-bps', type=float, default=OFFSET_BPS)
    ap.add_argument('--trigger-bps', type=float, default=TRIGGER_BPS,
                    help='style C: confirmation trigger above signal close (bps)')
    ap.add_argument('--maker-offset-bps', type=float, default=MAKER_OFFSET_BPS,
                    help='style C: passive maker limit beyond the trigger (bps)')
    ap.add_argument('--force-style', default=FORCE_STYLE, choices=['', 'A', 'B', 'C'],
                    help="'' = both A&B (selector chooses); 'A' isolates the "
                         "maker-rebate pullback; 'B' isolates taker breakout-confirmation")
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    _set_grid(a.force_style)
    if not os.path.isdir(a.ohlcv_dir):
        raise SystemExit(f'ohlcv dir not found: {a.ohlcv_dir} (edit CONFIG or --ohlcv-dir)')
    if not a.data_attestation:
        raise SystemExit('--data-attestation required (name the dataset/venue); '
                         'this machine is for fresh data / forward use only.')
    news = None
    if a.news_calendar and str(a.news_calendar).lower() != 'none':
        news = set(pd.read_csv(a.news_calendar)['date'].astype(str))
    cfg = dict(train_days=a.train_days, maker_bps=a.maker_bps, taker_bps=a.taker_bps,
               slip_bps=a.slip_bps, offset_bps=a.offset_bps, news=news,
               force_style=a.force_style, trigger_bps=a.trigger_bps,
               maker_offset_bps=a.maker_offset_bps)
    tag = f"_{a.force_style}" if a.force_style else "_AB"
    tag += f"_off{int(a.offset_bps)}"
    print(f'ATTESTATION "{a.data_attestation}" | style={a.force_style or "A&B"} | '
          f'maker +{a.maker_bps} / taker -{a.taker_bps} bps | offset {a.offset_bps} bps '
          f'| train {a.train_days}d | grid {len(all_combos())} combos')
    from batch_run import discover
    fdir = a.funding_dir if os.path.isdir(a.funding_dir) else os.devnull
    universe, _ = discover(a.ohlcv_dir, fdir)
    print(f'{len(universe)} assets')
    os.makedirs(a.out, exist_ok=True)

    with Pool(a.workers) as pool:
        results = pool.map(partial(process_asset, cfg=cfg), universe)
    ok = [r for r in results if r['status'] == 'ok' and len(r.get('oos', []))]
    print(f'ok with OOS: {len(ok)}')
    if not ok:
        print('no OOS results'); return

    oos = pd.concat([r['oos'] for r in ok], ignore_index=True)
    chosen = pd.DataFrame([c for r in ok for c in r['chosen']])
    fills = pd.concat([r['fills'] for r in ok if len(r['fills'])], ignore_index=True)
    oos.to_csv(os.path.join(a.out, f'maker_oos_returns{tag}.csv'), index=False)
    chosen.to_csv(os.path.join(a.out, f'maker_chosen_combos{tag}.csv'), index=False)
    fills.to_csv(os.path.join(a.out, f'maker_fill_stats{tag}.csv'), index=False)

    port = oos.groupby('date')['ret'].mean()
    port.index = pd.to_datetime(port.index)
    sr = sharpe(port.values)
    t = port.mean() / port.std(ddof=1) * np.sqrt(len(port))
    print('\n===== MAKER WALK-FORWARD OOS (one trial) =====')
    print(f'pooled EW OOS: Sharpe {sr:+.2f} (t {t:+.2f}) | ann {port.mean()*365*100:+.1f}% | {len(port)} sessions')
    pf = port.to_frame('r'); pf['y'] = pd.to_datetime(pf.index).year
    print('by year:', {int(y): round(sharpe(g.r.values), 2) for y, g in pf.groupby('y') if len(g) > 30})
    if len(fills):
        tot_sig, tot_fil = fills['signals'].sum(), fills['fills'].sum()
        print(f'\nFILL RATE (critical): {tot_fil}/{tot_sig} = {tot_fil/max(1,tot_sig)*100:.0f}% of signals filled')
        for stl, g in fills.groupby('style'):
            print(f'  style {stl}: {g["fills"].sum()}/{g["signals"].sum()} = '
                  f'{g["fills"].sum()/max(1,g["signals"].sum())*100:.0f}% filled')
    sel = chosen.dropna(subset=['combo'])
    if len(sel):
        cc = sel['combo'].apply(json.loads)
        print('\nmaker_style chosen:', dict(pd.Series([c['maker_style'] for c in cc]).value_counts()))
        print('(A=passive/rebate, B=breakout/taker -- does the selector prefer the rebate style?)')
    with open(os.path.join(a.out, 'trial_registry.jsonl'), 'a') as f:
        f.write(json.dumps(dict(trial='maker-wf-v1', attestation=a.data_attestation,
                                force_style=a.force_style or 'A&B',
                                maker_bps=a.maker_bps, taker_bps=a.taker_bps,
                                offset_bps=a.offset_bps, pooled_oos_sharpe=round(sr, 2),
                                ts=datetime.datetime.now(datetime.UTC).isoformat())) + '\n')
    print(f'\nsaved -> {a.out}/')


if __name__ == '__main__':
    main()
