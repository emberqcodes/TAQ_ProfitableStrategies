"""Deployable sized curve for the Style C maker machine.

Takes the walk-forward-selected Style C combo per quarter (the +1.17 machine),
applies the continuous intrabar catastrophic stop and fixed-fractional risk
sizing, and produces the final sized equity curve for the report.

Honesty guarantees preserved from the research chain:
  * Entries reproduce Style C exactly: confirm at +trigger_bps, then a passive
    maker limit +maker_offset_bps beyond it; fill earns +maker_bps rebate.
  * Catastrophic stop measured from the ACTUAL maker fill price (where the
    position sits), continuous intrabar, conservative fills (worse of
    {stop, bar close} on gap-through).
  * Exits/stops pay taker. Sizing uses the NOMINAL stop distance; gap-through
    trades realize > RISK_FRAC and that tail is reported, not assumed away.
  * Per-quarter combo comes from the SAME selection the walk-forward made
    (re-derived here on trailing data only -> causal, no lookahead).

Config in the block below or via CLI (paths, risk, stop, maker offsets).
"""
import argparse, json, os, datetime, traceback
from functools import partial
from multiprocessing import Pool
import numpy as np
import pandas as pd
from engine import load_bars, build_sessions, attach_funding, MARKS
import wf_engine as W
from walkforward_vconf import vol_profile_conf
import walkforward_maker as WM

# ============================ CONFIG (edit me) ============================
OHLCV_DIR   = '/path/to/ohlcv'
FUNDING_DIR = '/path/to/funding'
OUTPUT_DIR  = 'sized_C_out'
NEWS_CALENDAR = 'macro_calendar_2020_2026.csv'
DATA_ATTESTATION = ''
TRAIN_DAYS  = 180
RISK_FRAC   = 0.0001        # 0.01% of equity risked per trade (decimal, not %)
MAX_LEV     = 20.0
CAT_STOP_BPS = 400.0
MAKER_BPS   = 1.0
TAKER_BPS   = 2.0
TRIGGER_BPS = 5.0
MAKER_OFFSET_BPS = 3.0
# =========================================================================

ANN = np.sqrt(365)
MIN_TRAIN_TRADES = 30


def sharpe(x):
    x = x[np.isfinite(x)]
    if len(x) < 20 or x.std(ddof=1) == 0:
        return np.nan
    return float(x.mean() / x.std(ddof=1) * ANN)


def trace_C_trades(sessions, minbars, sigma, lev, prev_close, combo, conf,
                   cfg, day_indices):
    """Reproduce Style C entries for the given session indices and return a
    list of trades with entry price, side, and the intrabar path to exit so a
    catastrophic stop + sizing can be applied. Exits mirror run_combo_maker."""
    interval, k, stop = combo['interval'], combo['k'], combo['stop']
    step = interval // 30
    mark_ids = list(range(step - 1, len(MARKS), step))
    trig = cfg['trigger_bps'] / 1e4
    moff = cfg['maker_offset_bps'] / 1e4
    trades = []
    for d in day_indices:
        s = sessions[d]
        if not s.get('valid') or not np.isfinite(lev[d]):
            continue
        op = s['open']
        pc = prev_close[d] if np.isfinite(prev_close[d]) else op
        hi_a, lo_a = max(op, pc), min(op, pc)
        g = minbars.get(s['date'])
        if g is None:
            continue
        mod_arr = g['mod'].to_numpy(); lo_arr = g['low'].to_numpy(); hi_arr = g['high'].to_numpy()
        pos, entry_px, entry_mod = 0, np.nan, None
        for idx, i in enumerate(mark_ids):
            m = MARKS[i]; p = s['mark_close'][i]
            if not np.isfinite(p):
                continue
            sg = sigma[d, i]; has_band = np.isfinite(sg)
            ub = hi_a * (1 + k * sg) if has_band else np.nan
            lb = lo_a * (1 - k * sg) if has_band else np.nan
            vw = s['mark_vwap'][i]; fpx = s['mark_fill'][i]
            ok = True if conf is None else bool(conf[d, i])
            m_next = MARKS[mark_ids[idx + 1]] if idx + 1 < len(mark_ids) else 1439
            seg = (mod_arr > m) & (mod_arr <= m_next)
            # exit management (taker) at marks
            if pos != 0:
                flip = has_band and ((pos > 0 and p < lb) or (pos < 0 and p > ub))
                if stop == 'vwap':
                    sl = (max(vw, lb) if pos > 0 else min(vw, ub)) if has_band else vw
                    hit = np.isfinite(vw) and ((pos > 0 and p < sl) or (pos < 0 and p > sl))
                else:
                    hit = False
                if (flip or hit) and np.isfinite(fpx):
                    trades[-1]['exit'] = fpx; trades[-1]['xmod'] = m; trades[-1]['reason'] = 'flip/stop'
                    pos = 0
            # Style C entry
            if pos == 0 and has_band and ok and np.isfinite(fpx):
                sig = 1 if p > ub else (-1 if p < lb else 0)
                if sig != 0:
                    ref = fpx
                    if sig > 0:
                        trig_px = ref * (1 + trig); lim_px = trig_px * (1 + moff)
                        confirmed = np.any(seg & (hi_arr >= trig_px))
                        filled = confirmed and np.any(seg & (hi_arr >= lim_px))
                    else:
                        trig_px = ref * (1 - trig); lim_px = trig_px * (1 - moff)
                        confirmed = np.any(seg & (lo_arr <= trig_px))
                        filled = confirmed and np.any(seg & (lo_arr <= lim_px))
                    if filled:
                        pos, entry_px, entry_mod = sig, lim_px, m
                        trades.append(dict(date=s['date'], side=sig, entry=lim_px,
                                           emod=m, _g=g, exit=np.nan, xmod=None,
                                           reason=None, maker=True))
        if pos != 0:
            trades[-1]['exit'] = s['exit_close']; trades[-1]['xmod'] = 1439
            trades[-1]['reason'] = 'EOD'
    return [t for t in trades if np.isfinite(t.get('exit', np.nan))]


def apply_cat_stop(t, stop_bps):
    """Continuous intrabar catastrophic stop from the ACTUAL fill price,
    conservative fills. Returns (raw_ret_bps, stopped)."""
    side, epx, emod, xmod = t['side'], t['entry'], t['emod'], t['xmod']
    stop_px = epx * (1 - stop_bps / 1e4) if side > 0 else epx * (1 + stop_bps / 1e4)
    g = t['_g']
    seg = g[(g['mod'] >= emod) & (g['mod'] < xmod)]
    for _, bar in seg.iterrows():
        pierced = (bar['low'] <= stop_px) if side > 0 else (bar['high'] >= stop_px)
        if pierced:
            fill = min(stop_px, bar['close']) if side > 0 else max(stop_px, bar['close'])
            return side * (fill / epx - 1) * 1e4, True
    return side * (t['exit'] / epx - 1) * 1e4, False


def process_asset(rec, cfg):
    base = rec['base']
    WM._set_grid('C')
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
            return dict(base=base, status='short')
        lev, pc = W.causal_lev_prevclose(sessions)
        sigmas = {L: W.causal_sigma(sessions, L) for L in WM.GRID['L']}
        confs = {K: (None if K == 0 else vol_profile_conf(sessions, K)) for K in WM.GRID['K']}
        df = df.copy(); df['mod'] = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute
        df['date'] = df['datetime'].dt.date
        minbars = {dt: g[['mod', 'low', 'high', 'close']] for dt, g in df.groupby('date')}
        dates = pd.to_datetime([s['date'] for s in sessions])
        n = len(sessions)

        combos = WM.all_combos()
        # walk-forward returns per combo (for selection only) via the maker engine
        rets = np.full((len(combos), n), np.nan)
        ntrs = np.zeros((len(combos), n), dtype=int)
        for ci, c in enumerate(combos):
            r, nt, _, _, _ = WM.run_combo_maker(
                sessions, minbars, sigmas[c['L']], lev, pc, c, confs[c['K']],
                cfg['maker_bps'], cfg['taker_bps'], 0.0, cfg['trigger_bps'],
                cfg['trigger_bps'], cfg['maker_offset_bps'])
            rets[ci], ntrs[ci] = r, nt

        SHAPE = WM.SHAPE
        def cidx(ci): return np.unravel_index(ci, SHAPE)
        def flat(t): return int(np.ravel_multi_index(t, SHAPE))

        sized_rows = []
        train_floor = max(50, int(cfg['train_days'] * 200 / 365))
        for q in WM.quarter_starts([s['date'] for s in sessions], cfg['train_days']):
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
                vals = [raw[ci]] + [raw[flat(t)] for t in WM.neighbors(cidx(ci))]
                vals = [v for v in vals if np.isfinite(v)]
                plateau[ci] = np.mean(vals)
            if not np.isfinite(plateau).any() or np.nanmax(plateau) <= 0:
                continue
            best = combos[int(np.nanargmax(plateau))]
            # reproduce Style C trades for the chosen combo in the OOS quarter,
            # apply cat stop + fixed-risk sizing
            oos_idx = list(np.where(oos_mask)[0])
            trades = trace_C_trades(sessions, minbars, sigmas[best['L']], lev, pc,
                                    best, confs[best['K']], cfg, oos_idx)
            stop_dist = cfg['cat_stop_bps'] / 1e4
            notional_frac = min(cfg['risk_frac'] / stop_dist, cfg['max_lev'])
            cost = cfg['taker_bps'] / 1e4
            maker = cfg['maker_bps'] / 1e4
            for t in trades:
                raw_bps, stopped = apply_cat_stop(t, cfg['cat_stop_bps'])
                gross = notional_frac * (raw_bps / 1e4)
                # entry earns maker rebate; exit pays taker
                entry_fee = -maker * notional_frac          # negative fee = credit
                exit_fee = cost * notional_frac
                eq_ret = gross - entry_fee - exit_fee
                sized_rows.append(dict(base=base, date=t['date'], side=t['side'],
                                       raw_bps=raw_bps, stopped=stopped,
                                       notional_frac=notional_frac, equity_ret=eq_ret,
                                       realized_risk=(-eq_ret if eq_ret < 0 else 0.0)))
        return dict(base=base, status='ok', sized=pd.DataFrame(sized_rows))
    except Exception:
        return dict(base=base, status='error', err=traceback.format_exc(-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ohlcv-dir', default=OHLCV_DIR)
    ap.add_argument('--funding-dir', default=FUNDING_DIR)
    ap.add_argument('--out', default=OUTPUT_DIR)
    ap.add_argument('--news-calendar', default=NEWS_CALENDAR)
    ap.add_argument('--data-attestation', default=DATA_ATTESTATION)
    ap.add_argument('--train-days', type=int, default=TRAIN_DAYS)
    ap.add_argument('--risk-frac', type=float, default=RISK_FRAC)
    ap.add_argument('--max-lev', type=float, default=MAX_LEV)
    ap.add_argument('--cat-stop-bps', type=float, default=CAT_STOP_BPS)
    ap.add_argument('--maker-bps', type=float, default=MAKER_BPS)
    ap.add_argument('--taker-bps', type=float, default=TAKER_BPS)
    ap.add_argument('--trigger-bps', type=float, default=TRIGGER_BPS)
    ap.add_argument('--maker-offset-bps', type=float, default=MAKER_OFFSET_BPS)
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    if not os.path.isdir(a.ohlcv_dir):
        raise SystemExit(f'ohlcv dir not found: {a.ohlcv_dir} (edit CONFIG or --ohlcv-dir)')
    if not a.data_attestation:
        raise SystemExit('--data-attestation required (name the dataset/venue)')
    news = None
    if a.news_calendar and str(a.news_calendar).lower() != 'none':
        news = set(pd.read_csv(a.news_calendar)['date'].astype(str))
    cfg = dict(train_days=a.train_days, risk_frac=a.risk_frac, max_lev=a.max_lev,
               cat_stop_bps=a.cat_stop_bps, maker_bps=a.maker_bps, taker_bps=a.taker_bps,
               trigger_bps=a.trigger_bps, maker_offset_bps=a.maker_offset_bps, news=news)
    nf = min(a.risk_frac / (a.cat_stop_bps / 1e4), a.max_lev)
    print(f'Style C sized | risk {a.risk_frac*100:.3f}%/trade | stop {a.cat_stop_bps}bps '
          f'| notional {nf*100:.3f}% | maker +{a.maker_bps}/taker -{a.taker_bps} '
          f'| trigger {a.trigger_bps} maker-off {a.maker_offset_bps}')
    from batch_run import discover
    fdir = a.funding_dir if os.path.isdir(a.funding_dir) else os.devnull
    universe, _ = discover(a.ohlcv_dir, fdir)
    os.makedirs(a.out, exist_ok=True)
    with Pool(a.workers) as pool:
        res = pool.map(partial(process_asset, cfg=cfg), universe)
    ok = [r for r in res if r['status'] == 'ok' and len(r.get('sized', []))]
    print(f'ok: {len(ok)}')
    if not ok:
        print('no sized trades'); return
    sz = pd.concat([r['sized'] for r in ok], ignore_index=True)
    sz.to_csv(os.path.join(a.out, 'sized_C_trades.csv'), index=False)

    sz['date'] = pd.to_datetime(sz['date'])
    daily = sz.groupby('date')['equity_ret'].sum().sort_index()
    sr = sharpe(daily.values)
    eq = (1 + daily).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    downside = daily[daily < 0].std(ddof=1)
    sortino = daily.mean() / downside * ANN if downside > 0 else np.nan
    years = (daily.index.max() - daily.index.min()).days / 365.25
    cagr = eq.iloc[-1] ** (1 / years) - 1
    print(f'\n===== STYLE C SIZED (deployable) =====')
    print(f'Sharpe {sr:+.2f} | t {daily.mean()/daily.std(ddof=1)*np.sqrt(len(daily)):+.2f} '
          f'| Sortino {sortino:+.2f} | Calmar {cagr/abs(dd):+.2f}')
    print(f'CAGR {cagr*100:+.2f}% | maxDD {dd*100:.2f}% | final {eq.iloc[-1]:.4f}x | {len(daily)} days')
    print(f'trades {len(sz)} | stopped {sz.stopped.mean()*100:.1f}% | win {(sz.equity_ret>0).mean()*100:.1f}%')
    rr = sz['realized_risk'] / (a.risk_frac)
    print(f'realized risk vs nominal: median {rr[rr>0].median():.2f}x | 99th {np.percentile(rr,99):.2f}x | max {rr.max():.2f}x')
    for y, g in daily.groupby(daily.index.year):
        if len(g) > 30:
            print(f'  {y}: Sharpe {sharpe(g.values):+.2f} | ret {g.sum()*100:+.1f}%')
    m = eq.resample('ME').last()
    pd.DataFrame({'month': m.index.strftime('%Y-%m'), 'equity': m.values}).to_csv(
        os.path.join(a.out, 'sized_C_monthly_equity.csv'), index=False)
    print(f'saved -> {a.out}/')


if __name__ == '__main__':
    main()
