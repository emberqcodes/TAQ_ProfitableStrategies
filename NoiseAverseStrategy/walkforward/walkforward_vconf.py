"""Walk-forward grid search WITH volume-confirmation axis (wfv, Part 5 candidate).

FOR FRESH DATA ONLY. Part 4 Section 6 and the study's closing disposition
forbid running any further variant on the exhausted Binance USDT-perp
history. This script therefore REQUIRES --data-attestation, a short string
naming the dataset/venue, which is logged to the registry with the run.

Frozen grid (216 combos):
  interval {30, 60, 120 min} x band k {1.0, 1.5, 2.0} x stop {vwap, band}
  x noise lookback L {20, 60, 120} x volume confirmation K {0, 1.5, 2.0, 3.0}
  - K = 0 means gating fully disabled (the selector may honestly reject the
    filter). K > 0: entries permitted only when the 30-minute quote volume
    ending at the decision mark >= K x the trailing 14-session mean of the
    same window (>=10 valid priors). The 30-minute window is fixed for ALL
    intervals (matches vconf-v1 exactly; the interval-length alternative was
    considered and rejected ex-ante to avoid a hidden axis).
  - Part 2's entry-cap axis is REMOVED (volume gating subsumes churn control).

Walk-forward rules: identical to Part 2's frozen machine — calendar-quarter
refits, trailing --train-days window (default 365), >=30 train trades to
qualify, plateau-smoothed selection over grid neighbors, best score <= 0 ->
flat quarter, per-asset selection, stitched OOS pool judged as ONE trial.

Costs default to 4 bps/side and MUST be set to the target venue's actual
schedule via --fee-bps/--slip-bps at freeze time.

News blackout (--news-calendar): a CSV with a 'date' column (the audited
macro_calendar_2020_2026.csv). Sessions whose UTC date appears take ZERO
entries under every combo (K=0 included); bands, vol targets, and volume
profiles still update from those sessions' data. Fixed overlay, not a grid
axis — per the registered rule "when news is present, we do not trade."

Usage:
  python walkforward_vconf.py --ohlcv-dir X --funding-dir Y --out wfv_out \
      --data-attestation "OKX-perps-2020-2026" --news-calendar macro_calendar_2020_2026.csv \
      [--workers 8] [--train-days 365] [--fee-bps 2 --slip-bps 2]
"""
import argparse, json, os, datetime, traceback
from functools import partial
from multiprocessing import Pool
import numpy as np
import pandas as pd
from engine import load_bars, build_sessions, attach_funding, MARKS
import wf_engine as W

GRID = dict(interval=[30, 60, 120], k=[1.0, 1.5, 2.0],
            stop=['vwap', 'band'], L=[20, 60, 120], K=[0.0, 1.5, 2.0, 3.0])
AXES = list(GRID.keys())
SHAPE = tuple(len(GRID[a]) for a in AXES)
TRAIN_DAYS = 365
MIN_TRAIN_TRADES = 30
ANN = np.sqrt(365)


def all_combos():
    out = []
    for iv in GRID['interval']:
        for k in GRID['k']:
            for st in GRID['stop']:
                for L in GRID['L']:
                    for K in GRID['K']:
                        out.append(dict(interval=iv, k=k, stop=st, L=L, K=K))
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


def vol_profile_conf(sessions, K):
    """conf[d, i] for one K (>0): 30-min window quote volume >= K x trailing
    14-session mean of the same window (>=10 valid, strictly prior)."""
    n, nmk = len(sessions), len(MARKS)
    wv = np.full((n, nmk), np.nan)
    for d, s in enumerate(sessions):
        if s.get('valid'):
            wv[d] = s['mark_wvol']
    ok = np.zeros((n, nmk), dtype=bool)
    for d in range(n):
        w = wv[max(0, d - 14):d]
        if not len(w):
            continue
        cnt = np.isfinite(w).sum(axis=0)
        mu = np.nansum(w, axis=0) / np.maximum(cnt, 1)
        with np.errstate(invalid='ignore'):
            ok[d] = (cnt >= 10) & np.isfinite(wv[d]) & (wv[d] >= K * mu)
    return ok


def sharpe(x):
    x = x[np.isfinite(x)]
    if len(x) < 20 or x.std(ddof=1) == 0:
        return np.nan
    return float(x.mean() / x.std(ddof=1) * ANN)


def quarter_starts(dates, train_days):
    ds = pd.to_datetime(pd.Series(sorted(set(dates))))
    qs = sorted(set(pd.Timestamp(d.year, 3 * ((d.month - 1) // 3) + 1, 1)
                    for d in ds))
    return [q for q in qs if q > ds.min() + pd.Timedelta(days=train_days)]


def process_asset(rec, train_days=TRAIN_DAYS, cost=4e-4, news_dates=None):
    base = rec['base']
    min_needed = train_days + 90
    train_floor = max(50, int(train_days * 200 / 365))
    try:
        df = load_bars(rec['bars'])
        sessions = build_sessions(df)
        if rec['funding'] and os.path.exists(rec['funding']):
            sessions = attach_funding(sessions, rec['funding'])
        else:
            for s in sessions:
                s['funding_rate'] = np.nan
        n_valid = sum(1 for s in sessions if s.get('valid'))
        if n_valid == 0:
            return dict(base=base, status='no_valid_sessions',
                        err='no 13:30 anchors — 1-minute data?')
        if n_valid < min_needed:
            return dict(base=base, status='history_too_short',
                        err=f'{n_valid} valid sessions < {min_needed}')
        lev, pc = W.causal_lev_prevclose(sessions)
        sigmas = {L: W.causal_sigma(sessions, L) for L in GRID['L']}
        confs = {K: (None if K == 0 else vol_profile_conf(sessions, K))
                 for K in GRID['K']}
        if news_dates:
            blocked = np.array([str(s['date']) in news_dates for s in sessions])
            if blocked.any():
                nmk = len(MARKS)
                for K in GRID['K']:
                    c = confs[K]
                    if c is None:
                        c = np.ones((len(sessions), nmk), dtype=bool)
                    else:
                        c = c.copy()
                    c[blocked, :] = False
                    confs[K] = c
        dates = pd.to_datetime([s['date'] for s in sessions])
        n = len(sessions)

        combos = all_combos()
        rets = np.full((len(combos), n), np.nan)
        ntrs = np.zeros((len(combos), n), dtype=int)
        for ci, c in enumerate(combos):
            r, t, _ = W.run_combo(sessions, sigmas[c['L']], lev, pc,
                                  c['interval'], c['k'], 10 ** 9, c['stop'],
                                  cost=cost, conf=confs[c['K']])
            rets[ci], ntrs[ci] = r, t

        def cidx(ci):
            return np.unravel_index(ci, SHAPE)
        def flat(t):
            return int(np.ravel_multi_index(t, SHAPE))

        chosen, oos_rows, oos_trades = [], [], []
        for q in quarter_starts([s['date'] for s in sessions], train_days):
            q_end = q + pd.offsets.QuarterBegin(startingMonth=1)
            tr_mask = (dates >= q - pd.Timedelta(days=train_days)) & (dates < q)
            oos_mask = (dates >= q) & (dates < q_end)
            if oos_mask.sum() == 0 or tr_mask.sum() < train_floor:
                continue
            raw = np.array([sharpe(rets[ci][tr_mask])
                            if ntrs[ci][tr_mask].sum() >= MIN_TRAIN_TRADES
                            else np.nan for ci in range(len(combos))])
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
            chosen.append(dict(base=base, quarter=str(q.date()),
                               combo=json.dumps(c),
                               plateau_score=round(float(plateau[best]), 2)))
            _, _, trades = W.run_combo(sessions, sigmas[c['L']], lev, pc,
                                       c['interval'], c['k'], 10 ** 9,
                                       c['stop'], cost=cost,
                                       conf=confs[c['K']], collect_trades=True)
            qd = {pd.Timestamp(x) for x in dates[oos_mask]}
            for d_i in np.where(oos_mask)[0]:
                if np.isfinite(rets[best][d_i]):
                    oos_rows.append(dict(base=base, date=sessions[d_i]['date'],
                                         ret=float(rets[best][d_i])))
            for t in trades:
                if pd.Timestamp(t['date']) in qd:
                    t['base'] = base
                    t['combo'] = json.dumps(c)
                    oos_trades.append(t)
        return dict(base=base, status='ok', chosen=chosen,
                    oos=pd.DataFrame(oos_rows), trades=pd.DataFrame(oos_trades))
    except Exception:
        return dict(base=base, status='error', err=traceback.format_exc(-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ohlcv-dir', required=True)
    ap.add_argument('--funding-dir', required=True)
    ap.add_argument('--out', default='wfv_out')
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 4)
    ap.add_argument('--train-days', type=int, default=TRAIN_DAYS)
    ap.add_argument('--fee-bps', type=float, default=2.0)
    ap.add_argument('--slip-bps', type=float, default=2.0)
    ap.add_argument('--news-calendar', default=None,
                    help='CSV with a date column; those UTC dates take no entries')
    ap.add_argument('--data-attestation', required=True,
                    help='REQUIRED: name the dataset/venue (e.g. "OKX-perps"). '
                         'Running this on the exhausted Binance USDT-perp '
                         'history violates the Part 4/5 dispositions.')
    a = ap.parse_args()
    cost = (a.fee_bps + a.slip_bps) / 1e4
    print(f'DATA ATTESTATION: "{a.data_attestation}" | costs {a.fee_bps}+'
          f'{a.slip_bps} bps/side | train {a.train_days}d | grid 216 combos')
    print('Reminder: this machine is registered for FRESH data only.')
    from batch_run import discover
    universe, excluded = discover(a.ohlcv_dir, a.funding_dir)
    print(f'{len(universe)} in universe ({len(excluded)} tokenized excluded)')
    os.makedirs(a.out, exist_ok=True)

    news_dates = None
    if a.news_calendar:
        nd = pd.read_csv(a.news_calendar)
        news_dates = set(nd['date'].astype(str))
        print(f'news blackout ACTIVE: {len(news_dates)} calendar dates take no entries')
    with Pool(a.workers) as pool:
        results = pool.map(partial(process_asset, train_days=a.train_days,
                                   cost=cost, news_dates=news_dates), universe)
    ok = [r for r in results if r['status'] == 'ok' and len(r['oos'])]
    other = [r for r in results if not (r['status'] == 'ok' and
                                        r.get('oos') is not None and
                                        len(r.get('oos', [])))]
    print(f'\nok with OOS: {len(ok)} | other: {len(other)}')
    if other:
        pd.DataFrame([dict(base=r['base'], status=r['status'],
                           detail=r.get('err', '')) for r in other]) \
            .to_csv(os.path.join(a.out, 'wfv_diagnostics.csv'), index=False)
    if not ok:
        print('NO OOS RESULTS — see wfv_diagnostics.csv')
        return

    chosen = pd.DataFrame([c for r in ok for c in r['chosen']])
    oos = pd.concat([r['oos'] for r in ok], ignore_index=True)
    pool_tr = pd.concat([r['trades'] for r in ok], ignore_index=True)
    chosen.to_csv(os.path.join(a.out, 'wfv_chosen_combos.csv'), index=False)
    oos.to_csv(os.path.join(a.out, 'wfv_oos_returns.csv'), index=False)
    pool_tr.to_csv(os.path.join(a.out, 'wfv_trade_pool.csv'), index=False)
    per = oos.groupby('base')['ret'].agg(
        oos_sessions='size', oos_sharpe=lambda x: sharpe(x.values),
        oos_ann_ret=lambda x: x.mean() * 365).round(3).reset_index()
    per.to_csv(os.path.join(a.out, 'wfv_per_asset.csv'), index=False)

    port = oos.groupby('date')['ret'].mean()
    sr = sharpe(port.values)
    t = port.mean() / port.std(ddof=1) * np.sqrt(len(port))
    print('\n========== WFV OOS REPORT (one trial) ==========')
    print(f'pooled EW OOS: Sharpe {sr:+.2f} (t {t:+.2f}) | '
          f'ann {port.mean()*365*100:+.1f}% | {len(port)} sessions | '
          f'flat quarters {chosen["combo"].isna().mean()*100:.0f}%')
    pf = port.to_frame('r'); pf['y'] = pd.to_datetime(pf.index).year
    print('by year:', {int(y): round(sharpe(g.r.values), 2)
                       for y, g in pf.groupby('y') if len(g) > 30})
    sel = chosen.dropna(subset=['combo'])
    if len(sel):
        cc = sel['combo'].apply(json.loads)
        for ax in AXES:
            print(f'  {ax} marginals:',
                  dict(pd.Series([c[ax] for c in cc]).value_counts()))
        print('  (K marginals answer: does the selector independently adopt '
              'the volume filter on data it has never seen?)')
    with open(os.path.join(a.out, 'trial_registry.jsonl'), 'a') as f:
        f.write(json.dumps({'trial': 'wfv-news-v1' if a.news_calendar else 'wfv-v1',
                            'news_blackout': bool(a.news_calendar),
                            'attestation': a.data_attestation,
                            'grid': GRID, 'train_days': a.train_days,
                            'fee_bps': a.fee_bps, 'slip_bps': a.slip_bps,
                            'pooled_oos_sharpe': round(sr, 2),
                            'ts': datetime.datetime.now(datetime.UTC)
                            .isoformat()}) + '\n')


if __name__ == '__main__':
    main()
