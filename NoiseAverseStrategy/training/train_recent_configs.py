"""
Train Style C on the most recent N days and emit a deployable config per asset.

For each asset: take the most recent --train-days of data, run the SAME
plateau-smoothed grid selection the walk-forward uses (Style C), pick the best
combo, and write config_<SYMBOL>.json in the TaQuant import format.

IMPORTANT -- honest labeling: these configs are TRAINED on the recent window,
NOT validated on held-out data. Their genuine out-of-sample test is what
happens AFTER deployment (the forward test). Deploying them is the forward-test
freeze; do not treat the training-window fit as proof of edge.

Uses the identical selection logic as walkforward_maker.py (same grid, same
plateau smoothing, same flat-if-<=0 rule) so results are consistent with the
study. Style is fixed to C; offset params are frozen (not swept).

Config in the block below or via CLI. Reuses walkforward_maker for the engine.
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
OUTPUT_DIR  = 'trained_configs'
NEWS_CALENDAR = 'macro_calendar_2020_2026.csv'
TRAIN_DAYS  = 180
# frozen deployable params written into every config (not selected):
RISK_FRAC   = 0.0001
MAX_LEV     = 20
CAPITAL_USD = 100000
CAT_STOP_BPS = 400
TRIGGER_BPS = 5
MAKER_OFFSET_BPS = 3
MAKER_REBATE_BPS = 1
TAKER_BPS   = 2
MIN_PLATEAU = None      # e.g. 1.0 to only emit configs for higher-conviction assets; None = all
# =========================================================================

MIN_TRAIN_TRADES = 30


def select_combo_recent(sessions, minbars, sigmas, confs, lev, pc, dates,
                        train_days, cfg):
    """Run Style C over the grid on the most-recent train_days window; return
    the plateau-selected best combo, or None if nothing clears (flat)."""
    WM._set_grid('C')
    combos = WM.all_combos()
    last_date = dates.max()
    cutoff = last_date - pd.Timedelta(days=train_days)
    mask = (dates >= cutoff) & (dates <= last_date)
    if mask.sum() < max(50, int(train_days * 200 / 365)):
        return None, mask.sum()

    rets = np.full((len(combos), len(sessions)), np.nan)
    ntrs = np.zeros((len(combos), len(sessions)), dtype=int)
    for ci, c in enumerate(combos):
        r, nt, _, _, _ = WM.run_combo_maker(
            sessions, minbars, sigmas[c['L']], lev, pc, c, confs[c['K']],
            cfg['maker_bps'], cfg['taker_bps'], 0.0, cfg['trigger_bps'],
            cfg['trigger_bps'], cfg['maker_offset_bps'])
        rets[ci], ntrs[ci] = r, nt

    SHAPE = WM.SHAPE
    def cidx(ci): return np.unravel_index(ci, SHAPE)
    def flat(t): return int(np.ravel_multi_index(t, SHAPE))

    raw = np.array([WM.sharpe(rets[ci][mask])
                    if ntrs[ci][mask].sum() >= MIN_TRAIN_TRADES else np.nan
                    for ci in range(len(combos))])
    plateau = np.full(len(combos), np.nan)
    for ci in range(len(combos)):
        if not np.isfinite(raw[ci]):
            continue
        vals = [raw[ci]] + [raw[flat(t)] for t in WM.neighbors(cidx(ci))]
        vals = [v for v in vals if np.isfinite(v)]
        plateau[ci] = np.mean(vals)
    if not np.isfinite(plateau).any() or np.nanmax(plateau) <= 0:
        return None, mask.sum()
    best = int(np.nanargmax(plateau))
    return dict(combo=combos[best], plateau=float(plateau[best]),
                train_sessions=int(mask.sum())), mask.sum()


def process_asset(rec, cfg):
    base = rec['base']
    try:
        df = load_bars(rec['bars'])
        sessions = build_sessions(df)
        if rec['funding'] and os.path.exists(rec['funding']):
            sessions = attach_funding(sessions, rec['funding'])
        else:
            for s in sessions:
                s['funding_rate'] = np.nan
        n_valid = sum(1 for s in sessions if s.get('valid'))
        if n_valid < cfg['train_days'] * 0.6:
            return dict(base=base, status='short', n=n_valid)
        WM._set_grid('C')
        lev, pc = W.causal_lev_prevclose(sessions)
        sigmas = {L: W.causal_sigma(sessions, L) for L in WM.GRID['L']}
        confs = {K: (None if K == 0 else vol_profile_conf(sessions, K)) for K in WM.GRID['K']}
        df = df.copy(); df['mod'] = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute
        df['date'] = df['datetime'].dt.date
        minbars = {dt: g[['mod', 'low', 'high', 'close']] for dt, g in df.groupby('date')}
        dates = pd.to_datetime([s['date'] for s in sessions])

        picked, ntrain = select_combo_recent(sessions, minbars, sigmas, confs,
                                              lev, pc, dates, cfg['train_days'], cfg)
        if picked is None:
            return dict(base=base, status='flat', n=ntrain)
        return dict(base=base, status='ok', **picked)
    except Exception:
        return dict(base=base, status='error', err=traceback.format_exc(-1))


def write_config(sym, combo, cfg):
    return {
        "risk_frac": cfg['risk_frac'], "max_leverage": cfg['max_lev'],
        "capital_usd": cfg['capital_usd'], "cat_stop_bps": cfg['cat_stop_bps'],
        "trigger_bps": cfg['trigger_bps'], "maker_offset_bps": cfg['maker_offset_bps'],
        "maker_rebate_bps": cfg['maker_bps'], "taker_bps": cfg['taker_bps'],
        "anchor_utc_min": 810, "close_utc_min": 1439,
        "mark_interval_min": combo['interval'], "first_mark_min": 840,
        "noise_lookback": combo['L'], "min_obs": 10, "band_k": combo['k'],
        "vol_conf_k": combo['K'], "vol_window_min": 30, "stop_style": combo['stop'],
        "news_dates": cfg['news']
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ohlcv-dir', default=OHLCV_DIR)
    ap.add_argument('--funding-dir', default=FUNDING_DIR)
    ap.add_argument('--out', default=OUTPUT_DIR)
    ap.add_argument('--news-calendar', default=NEWS_CALENDAR)
    ap.add_argument('--train-days', type=int, default=TRAIN_DAYS)
    ap.add_argument('--risk-frac', type=float, default=RISK_FRAC)
    ap.add_argument('--max-lev', type=float, default=MAX_LEV)
    ap.add_argument('--capital-usd', type=float, default=CAPITAL_USD)
    ap.add_argument('--cat-stop-bps', type=float, default=CAT_STOP_BPS)
    ap.add_argument('--trigger-bps', type=float, default=TRIGGER_BPS)
    ap.add_argument('--maker-offset-bps', type=float, default=MAKER_OFFSET_BPS)
    ap.add_argument('--maker-bps', type=float, default=MAKER_REBATE_BPS)
    ap.add_argument('--taker-bps', type=float, default=TAKER_BPS)
    ap.add_argument('--min-plateau', type=float, default=MIN_PLATEAU)
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    if not os.path.isdir(a.ohlcv_dir):
        raise SystemExit(f'ohlcv dir not found: {a.ohlcv_dir} (edit CONFIG or --ohlcv-dir)')
    news = ''
    if a.news_calendar and str(a.news_calendar).lower() != 'none':
        news = ",".join(sorted(set(pd.read_csv(a.news_calendar)['date'].astype(str))))
    cfg = dict(train_days=a.train_days, risk_frac=a.risk_frac, max_lev=a.max_lev,
               capital_usd=a.capital_usd, cat_stop_bps=a.cat_stop_bps,
               trigger_bps=a.trigger_bps, maker_offset_bps=a.maker_offset_bps,
               maker_bps=a.maker_bps, taker_bps=a.taker_bps, news=news)
    print(f'Training Style C on most recent {a.train_days} days | grid {len(WM.all_combos()) if WM._set_grid("C") or True else 0} combos '
          f'| offset trigger {a.trigger_bps}/maker {a.maker_offset_bps}')
    from batch_run import discover
    fdir = a.funding_dir if os.path.isdir(a.funding_dir) else os.devnull
    universe, _ = discover(a.ohlcv_dir, fdir)
    print(f'{len(universe)} assets')
    os.makedirs(a.out, exist_ok=True)

    with Pool(a.workers) as pool:
        results = pool.map(partial(process_asset, cfg=cfg), universe)

    ok = [r for r in results if r['status'] == 'ok']
    flat = [r for r in results if r['status'] == 'flat']
    short = [r for r in results if r['status'] == 'short']
    err = [r for r in results if r['status'] == 'error']
    print(f'\nselected: {len(ok)} | flat (no combo cleared): {len(flat)} | '
          f'too-short history: {len(short)} | errors: {len(err)}')

    written, index_rows = 0, []
    for r in ok:
        if a.min_plateau is not None and r['plateau'] < a.min_plateau:
            continue
        sym = r['base'].upper()
        if not sym.endswith('USDT'):
            sym += 'USDT'
        c = r['combo']
        with open(os.path.join(a.out, f'config_{sym}.json'), 'w') as f:
            json.dump(write_config(sym, c, cfg), f, indent=2)
        written += 1
        index_rows.append(dict(symbol=sym, file=f'config_{sym}.json',
                               mark_interval_min=c['interval'], band_k=c['k'],
                               stop_style=c['stop'], noise_lookback=c['L'],
                               vol_conf_k=c['K'], plateau=round(r['plateau'], 2),
                               train_sessions=r['train_sessions']))
    idx = pd.DataFrame(index_rows).sort_values('symbol') if index_rows else pd.DataFrame()
    idx.to_csv(os.path.join(a.out, '_INDEX.csv'), index=False)
    with open(os.path.join(a.out, '_README.txt'), 'w') as f:
        f.write(f"""Per-asset TaQuant Style C configs -- TRAINED on most recent {a.train_days} days.
Generated {datetime.datetime.now(datetime.UTC).date()}.

{written} configs written{'' if a.min_plateau is None else f' (plateau >= {a.min_plateau})'}.

HONEST STATUS: these are TRAINED, not validated. They are fit on the recent
window; their real out-of-sample test is what happens after you deploy them
(the forward test). Do not treat the training fit as proof of edge.

Per instance before running: set capital_usd to match UI Initial Capital;
risk_frac 0.0001 = 0.01% (0.01 = 1%); commission = maker-entry/taker-exit.

_INDEX.csv lists each symbol, combo, plateau (conviction), and train_sessions.
""")
    print(f'wrote {written} configs -> {a.out}/  (+ _INDEX.csv, _README.txt)')
    if len(ok):
        pl = np.array([r['plateau'] for r in ok])
        print(f'plateau: >1.0 {(pl>1).sum()} | 0-1 {((pl>0)&(pl<=1)).sum()} | median {np.median(pl):.2f}')


if __name__ == '__main__':
    main()
