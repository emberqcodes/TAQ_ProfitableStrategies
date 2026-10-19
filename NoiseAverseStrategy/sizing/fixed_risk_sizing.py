"""Fixed-fractional risk sizing — measurement overlay.

Sizing rule (frozen, risk-budget parameter, NOT fitted):
    stop_distance = catastrophic stop level as a fraction (e.g. 0.04 = 400 bps)
    notional      = min( Equity * RISK_FRAC / stop_distance , Equity * MAX_LEV )
    RISK_FRAC = 0.0001 (0.01% of equity risked per trade)
    MAX_LEV   = 20

Fill realism (option a): the catastrophic stop uses CONSERVATIVE fills (worse
of {stop, bar close} on gap-through). Sizing is computed off the NOMINAL stop
distance, so gap-through trades realize losses > RISK_FRAC. We report that
realized-risk distribution rather than assuming perfect fills -- this is the
honest tail, matching live behaviour.

Scope: measurement on the frozen combo to seed a forward pre-registration.
Reports before/after sizing on Sharpe, P&L concentration (does equalizing
trade size flatten the fat right tail that is the edge?), and the realized
risk-per-trade distribution. NOT an optimizer; RISK_FRAC is a risk-budget
choice, not tuned on P&L.

Usage (CONFIG block or CLI):
  python fixed_risk_sizing.py --ohlcv-dir X --funding-dir Y --out sz_out \
      --book majors7 --news-calendar macro_calendar_2020_2026.csv \
      --stop-bps 400 --risk-frac 0.0001 --max-lev 20
"""
import argparse, json, os
import numpy as np
import pandas as pd
from engine import load_bars, build_sessions, attach_funding, MARKS
import wf_engine as W
from walkforward_vconf import vol_profile_conf
from catastrophic_stop import trace_with_paths, apply_stop

OHLCV_DIR = '/path/to/ohlcv'
FUNDING_DIR = '/path/to/funding'
OUTPUT_DIR = 'sz_out'
NEWS_CALENDAR = 'macro_calendar_2020_2026.csv'
BOOK = 'majors7'
COMBO = '{"interval":120,"k":1.0,"stop":"band","L":60,"K":2.0}'
STOP_BPS = 400
RISK_FRAC = 0.0001      # DECIMAL fraction, NOT percent.
                        #   0.0001 -> 0.01% risk (0.25% notional at 400bps stop)
                        #   0.001  -> 0.1%       (2.5% notional)
                        #   0.01   -> 1%         (25% notional) <- made the majors7 CSV
                        #   1.0    -> 100%       (do NOT)
MAX_LEV = 20.0
COST_BPS = 4.0          # per side

M7 = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE', 'ADA']
BOOKS = {'all': None, 'majors7': M7, 'trio': ['BTC', 'ETH', 'SOL'], 'ETH': ['ETH']}
ANN = np.sqrt(365)


def sized_returns(trades, stop_bps, risk_frac, max_lev, cost_bps):
    """Per-trade equity-fraction P&L under fixed-risk sizing + conservative stop.

    notional_frac = min(risk_frac / stop_dist, max_lev)
    equity P&L    = notional_frac * (raw_return_bps/1e4) - cost
    Returns DataFrame: date, base, pnl_bps_raw, stopped, notional_frac,
    equity_ret (fraction of equity), realized_risk (loss as frac of equity).
    """
    stop_dist = stop_bps / 1e4
    notional_frac = min(risk_frac / stop_dist, max_lev)
    cost = cost_bps / 1e4
    rows = []
    for t in trades:
        raw, stopped = apply_stop(t, stop_bps)          # bps, conservative fill
        gross_ret = notional_frac * (raw / 1e4)
        fee = notional_frac * cost * 2                   # both legs on notional
        eq_ret = gross_ret - fee
        rows.append(dict(date=t['_s']['date'], base=t.get('base', '?'),
                         pnl_bps_raw=raw, stopped=stopped,
                         notional_frac=notional_frac, equity_ret=eq_ret,
                         realized_risk=(-eq_ret if eq_ret < 0 else 0.0)))
    return pd.DataFrame(rows)


def stats(daily):
    r = daily.values
    sr = r.mean() / r.std(ddof=1) * ANN if r.std() > 0 else np.nan
    eq = np.cumprod(1 + r)
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    return sr, r.mean() * 365 * 100, dd * 100


def _process_one(args):
    """Worker: trace one asset and apply stop+sizing, returning light rows only
    (never ships the 1-minute bar frames back across the process boundary)."""
    u, combo, news, stop_bps, risk_frac, max_lev, cost_bps = args
    df = load_bars(u['bars'])
    sessions = build_sessions(df)
    if u['funding'] and os.path.exists(u['funding']):
        sessions = attach_funding(sessions, u['funding'])
    else:
        for s in sessions:
            s['funding_rate'] = np.nan
    conf = None if combo['K'] == 0 else vol_profile_conf(sessions, combo['K'])
    tt = trace_with_paths(df, sessions, combo, conf, news)
    for t in tt:
        t['base'] = u['base']
    part = sized_returns(tt, stop_bps, risk_frac, max_lev, cost_bps)
    return part


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ohlcv-dir', default=OHLCV_DIR)
    ap.add_argument('--funding-dir', default=FUNDING_DIR)
    ap.add_argument('--out', default=OUTPUT_DIR)
    ap.add_argument('--book', default=BOOK, choices=list(BOOKS))
    ap.add_argument('--news-calendar', default=NEWS_CALENDAR)
    ap.add_argument('--combo', default=COMBO)
    ap.add_argument('--stop-bps', type=int, default=STOP_BPS)
    ap.add_argument('--risk-frac', type=float, default=RISK_FRAC)
    ap.add_argument('--max-lev', type=float, default=MAX_LEV)
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    if not os.path.isdir(a.ohlcv_dir):
        raise SystemExit(f'ohlcv dir not found: {a.ohlcv_dir} (edit CONFIG or --ohlcv-dir)')
    news = None
    if a.news_calendar and str(a.news_calendar).lower() != 'none':
        news = set(pd.read_csv(a.news_calendar)['date'].astype(str))
    from batch_run import discover
    fdir = a.funding_dir if os.path.isdir(a.funding_dir) else os.devnull
    universe, _ = discover(a.ohlcv_dir, fdir)
    keep = BOOKS[a.book]
    if keep is not None:
        universe = [u for u in universe if u['base'] in keep]
    combo = json.loads(a.combo)
    os.makedirs(a.out, exist_ok=True)

    jobs = [(u, combo, news, a.stop_bps, a.risk_frac, a.max_lev, COST_BPS)
            for u in universe]
    print(f"processing {len(jobs)} assets on {a.workers} workers...")
    from multiprocessing import Pool
    if a.workers > 1:
        with Pool(a.workers) as pool:
            parts = pool.map(_process_one, jobs)
    else:
        parts = [_process_one(j) for j in jobs]
    sz = pd.concat([p for p in parts if len(p)], ignore_index=True)
    sz.to_csv(os.path.join(a.out, f'sized_trades_{a.book}.csv'), index=False)
    nf = min(a.risk_frac / (a.stop_bps / 1e4), a.max_lev)
    print(f"book={a.book} | {len(sz)} trades | stop {a.stop_bps}bps | "
          f"risk/trade {a.risk_frac*100:.3f}% | notional/trade {nf*100:.3f}% of equity "
          f"| lev cap {a.max_lev:.0f}x ({'BINDS' if a.risk_frac/(a.stop_bps/1e4) >= a.max_lev else 'not binding'})")

    # pooled daily equity returns (sum concurrent positions per session-date)
    daily = sz.groupby('date')['equity_ret'].sum()
    sr, ann, dd = stats(daily)
    print(f"\nSIZED book: Sharpe {sr:+.2f} | ann {ann:+.2f}% | max DD {dd:.1f}% "
          f"| {len(daily)} session-dates")

    # concentration: does fixed-risk flatten the fat tail?
    # Measure gross positive P&L share (robust to net sign) + Gini-style read.
    pnl = sz['equity_ret'].values
    pos = np.sort(pnl[pnl > 0])[::-1]
    pos_tot = pos.sum()
    if pos_tot > 0:
        conc = {k: round(pos[:k].sum() / pos_tot * 100, 0) for k in (5, 10, 20, 50)
                if k <= len(pos)}
        print(f"P&L concentration (top-k share of GROSS POSITIVE P&L): {conc}")
        print("  (vol-target baseline was ~96% of NET in top 50 trades; here we")
        print("   show share of gross wins -- lower top-k share = more even sizing)")
    else:
        print("P&L concentration: no positive trades in this slice")

    # realized risk-per-trade distribution (option a: conservative fills exceed nominal)
    rr = sz['realized_risk'].values * 1e4  # in bps of equity
    nominal = a.risk_frac * 1e4
    over = (sz['realized_risk'] > a.risk_frac * 1.001).mean() * 100
    print(f"\nrealized risk/trade (bps of equity): nominal {nominal:.2f} | "
          f"median {np.median(rr):.2f} | 95th {np.percentile(rr,95):.2f} | "
          f"99th {np.percentile(rr,99):.2f} | max {rr.max():.2f}")
    print(f"  trades exceeding nominal risk (gap-through): {over:.1f}% "
          f"| worst = {rr.max()/nominal:.1f}x nominal")
    print(f"  stopped trades: {sz['stopped'].mean()*100:.1f}%")
    print("\nRead: judge on Sharpe + concentration shift + the realized-risk TAIL, "
          "not on annual return magnitude (that's just the risk dial). Fixed-risk "
          "is a risk-budget choice; do not tune risk-frac on these numbers.")
    print(f"saved -> {os.path.join(a.out, f'sized_trades_{a.book}.csv')}")


if __name__ == '__main__':
    main()
