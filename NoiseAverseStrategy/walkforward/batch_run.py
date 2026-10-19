"""Batch runner — frozen spec_v1 across a directory of perp 1m parquets.

Bias rules enforced here:
  * Universe filter is TYPE-based (tokenized assets), applied before any
    result exists. Use --dry-run to review the universe first.
  * Every symbol that runs is reported. Nothing is dropped for performing
    badly. Results CSV is sorted alphabetically, not by performance.
  * Multiple-testing: the report prints the expected MAX Sharpe among N
    symbols under the null of zero true edge (Bailey & Lopez de Prado).
    Any symbol below that line is indistinguishable from luck.
  * Every run is appended to the trial registry.

Usage (paths can be set in the DEFAULTS block below, or via CLI):
  python batch_run.py --ohlcv-dir /path/ohlcv --funding-dir /path/funding \
      --out results [--workers 8] [--dry-run] [--min-sessions 120] \
      [--exclude-extra FOO,BAR]

Both folders hold identically named files, e.g.
  <ohlcv-dir>/SOL_USDT_USDT.parquet     1m bars: timestamp, datetime, o,h,l,c,v
  <funding-dir>/SOL_USDT_USDT.parquet   funding: timestamp, fundingRate
Funding files are optional per symbol (treated as 0 with a warning).
"""
import argparse, json, hashlib, datetime, os, re, sys, traceback
from multiprocessing import Pool
import numpy as np
import pandas as pd
from engine import (load_bars, build_sessions, causal_features, run_backtest,
                    attach_funding, summarize, P)

# ------------------------- DEFAULTS (edit me) -------------------------
OHLCV_DIR = '/path/to/ohlcv'        # folder with <BASE>_USDT_USDT.parquet 1m bars
FUNDING_DIR = '/path/to/funding'    # folder with <BASE>_USDT_USDT.parquet funding
OUTPUT_DIR = 'batch_out'            # results folder
# ----------------------------------------------------------------------

# ---------------- universe filter (type-based, fixed ex-ante) ----------------
STOCK_TICKERS = {
    'AAPL', 'TSLA', 'NVDA', 'MSFT', 'META', 'AMZN', 'GOOGL', 'GOOG', 'COIN',
    'HOOD', 'MSTR', 'NFLX', 'AMD', 'INTC', 'PLTR', 'GME', 'CRCL', 'AVGO',
    'ORCL', 'TSM', 'BABA', 'SPY', 'QQQ', 'IBIT', 'SBET', 'AMC',
}
COMMODITY_FX_INDEX = {
    'PAXG', 'XAUT', 'XAU', 'XAG', 'XPT', 'XPD', 'USOIL', 'UKOIL', 'WTI',
    'SPX', 'US500', 'US30', 'US100', 'NDX', 'USTECH', 'DJI', 'DXY', 'TLT',
    'EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'CADUSD', 'NZD',
}


def is_tokenized(base: str) -> bool:
    b = base.upper()
    if b in COMMODITY_FX_INDEX or b in STOCK_TICKERS:
        return True
    if b.endswith('X') and b[:-1] in STOCK_TICKERS:      # AAPLX, TSLAX, ...
        return True
    return False


def discover(ohlcv_dir, funding_dir, exclude_extra=(), include_override=()):
    pat = re.compile(r'^(.+)_USDT_USDT\.parquet$')
    universe, excluded = [], []
    for fn in sorted(os.listdir(ohlcv_dir)):
        m = pat.match(fn)
        if not m:
            continue
        base = m.group(1)
        fund = os.path.join(funding_dir, fn)
        rec = dict(base=base, bars=os.path.join(ohlcv_dir, fn),
                   funding=fund if os.path.exists(fund) else None)
        if base.upper() in include_override:
            universe.append(rec)
        elif is_tokenized(base) or base.upper() in exclude_extra:
            excluded.append(base)
        else:
            universe.append(rec)
    return universe, excluded


# ---------------- per-symbol worker ----------------
def run_symbol(rec, min_sessions):
    base = rec['base']
    try:
        df = load_bars(rec['bars'])
        sessions = build_sessions(df)
        if rec['funding']:
            sessions = attach_funding(sessions, rec['funding'])
        else:
            for s in sessions:
                s['funding_rate'] = np.nan
        sigma, lev, pc = causal_features(sessions)
        trades, daily = run_backtest(sessions, sigma, lev, pc)
        if len(daily) < min_sessions:
            return dict(base=base, status='skipped_short_history',
                        sessions=len(daily))
        eq_prev = daily['equity'].shift(1).fillna(P['start_equity'])
        rnet = (daily['net'] / eq_prev)
        rgross = ((daily['gross'] + daily['funding']) / eq_prev)
        ann = np.sqrt(365)
        s = summarize(daily, trades)
        s.update(base=base, status='ok',
                 gross_sharpe=round(float(rgross.mean() / rgross.std(ddof=1) * ann), 2)
                 if rgross.std() > 0 else np.nan,
                 has_funding=rec['funding'] is not None,
                 first=str(daily['date'].iloc[0]), last=str(daily['date'].iloc[-1]))
        series = pd.DataFrame(dict(date=daily['date'], base=base,
                                   ret_net=rnet, ret_gross=rgross))
        return dict(base=base, status='ok', summary=s, series=series)
    except Exception:
        return dict(base=base, status='error', err=traceback.format_exc(-1))


# ---------------- multiple-testing benchmark ----------------
def expected_max_sharpe(n_trials, n_sessions_median):
    """E[max annualized SR] over n_trials independent zero-edge strategies
    observed for n_sessions_median sessions (Bailey & Lopez de Prado, 2014)."""
    from scipy.stats import norm
    if n_trials < 2:
        return np.nan
    gamma = 0.5772156649
    sd_ann = np.sqrt(365.0 / n_sessions_median)
    z1 = norm.ppf(1 - 1.0 / n_trials)
    z2 = norm.ppf(1 - 1.0 / (n_trials * np.e))
    return sd_ann * ((1 - gamma) * z1 + gamma * z2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ohlcv-dir', default=OHLCV_DIR)
    ap.add_argument('--funding-dir', default=FUNDING_DIR)
    ap.add_argument('--out', default=OUTPUT_DIR)
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 4)
    ap.add_argument('--min-sessions', type=int, default=120)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--exclude-extra', default='',
                    help='comma-separated extra bases to exclude (type-based only!)')
    ap.add_argument('--include-override', default='',
                    help='comma-separated bases to force-include despite filter')
    a = ap.parse_args()
    if not os.path.isdir(a.ohlcv_dir):
        sys.exit(f'ohlcv dir not found: {a.ohlcv_dir} '
                 f'(set OHLCV_DIR in the script or pass --ohlcv-dir)')
    if not os.path.isdir(a.funding_dir):
        print(f'WARNING: funding dir not found: {a.funding_dir} — '
              f'ALL funding treated as 0')
        a.funding_dir = os.devnull

    excl = {x.strip().upper() for x in a.exclude_extra.split(',') if x.strip()}
    incl = {x.strip().upper() for x in a.include_override.split(',') if x.strip()}
    universe, excluded = discover(a.ohlcv_dir, a.funding_dir, excl, incl)
    print(f'discovered {len(universe) + len(excluded)} symbols; '
          f'{len(excluded)} excluded as tokenized/RWA; {len(universe)} in universe')
    if excluded:
        print('excluded:', ', '.join(excluded))
    no_fund = [u['base'] for u in universe if not u['funding']]
    if no_fund:
        print(f'WARNING: {len(no_fund)} symbols missing funding files '
              f'(funding treated as 0): {", ".join(no_fund[:20])}'
              + (' ...' if len(no_fund) > 20 else ''))
    if a.dry_run:
        print('\n-- dry run: review the universe above, then re-run without '
              '--dry-run --')
        for u in universe:
            print(' ', u['base'])
        return

    os.makedirs(a.out, exist_ok=True)
    spec_hash = hashlib.sha256(json.dumps(P, sort_keys=True).encode()).hexdigest()[:12]
    with Pool(a.workers) as pool:
        results = pool.starmap(run_symbol,
                               [(u, a.min_sessions) for u in universe])

    ok = [r for r in results if r['status'] == 'ok']
    skipped = [r for r in results if r['status'] == 'skipped_short_history']
    errors = [r for r in results if r['status'] == 'error']

    res = pd.DataFrame([r['summary'] for r in ok]).sort_values('base')
    res.to_csv(os.path.join(a.out, 'batch_results.csv'), index=False)
    pd.concat([r['series'] for r in ok]).to_parquet(
        os.path.join(a.out, 'daily_returns.parquet'), index=False)
    with open(os.path.join(a.out, 'trial_registry.jsonl'), 'a') as f:
        ts = datetime.datetime.now(datetime.UTC).isoformat()
        for r in ok:
            f.write(json.dumps({'symbol': r['base'], 'spec': 'v1',
                                'hash': spec_hash,
                                'net_sharpe': r['summary']['sharpe'],
                                'gross_sharpe': r['summary']['gross_sharpe'],
                                'ts': ts}) + '\n')
    if skipped:
        pd.DataFrame(skipped).to_csv(os.path.join(a.out, 'skipped.csv'), index=False)
    if errors:
        pd.DataFrame(errors).to_csv(os.path.join(a.out, 'errors.csv'), index=False)

    # ---------------- aggregate report ----------------
    n = len(res)
    med_sessions = res['sessions'].median()
    try:
        emax = expected_max_sharpe(n, med_sessions)
    except ImportError:
        emax = np.nan
    print(f'\n================ AGGREGATE REPORT (spec {spec_hash}) ================')
    print(f'symbols run: {n} | skipped short history: {len(skipped)} | errors: {len(errors)}')
    print(f'median sessions/symbol: {med_sessions:.0f}')
    for col, name in [('sharpe', 'NET Sharpe'), ('gross_sharpe', 'GROSS Sharpe')]:
        v = res[col].dropna()
        print(f'{name:>13}: median {v.median():+.2f} | mean {v.mean():+.2f} | '
              f'%>0: {(v > 0).mean() * 100:.0f}% | best {v.max():+.2f} | worst {v.min():+.2f}')
    print(f'\nexpected MAX Sharpe among {n} zero-edge strategies: {emax:.2f}')
    print('=> any single symbol below that line is consistent with pure luck.')
    print('=> (crypto returns are cross-correlated, so effective N is lower and')
    print('   the true luck threshold is somewhat below this number.)')

    # pooled equal-weight portfolio (the statistic that matters most)
    allr = pd.concat([r['series'] for r in ok])
    for col, name in [('ret_net', 'NET'), ('ret_gross', 'GROSS')]:
        port = allr.groupby('date')[col].mean()
        if port.std() > 0:
            sr = port.mean() / port.std(ddof=1) * np.sqrt(365)
            print(f'pooled EW portfolio {name:>5} Sharpe: {sr:+.2f} '
                  f'({len(port)} sessions)')
    print('\nCaveats to carry into any conclusion:')
    print(' * survivorship: this universe is today\'s listings; delisted perps absent.')
    print(' * uniform 7bps/side is optimistic for illiquid alts; their results are flattered.')
    print(' * this is ONE pre-registered spec. No parameter was tuned on these results,')
    print('   and none may be without a train/test split going forward.')


if __name__ == '__main__':
    main()
