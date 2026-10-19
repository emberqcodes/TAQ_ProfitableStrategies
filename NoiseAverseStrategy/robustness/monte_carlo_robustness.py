"""Monte Carlo robustness suite — operates on REGISTERED OUTPUT FILES only.

Scope rule (bias protocol): this script quantifies the uncertainty of
already-run results. It never re-runs the strategy, never perturbs
parameters, never touches price data. Everything here is attribution-level.

Tests:
  1. Stationary block bootstrap (Politis-Romano) of pooled daily OOS returns
     -> Sharpe/annual-return confidence intervals, P(Sharpe <= 0)
  2. Null-centered bootstrap -> luck p-value (could zero-edge produce this?)
  3. Max-drawdown distribution vs realized (path-risk calibration)
  4. Concentration: Sharpe after removing the best k days; drop-one-month
     jackknife (the "October 2025 question")
  5. Drop-one-asset jackknife -> most influential assets on the pooled book
  6. Cost stress: extra per-trade slippage -> annual drag on the trade pool
  7. Probabilistic & Deflated Sharpe (Bailey & Lopez de Prado) with
     skew/kurtosis and a trial-count sensitivity sweep

Usage:
  python monte_carlo_robustness.py --oos wfv_oos_returns.csv \
      [--trades wfv_trade_pool.csv] [--book all|majors7|trio|ETH]
      [--n-boot 10000] [--block 10] [--seed 7]
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

ANN = np.sqrt(365)
M7 = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE', 'ADA']
BOOKS = {'all': None, 'majors7': M7, 'trio': ['BTC', 'ETH', 'SOL'],
         'ETH': ['ETH']}


def pooled_series(oos, book):
    sel = oos if BOOKS[book] is None else oos[oos.base.isin(BOOKS[book])]
    p = sel.groupby('date')['ret'].mean()
    p.index = pd.to_datetime(p.index)
    return p.dropna()


def sharpe(x):
    return float(np.mean(x) / np.std(x, ddof=1) * ANN)


def stationary_bootstrap(x, n_boot, mean_block, rng):
    """Politis-Romano stationary bootstrap. Returns (n_boot, T) matrix."""
    T = len(x)
    p = 1.0 / mean_block
    starts = rng.integers(0, T, size=(n_boot, T))
    cont = rng.random((n_boot, T)) >= p          # continue current block?
    idx = np.empty((n_boot, T), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for t in range(1, T):
        idx[:, t] = np.where(cont[:, t], (idx[:, t - 1] + 1) % T, starts[:, t])
    return x[idx]


def max_dd(paths):
    eq = np.cumprod(1 + paths, axis=1)
    peak = np.maximum.accumulate(eq, axis=1)
    return (eq / peak - 1).min(axis=1)


def psr_dsr(x, n_trials_list):
    """Probabilistic Sharpe vs 0 and Deflated Sharpe vs E[max of N nulls]."""
    T = len(x)
    sr = np.mean(x) / np.std(x, ddof=1)          # per-period
    g3, g4 = skew(x), kurtosis(x, fisher=False)
    denom = np.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    psr0 = norm.cdf(sr * np.sqrt(T - 1) / denom)
    out = {'PSR_vs_0': round(float(psr0), 3)}
    gamma = 0.5772156649
    sd_sr = np.sqrt((1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) / (T - 1))
    for N in n_trials_list:
        if N < 2:
            sr_star = 0.0
        else:
            sr_star = sd_sr * ((1 - gamma) * norm.ppf(1 - 1 / N)
                               + gamma * norm.ppf(1 - 1 / (N * np.e)))
        dsr = norm.cdf((sr - sr_star) * np.sqrt(T - 1) / denom)
        out[f'DSR_N={N}'] = round(float(dsr), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oos', required=True)
    ap.add_argument('--trades', default=None)
    ap.add_argument('--book', default='all', choices=list(BOOKS))
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--block', type=int, default=10,
                    help='mean bootstrap block length in days')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', default=None,
                    help='path for JSON report (default: mc_<book>.json)')
    a = ap.parse_args()
    report = {}
    rng = np.random.default_rng(a.seed)

    oos = pd.read_csv(a.oos)
    p = pooled_series(oos, a.book)
    x = p.values
    T = len(x)
    sr_real = sharpe(x)
    eq = np.cumprod(1 + x)
    dd_real = float((eq / np.maximum.accumulate(eq) - 1).min())
    report.update(book=a.book, sessions=T,
                  start=str(p.index.min().date()), end=str(p.index.max().date()),
                  realized_sharpe=round(sr_real, 3),
                  ann_ret_pct=round(float(np.mean(x) * 365 * 100), 2),
                  max_dd_pct=round(dd_real * 100, 2))
    print(f'book={a.book} | {T} sessions {p.index.min().date()}..{p.index.max().date()} '
          f'| realized Sharpe {sr_real:+.2f} | ann {np.mean(x)*365*100:+.1f}% '
          f'| max DD {dd_real*100:.1f}%')

    # 1 & 3: block bootstrap CIs and DD distribution
    B = stationary_bootstrap(x, a.n_boot, a.block, rng)
    srs = B.mean(axis=1) / B.std(axis=1, ddof=1) * ANN
    dds = max_dd(B)
    print(f'\n[1] block bootstrap ({a.n_boot} paths, mean block {a.block}d):')
    print(f'    Sharpe 90% CI [{np.percentile(srs,5):+.2f}, {np.percentile(srs,95):+.2f}] '
          f'| P(Sharpe<=0) = {float((srs<=0).mean()):.3f}')
    print(f'    ann ret 90% CI [{np.percentile(B.mean(axis=1)*365,5)*100:+.1f}%, '
          f'{np.percentile(B.mean(axis=1)*365,95)*100:+.1f}%]')
    print(f'[3] max-DD distribution: median {np.median(dds)*100:.1f}% | '
          f'5th pct {np.percentile(dds,5)*100:.1f}% | realized {dd_real*100:.1f}% '
          f'(realized at {float((dds<=dd_real).mean())*100:.0f}th pct of paths)')

    # 2: luck p-value under zero-edge null
    x0 = x - x.mean()
    B0 = stationary_bootstrap(x0, a.n_boot, a.block, rng)
    srs0 = B0.mean(axis=1) / B0.std(axis=1, ddof=1) * ANN
    pval = float((srs0 >= sr_real).mean())
    report.update(sharpe_ci90=[round(float(np.percentile(srs, 5)), 3),
                               round(float(np.percentile(srs, 95)), 3)],
                  p_sharpe_le_0=round(float((srs <= 0).mean()), 4),
                  dd_median_pct=round(float(np.median(dds)) * 100, 2),
                  dd_5pct_pct=round(float(np.percentile(dds, 5)) * 100, 2),
                  luck_pvalue=round(pval, 4))
    print(f'[2] luck test: P(zero-edge book shows Sharpe >= {sr_real:+.2f}) = {pval:.3f}')

    # 4: concentration + month jackknife
    print('[4] concentration:')
    order = np.argsort(x)[::-1]
    for k in (1, 5, 10, 20):
        xk = np.delete(x, order[:k])
        report[f'sharpe_ex_best_{k}_days'] = round(sharpe(xk), 3)
        print(f'    remove best {k:>2} days -> Sharpe {sharpe(xk):+.2f}')
    mo = p.groupby(p.index.to_period('M'))
    drops = {str(m): sharpe(p[p.index.to_period('M') != m].values) for m, _ in mo}
    worst_m = min(drops, key=drops.get); best_m = max(drops, key=drops.get)
    print(f'    drop-one-month jackknife: min {drops[worst_m]:+.2f} (without {worst_m}) '
          f'| max {drops[best_m]:+.2f} (without {best_m})')
    report.update(jackknife_month_min={worst_m: round(drops[worst_m], 3)},
                  jackknife_month_max={best_m: round(drops[best_m], 3)})

    # 5: drop-one-asset jackknife (skip for single-asset books)
    if BOOKS[a.book] is None or len(BOOKS[a.book]) > 1:
        bases = (oos if BOOKS[a.book] is None else
                 oos[oos.base.isin(BOOKS[a.book])])['base'].unique()
        infl = []
        sel = oos if BOOKS[a.book] is None else oos[oos.base.isin(BOOKS[a.book])]
        for b in bases:
            q = sel[sel.base != b].groupby('date')['ret'].mean().dropna()
            infl.append((b, sharpe(q.values) - sr_real))
        infl.sort(key=lambda z: z[1])
        print('[5] drop-one-asset (most positive influence first):')
        for b, d in infl[:3]:
            print(f'    without {b}: Sharpe change {d:+.3f}')
        for b, d in infl[-2:]:
            print(f'    without {b}: Sharpe change {d:+.3f}')

    # 6: cost stress on the trade pool
    if a.trades:
        tp = pd.read_csv(a.trades)
        if BOOKS[a.book] is not None:
            tp = tp[tp.base.isin(BOOKS[a.book])]
        sel = oos if BOOKS[a.book] is None else oos[oos.base.isin(BOOKS[a.book])]
        asset_sessions = max(1, len(sel))           # per-asset session count
        tr_per_sess = len(tp) / asset_sessions      # trades per ASSET-session
        print(f'[6] cost stress ({len(tp)} trades, {tr_per_sess:.2f}/asset-session):')
        for extra in (1.0, 2.0, 4.0):
            # extra one-way slippage on both legs, drawn half-normal, mean=extra
            draws = np.abs(rng.normal(0, extra * np.sqrt(np.pi / 2) * 1e-4,
                                      size=len(tp))) * 2   # round trip
            drag = draws.mean() * tr_per_sess * 365 * 100  # %/yr on equity approx
            net = tp['ret'].values - draws
            print(f'    +{extra:.0f} bps/side extra slippage: net/trade '
                  f'{tp.ret.mean()*1e4:+.1f} -> {net.mean()*1e4:+.1f} bps '
                  f'| ~{drag:.1f}%/yr drag')

    # 7: PSR / DSR
    print('[7] probabilistic & deflated Sharpe (skew %.2f, kurt %.1f):'
          % (skew(x), kurtosis(x, fisher=False)))
    dsr = psr_dsr(x, [1, 5, 10, 20, 50])
    report['psr_dsr'] = dsr
    report.update(skew=round(float(skew(x)), 3),
                  kurtosis=round(float(kurtosis(x, fisher=False)), 2))
    print('   ', dsr)
    import json as _json
    outp = a.out or f'mc_{a.book}.json'
    with open(outp, 'w') as f:
        _json.dump(report, f, indent=1)
    print(f'report saved -> {outp}')
    print('\nRead: [2] is the headline (luck p-value). [4] tells you how much is '
          'a few days. [7] DSR at your honest trial count is the number to cite.')


if __name__ == '__main__':
    main()
