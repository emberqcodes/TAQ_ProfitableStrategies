"""Anti-bias test suite. Must pass BEFORE any strategy result is believed."""
import numpy as np
import pandas as pd
from engine import (load_bars, build_sessions, causal_features, run_backtest,
                    attach_funding, P, SESSION_OPEN_MIN)

BARS = '/mnt/user-data/uploads/SOL_USDT_USDT.parquet'
FUND = '/mnt/user-data/uploads/SOL_USDT_USDT_funding.parquet'


def full_run(df):
    sessions = attach_funding(build_sessions(df), FUND)
    sigma, lev, pc = causal_features(sessions)
    trades, daily = run_backtest(sessions, sigma, lev, pc)
    return sessions, sigma, lev, trades, daily


def test_truncation(df, trades_full, seed=7, n_cuts=3):
    """Deleting all data after T must not change any trade completed before T."""
    rng = np.random.default_rng(seed)
    dates = sorted(df['date'].unique())
    ok = True
    for T in rng.choice(dates[200:-30], n_cuts, replace=False):
        sub = df[df['date'] <= T]
        _, _, _, tr_cut, _ = full_run(sub)
        a = trades_full[trades_full['date'] < T].reset_index(drop=True)
        b = tr_cut[tr_cut['date'] < T].reset_index(drop=True)
        same = a.shape == b.shape and np.allclose(
            a[['gross', 'fee', 'entry', 'exit', 'qty']].values,
            b[['gross', 'fee', 'entry', 'exit', 'qty']].values)
        print(f"  truncate at {T}: trades before T identical -> {same}")
        ok &= same
    return ok


def test_perturbation(df, sigma_full, lev_full, seed=11):
    """Perturbing session d's prices must not change bands/leverage for <= d,
    and MUST change them for d+1..d+lookback."""
    rng = np.random.default_rng(seed)
    dates = sorted(df[df['mod'] >= SESSION_OPEN_MIN]['date'].unique())
    d_idx = len(dates) // 2
    d_date = dates[d_idx]
    df2 = df.copy()
    mask = (df2['date'] == d_date) & (df2['mod'] >= SESSION_OPEN_MIN)
    noise = 1 + rng.normal(0, 0.01, mask.sum())
    for c in ['open', 'high', 'low', 'close']:
        df2.loc[mask, c] = df2.loc[mask, c] * noise
    sessions2 = attach_funding(build_sessions(df2), FUND)
    sigma2, lev2, _ = causal_features(sessions2)

    past_same = (np.allclose(np.nan_to_num(sigma_full[:d_idx + 1]),
                             np.nan_to_num(sigma2[:d_idx + 1])) and
                 np.allclose(np.nan_to_num(lev_full[:d_idx + 1]),
                             np.nan_to_num(lev2[:d_idx + 1])))
    fut = slice(d_idx + 1, d_idx + 1 + P['lookback'])
    future_diff = not np.allclose(np.nan_to_num(sigma_full[fut]),
                                  np.nan_to_num(sigma2[fut]))
    print(f"  perturbed {d_date}: features<=d unchanged -> {past_same}; "
          f"features d+1..d+{P['lookback']} changed -> {future_diff}")
    return past_same and future_diff


def test_shuffle_kill(df, seed=23):
    """Shuffle minute log-returns WITHIN each session: destroys intraday
    momentum, preserves daily vol. Net result should be ~ -costs."""
    rng = np.random.default_rng(seed)
    df2 = df.sort_values('timestamp').reset_index(drop=True).copy()
    close = df2['close'].values.copy()
    logret = np.diff(np.log(close), prepend=np.log(close[0]))
    in_sess = (df2['mod'] >= SESSION_OPEN_MIN).values
    for date, g in df2[in_sess].groupby('date'):
        pos = g.index.values
        if len(pos) < 10:
            continue
        seg = logret[pos[1:]].copy()          # keep first bar's level
        rng.shuffle(seg)
        logret[pos[1:]] = seg
    new_close = np.exp(np.cumsum(logret)) * close[0] / np.exp(logret[0])
    df2['close'] = new_close
    prev = np.roll(new_close, 1); prev[0] = new_close[0]
    df2['open'] = prev
    df2['high'] = np.maximum(df2['open'], df2['close'])
    df2['low'] = np.minimum(df2['open'], df2['close'])
    _, _, _, trades, daily = full_run(df2)
    tot_net = daily['net'].sum()
    tot_costs = daily['fees'].sum() - daily['funding'].sum()
    print(f"  shuffled-returns run: net P&L = {tot_net:,.0f} "
          f"(costs paid = {tot_costs:,.0f}, gross = {daily['gross'].sum():,.0f})")
    return tot_net < 0.25 * P['start_equity']


if __name__ == '__main__':
    df = load_bars(BARS)
    sessions, sigma, lev, trades, daily = full_run(df)
    print(f"full run: {len(daily)} sessions, {len(trades)} trades")
    print("[1] Truncation invariance")
    r1 = test_truncation(df, trades)
    print("[2] Future perturbation")
    r2 = test_perturbation(df, sigma, lev)
    print("[3] Shuffle-kill (signal destruction)")
    r3 = test_shuffle_kill(df)
    print(f"\nRESULT: truncation={r1} perturbation={r2} shuffle={r3}")
