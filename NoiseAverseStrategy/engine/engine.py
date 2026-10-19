"""Intraday momentum backtest engine — spec_v1 (frozen).

Causality guarantees:
- Bands and leverage for session d are pure functions of sessions < d.
- All decisions signal on the bar CLOSING at a half-hour mark and fill at the
  open of the first bar AT/after the mark (<=5 min tolerance).
- Forced exit fills at the 23:59 bar close.
- Funding at 16:00 UTC applies to the position carried INTO the mark.
"""
import numpy as np
import pandas as pd

SESSION_OPEN_MIN = 13 * 60 + 30            # 13:30 UTC
MARKS = list(range(14 * 60, 23 * 60 + 31, 30))   # 14:00 .. 23:30 (20 marks)
EXIT_MIN = 23 * 60 + 59                    # 23:59 bar (closes at 24:00)
FUNDING_MIN = 16 * 60                      # 16:00 UTC funding timestamp

P = dict(
    lookback=14,          # sessions, noise area
    vol_lookback=14,      # sessions, sizing vol
    vol_target=0.02,      # per-session vol target
    max_leverage=4.0,
    fee_bps=5.0,          # taker, per side
    slip_bps=2.0,         # per side
    min_band_obs=10,      # min valid sessions in noise window
    fill_tolerance=5,     # minutes after mark to find a fill bar
    start_equity=100_000.0,
)


def load_bars(path):
    df = pd.read_parquet(path).sort_values('timestamp').reset_index(drop=True)
    df = df.drop_duplicates(subset='timestamp', keep='first')
    dt = df['datetime'].dt
    df['date'] = dt.date
    df['mod'] = dt.hour * 60 + dt.minute      # minute of day, UTC
    return df


def build_sessions(df):
    """Assemble per-session arrays. Purely descriptive — no strategy state."""
    nmk = len(MARKS)
    base = SESSION_OPEN_MIN
    idx = np.arange(base, 1440)
    rel_marks = [m - base for m in MARKS]
    sess = df[df['mod'] >= base]
    out = []
    for date, g in sess.groupby('date', sort=True):
        g2 = g.set_index('mod').reindex(idx)
        op_a = g2['open'].to_numpy()
        cl_a = g2['close'].to_numpy()
        hi_a = g2['high'].to_numpy()
        lo_a = g2['low'].to_numpy()
        vo_a = g2['volume'].to_numpy()
        if 'quote_volume' in g2.columns:
            qv_a = g2['quote_volume'].to_numpy()
        else:
            qv_a = (hi_a + lo_a + cl_a) / 3.0 * vo_a
        if not np.isfinite(op_a[0]):
            out.append(dict(date=date, valid=False))
            continue
        open_px = float(op_a[0])

        hlc3v = (hi_a + lo_a + cl_a) / 3.0 * vo_a
        cum_pv = np.nancumsum(hlc3v)
        cum_v = np.nancumsum(vo_a)

        mark_close = np.full(nmk, np.nan)
        mark_fill = np.full(nmk, np.nan)
        mark_vwap = np.full(nmk, np.nan)
        mark_wvol = np.full(nmk, np.nan)
        cum_qv = np.nancumsum(qv_a)
        tol = P['fill_tolerance']
        for i, rm in enumerate(rel_marks):
            lo = rm - 30
            mark_wvol[i] = cum_qv[rm - 1] - (cum_qv[lo - 1] if lo > 0 else 0.0)
            if np.isfinite(cl_a[rm - 1]):
                mark_close[i] = cl_a[rm - 1]
                mark_vwap[i] = cum_pv[rm - 1] / cum_v[rm - 1] if cum_v[rm - 1] > 0 else open_px
            win = op_a[rm:rm + tol + 1]
            fin = np.isfinite(win)
            if fin.any():
                mark_fill[i] = win[np.argmax(fin)]

        rf = FUNDING_MIN - base
        px16 = cl_a[rf - 1] if np.isfinite(cl_a[rf - 1]) else np.nan
        tail = cl_a[rel_marks[-1]:EXIT_MIN - base + 1]
        tail = tail[np.isfinite(tail)]
        exit_close = float(tail[-1]) if len(tail) else np.nan

        out.append(dict(date=date, valid=np.isfinite(exit_close), open=open_px,
                        mark_close=mark_close, mark_fill=mark_fill,
                        mark_vwap=mark_vwap, mark_wvol=mark_wvol,
                        px16=px16, exit_close=exit_close))
    return out


def causal_features(sessions):
    """Bands sigma and leverage for each session from PRIOR sessions only."""
    n, nmk = len(sessions), len(MARKS)
    absmove = np.full((n, nmk), np.nan)
    oc_ret = np.full(n, np.nan)
    prev_close = np.full(n, np.nan)
    last_close = np.nan
    for d, s in enumerate(sessions):
        if s.get('valid'):
            absmove[d] = np.abs(s['mark_close'] / s['open'] - 1.0)
            oc_ret[d] = s['exit_close'] / s['open'] - 1.0
            prev_close[d] = last_close
            last_close = s['exit_close']
        else:
            prev_close[d] = last_close

    sigma = np.full((n, nmk), np.nan)
    lev = np.full(n, np.nan)
    L, VL = P['lookback'], P['vol_lookback']
    for d in range(n):
        w = absmove[max(0, d - L):d]                      # strictly prior sessions
        if len(w):
            cnt = np.isfinite(w).sum(axis=0)
            mu = np.nansum(w, axis=0) / np.maximum(cnt, 1)  # cnt==0 -> discarded below
            sigma[d] = np.where(cnt >= P['min_band_obs'], mu, np.nan)
        r = oc_ret[max(0, d - VL):d]
        r = r[np.isfinite(r)]
        if len(r) >= VL:
            v = r.std(ddof=1)
            if v > 0:
                lev[d] = min(P['vol_target'] / v, P['max_leverage'])
    return sigma, lev, prev_close


def run_backtest(sessions, sigma, lev, prev_close, conf_ok=None):
    cost = (P['fee_bps'] + P['slip_bps']) / 1e4
    equity = P['start_equity']
    trades, daily = [], []

    for d, s in enumerate(sessions):
        if not s.get('valid') or not np.isfinite(lev[d]):
            continue
        op = s['open']
        pc = prev_close[d] if np.isfinite(prev_close[d]) else op
        hi_anchor, lo_anchor = max(op, pc), min(op, pc)
        notional = equity * lev[d]
        pos, qty, entry_px, entry_t = 0, 0.0, np.nan, None
        pnl_gross = fees = funding = 0.0
        funded = False

        def fill(i):
            return s['mark_fill'][i]

        def close_trade(px, tag, mark_i):
            nonlocal pnl_gross, fees, pos, qty
            gross = pos * qty * (px - entry_px)
            fee = cost * qty * (entry_px + px)
            pnl_gross += gross
            fees += fee
            trades.append(dict(date=s['date'], side=pos, qty=qty,
                               entry=entry_px, exit=px, entry_mark=entry_t,
                               exit_mark=tag, gross=gross, fee=fee))
            pos, qty = 0, 0.0

        for i, m in enumerate(MARKS):
            # funding on position carried into 16:00
            if not funded and m >= FUNDING_MIN:
                funded = True
                if pos != 0 and np.isfinite(s['px16']):
                    fr = s.get('funding_rate', np.nan)
                    if np.isfinite(fr):
                        funding += -pos * qty * s['px16'] * fr
            p = s['mark_close'][i]
            if not np.isfinite(p):
                continue
            sg = sigma[d, i]
            has_band = np.isfinite(sg)
            ub = hi_anchor * (1 + sg) if has_band else np.nan
            lb = lo_anchor * (1 - sg) if has_band else np.nan
            vw = s['mark_vwap'][i]
            fpx = fill(i)

            ok_entry = True if conf_ok is None else bool(conf_ok[d, i])
            if pos > 0:
                stop = max(vw, lb) if has_band else vw
                if has_band and p < lb and np.isfinite(fpx):
                    close_trade(fpx, m, i)
                    if ok_entry:
                        pos, qty, entry_px, entry_t = -1, notional / fpx, fpx, m
                elif np.isfinite(vw) and p < stop and np.isfinite(fpx):
                    close_trade(fpx, m, i)
            elif pos < 0:
                stop = min(vw, ub) if has_band else vw
                if has_band and p > ub and np.isfinite(fpx):
                    close_trade(fpx, m, i)
                    if ok_entry:
                        pos, qty, entry_px, entry_t = 1, notional / fpx, fpx, m
                elif np.isfinite(vw) and p > stop and np.isfinite(fpx):
                    close_trade(fpx, m, i)
            elif has_band and np.isfinite(fpx) and ok_entry:
                if p > ub:
                    pos, qty, entry_px, entry_t = 1, notional / fpx, fpx, m
                elif p < lb:
                    pos, qty, entry_px, entry_t = -1, notional / fpx, fpx, m

        if pos != 0:
            close_trade(s['exit_close'], 'EOD', None)

        pnl_net = pnl_gross - fees + funding
        equity += pnl_net
        daily.append(dict(date=s['date'], gross=pnl_gross, fees=fees,
                          funding=funding, net=pnl_net, equity=equity,
                          lev=lev[d]))
        if equity <= 0:
            break
    return pd.DataFrame(trades), pd.DataFrame(daily)


def attach_funding(sessions, funding_path):
    f = pd.read_parquet(funding_path)
    f = f[f['timestamp'].dt.hour == 16]
    fmap = dict(zip(f['timestamp'].dt.date, f['fundingRate']))
    for s in sessions:
        s['funding_rate'] = fmap.get(s['date'], np.nan)
    return sessions


def summarize(daily, trades, trading_days_per_year=365):
    r = daily['net'] / daily['equity'].shift(1).fillna(P['start_equity'])
    n = len(daily)
    yrs = n / trading_days_per_year
    eq = daily['equity']
    cagr = (eq.iloc[-1] / P['start_equity']) ** (1 / yrs) - 1
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(trading_days_per_year) if r.std() > 0 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    tr = trades.copy()
    tr['net_leg'] = tr['gross'] - tr['fee']
    wins = tr['net_leg'] > 0
    return dict(sessions=n, years=round(yrs, 2), trades=len(tr),
                cagr=round(cagr * 100, 2), sharpe=round(sharpe, 2),
                max_dd=round(dd * 100, 2),
                win_rate=round(wins.mean() * 100, 1),
                payoff=round(tr.loc[wins, 'net_leg'].mean() / -tr.loc[~wins, 'net_leg'].mean(), 2) if (~wins).any() else np.nan,
                gross_pnl=round(tr['gross'].sum(), 0),
                fees=round(daily['fees'].sum(), 0),
                funding=round(daily['funding'].sum(), 0),
                net_pnl=round(daily['net'].sum(), 0),
                exposure_pct=round((daily['gross'] != 0).mean() * 100, 1))
