"""
21_execution_engine.py -- hourly intrabar execution engine (reconstructed from the
EXECUTION_LAYER_WORK_SUMMARY spec). Daily Option-A regime -> 1h bars, immediate entry,
trailing ATR(14)x15 stop, regime-flip exit, after-stop next-day lockout, risk-based
sizing ($risk to initial stop, lev cap), T+2 lag, funding applied, honest dollar accounting.

Causality is mechanical: a bar's stop/ATR use info through the PRIOR bar; market orders
fill at the bar open; a gap through the stop fills at the bar open (honest worse fill).
"""
import numpy as np, pandas as pd

def atr_causal(h, l, c, n=14):
    pc = c.shift(1)
    tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean().shift(1)   # shift(1): use through prior bar

def run_execution(bars, perm_daily, funding_8h=None, *, atr_n=14, trail=15.0,
                  risk=2000.0, base=30000.0, max_lev=25.0, taker_bps=5, slip_bps=5,
                  lag_days=2):
    """
    bars: DataFrame[open,high,low,close] indexed by hourly UTC timestamp.
    perm_daily: Series of daily LONG/FLAT/SHORT (index = calendar date, pre-lag).
    funding_8h: Series of funding_rate at 8h stamps (index = timestamp), optional.
    Returns per-bar ledger DataFrame + summary stats.
    """
    o,h,l,c = bars["open"].values, bars["high"].values, bars["low"].values, bars["close"].values
    atr = atr_causal(bars["open"]*0+bars["high"], bars["low"], bars["close"], atr_n).values
    idx = bars.index
    day = idx.normalize()
    # T+2 lagged permission, mapped to each hourly bar by calendar day
    perm_lagged = perm_daily.shift(lag_days)
    permmap = perm_lagged.reindex(pd.DatetimeIndex(np.unique(day))).ffill()
    perm = pd.Series(day, index=idx).map(permmap).fillna(0).astype(int).values
    fund = pd.Series(0.0, index=idx)
    if funding_8h is not None:
        fund.loc[fund.index.isin(funding_8h.index)] = funding_8h.reindex(fund.index).fillna(0.0)
        fund = pd.Series(funding_8h.reindex(idx).fillna(0.0).values, index=idx)
    fund = fund.values
    cpt = (taker_bps+slip_bps)/1e4
    N = len(bars)

    pos=0; notional=0.0; entry=0.0; stop=np.nan; hw=np.nan; lock_day=None
    barpnl=np.zeros(N); realized=np.zeros(N); exit_reason=[None]*N; legdir=np.zeros(N)
    for t in range(1, N):
        pdir = perm[t]; pnl=0.0
        # ---- manage open position ----
        if pos!=0:
            # 1) regime-flip exit at this bar's open (permission changed vs position)
            if pdir != pos:
                fill=o[t]; pnl += pos*notional*(fill/c[t-1]-1.0); pnl -= cpt*notional
                realized[t]+=pos*notional*(fill/entry-1.0); exit_reason[t]="flip"
                pos=0; notional=0.0; stop=hw=np.nan
            else:
                # 2) trailing stop, intrabar; gap through stop fills at open
                if pos==1:
                    hw=max(hw,h[t]); 
                    if np.isfinite(atr[t]): stop=max(stop, hw-trail*atr[t])
                    if o[t]<=stop: fill=o[t]; hit=True
                    elif l[t]<=stop: fill=stop; hit=True
                    else: hit=False
                else:
                    hw=min(hw,l[t])
                    if np.isfinite(atr[t]): stop=min(stop, hw+trail*atr[t])
                    if o[t]>=stop: fill=o[t]; hit=True
                    elif h[t]>=stop: fill=stop; hit=True
                    else: hit=False
                if hit:
                    pnl += pos*notional*(fill/c[t-1]-1.0); pnl -= cpt*notional
                    realized[t]+=pos*notional*(fill/entry-1.0); exit_reason[t]="stop"
                    lock_day=day[t]; pos=0; notional=0.0; stop=hw=np.nan
                else:
                    pnl += pos*notional*(c[t]/c[t-1]-1.0)             # mark-to-market
                    if fund[t]!=0: pnl -= pos*notional*fund[t]        # long pays + funding
        # ---- entries (flat) ----
        if pos==0 and pdir!=0 and (lock_day is None or day[t]>lock_day):
            a=atr[t]
            if np.isfinite(a) and a>0:
                entry=o[t]; sd=trail*a
                notional=min(risk*entry/sd, max_lev*base)
                pos=pdir; stop=entry-pos*sd; hw=entry
                pnl += pos*notional*(c[t]/entry-1.0); pnl -= cpt*notional
                if fund[t]!=0: pnl -= pos*notional*fund[t]
        barpnl[t]=pnl; legdir[t]=pos
    led=pd.DataFrame({"pnl":barpnl,"realized":realized,"pos":legdir}, index=idx)
    eq = base + led["pnl"].cumsum()
    daily = led["pnl"].groupby(day).sum()
    dret = daily/base
    sharpe = dret.mean()/dret.std()*np.sqrt(365) if dret.std()>0 else np.nan
    dd = ((eq - eq.cummax())/base).min()
    long_pnl = led.loc[led["pos"].shift().fillna(0)>0,"pnl"].sum()
    short_pnl= led.loc[led["pos"].shift().fillna(0)<0,"pnl"].sum()
    summary=dict(sharpe=sharpe, total_pnl=led["pnl"].sum(), ret_pct=led["pnl"].sum()/base*100,
                 max_dd_pct=dd*100, long_pnl=long_pnl, short_pnl=short_pnl,
                 n_stop=exit_reason.count("stop"), n_flip=exit_reason.count("flip"))
    led.attrs["summary"]=summary
    return led, summary
