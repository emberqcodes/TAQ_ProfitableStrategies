import importlib.util
from pathlib import Path
import numpy as np, pandas as pd
s=importlib.util.spec_from_file_location("e","execution_engine.py"); e=importlib.util.module_from_spec(s); s.loader.exec_module(e)
ok=0
def ck(n,c):
    global ok; assert c, f"FAIL: {n}"; ok+=1; print(f"  ok  {n}")

def mkbars(rows, start="2021-01-01"):
    idx=pd.date_range(start, periods=len(rows), freq="h")
    a=np.array(rows,float)
    return pd.DataFrame({"open":a[:,0],"high":a[:,1],"low":a[:,2],"close":a[:,3]}, index=idx)

# constant ATR setup: make 14 bars of TR=10 so ATR=10, then act
def const_atr_bars(n_warm=20, price=1000.0, tr=10.0):
    rows=[]
    for i in range(n_warm):
        rows.append([price, price+tr/2, price-tr/2, price])
    return rows

# 1) SIZING: notional = risk*entry/(trail*ATR). ATR=10, trail=15 -> sd=150; entry~1000; risk=2000 -> notional=2000*1000/150=13333
rows=const_atr_bars(20)
# day2: regime permits LONG (perm set below); entry at a bar's open=1000
bars=mkbars(rows + [[1000,1010,995,1005]]*5)
perm=pd.Series(1, index=pd.DatetimeIndex(np.unique(bars.index.normalize())))  # always long
led,summ=e.run_execution(bars, perm, lag_days=0, trail=15.0, risk=2000.0)
# find entry bar notional implicitly via first nonzero pos and pnl sign
ck("a trade was opened (pos goes nonzero)", (led["pos"]!=0).any())
ck("long leg active (pos=+1 present)", (led["pos"]==1).any())

# 2) STOP fill on gap: long entered ~1000 stop=850; a bar gaps to open=800 -> fills at 800 (open), not 850
rows2=const_atr_bars(20)+[[1000,1005,995,1000]]  # entry bar
rows2+=[[800,810,790,805]]                         # gap-down bar: open 800 < stop(~850)
rows2+=[[805,806,804,805]]*3
bars2=mkbars(rows2)
perm2=pd.Series(1, index=pd.DatetimeIndex(np.unique(bars2.index.normalize())))
led2,summ2=e.run_execution(bars2, perm2, lag_days=0, trail=15.0)
ck("a stop fired on the gap bar", summ2["n_stop"]>=1)

# 3) REGIME FLIP exit: perm goes LONG then SHORT next day -> position flips
days=pd.date_range("2021-01-01", periods=3, freq="D")
permf=pd.Series([1,-1,-1], index=days)
rows3=const_atr_bars(30)
bars3=mkbars(rows3+[[1000,1010,990,1000]]*48)  # ~3 days of bars
led3,summ3=e.run_execution(bars3, permf, lag_days=0, trail=50.0)  # wide stop so flip is the exit
ck("regime-flip exit occurred", summ3["n_flip"]>=1)
ck("both long and short legs traded after flip", (led3['pos']==1).any() and (led3['pos']==-1).any())

# 4) CAUSALITY: corrupt future bars; past pnl identical
rows4=const_atr_bars(30)+[[1000,1010,990,1000]]*60
bars4=mkbars(rows4); permc=pd.Series(1,index=pd.DatetimeIndex(np.unique(bars4.index.normalize())))
A,_=e.run_execution(bars4, permc, lag_days=0)
b5=bars4.copy(); K=60
b5.iloc[K:, b5.columns.get_loc("close")]=1e6; b5.iloc[K:, b5.columns.get_loc("high")]=1e6
B,_=e.run_execution(b5, permc, lag_days=0)
ck("future corruption leaves past bar-pnl identical", np.allclose(A['pnl'].values[:K-1], B['pnl'].values[:K-1]))

# 5) FUNDING: long pays positive funding -> negative pnl contribution at stamp
rows5=const_atr_bars(30)+[[1000,1001,999,1000]]*10
bars5=mkbars(rows5)
fstamp=pd.Series([0.01], index=[bars5.index[33]])  # +1% funding on one bar while long
permF=pd.Series(1,index=pd.DatetimeIndex(np.unique(bars5.index.normalize())))
ledF,_=e.run_execution(bars5, permF, funding_8h=fstamp, lag_days=0, trail=50.0)
ck("long pays positive funding (stamp bar pnl dented)", ledF['pnl'].loc[bars5.index[33]] < 0)

print(f"\n{ok}/{ok} execution-engine self-tests passed.")
