"""
07_current_book.py — worked example of the live decision, on real numbers.

Prints, for the 8h book:
  A. the 20 names for the latest month in universe.parquet (the book you would deploy), with the
     liquidity numbers that put them there, and the next 20 (the bench).
  B. one full rebalance, step by step, at the last grid time in the prediction file:
     prediction -> demean -> rank -> long/short legs -> inverse-ATR weights -> notional per name
     -> orders vs the previous period's book -> turnover and fees -> realised P&L of that period.
  C. the same for the 12 most recent grid times in compact form (to see how the book rolls).

Usage:  python 07_current_book.py [equity_usdt]     (default equity 100,000)
Reads:  data_processed/xs/universe.parquet, results/xs_predictions_h480_tr12.parquet (via 13's resolver)
Writes: results/current_book.md
"""
import sys, importlib
import numpy as np
import pandas as pd
import config as C

x13 = importlib.import_module("05_backtest")
EQ = float(sys.argv[1]) if len(sys.argv) > 1 else 100_000.0
N = C.XS_MAKER_N
HALF = C.XS_GROSS_LEVERAGE / 2.0
out = []
def md(s=""): out.append(s); print(s)
def tbl(df, nd=4):
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f": d[c] = d[c].round(nd)
    md("| " + " | ".join(map(str, d.columns)) + " |"); md("|" + "---|" * len(d.columns))
    for _, r in d.iterrows(): md("| " + " | ".join(map(str, r.values)) + " |")
    md()

# ---------------------------------------------------------------------------
# A. the deployable 20
# ---------------------------------------------------------------------------
U = pd.read_parquet(C.XS_DIR / "universe.parquet")
M = U["month"].max()
u = U[U.month == M].sort_values("liq_rank")
md(f"# Current book — worked example\n")
md(f"## A. Universe for {M:%Y-%m}  (ranked on the {C.XS_LIQ_WINDOW_DAYS} days before {M:%Y-%m-%d})\n")
md(f"Rule: median daily quote volume over the trailing {C.XS_LIQ_WINDOW_DAYS} days, computed only from days strictly before the 1st; "
   f"name must have >= {C.XS_MIN_HISTORY_DAYS} days of history and median >= {C.XS_MIN_DAILY_QUOTE_VOL/1e6:.0f}M USDT. Top {C.XS_BOOK_SIZE} = book, "
   f"ranks {C.XS_BOOK_SIZE+1}-{C.XS_TRAIN_POOL} = training pool only (never traded). Re-run on the 1st of every month.\n")
u2 = u[["liq_rank", "symbol", "trailing_qv", "in_book"]].copy()
u2["trailing_qv_musdt"] = (u2.pop("trailing_qv") / 1e6).round(1)
tbl(u2[u2.liq_rank <= C.XS_TRAIN_POOL], 1)
book_syms = u.loc[u.in_book, "symbol"].tolist()

# ---------------------------------------------------------------------------
# B. one rebalance, step by step
# ---------------------------------------------------------------------------
P = pd.read_parquet(x13.pred_file())
P = P[P.in_book].copy()
P["known_at_ts"] = pd.to_datetime(P["known_at_ts"], utc=True)
P = P[P["known_at_ts"].dt.hour % C.XS_GRID_HOURS == 0]
P = P[(P["known_at_ts"].dt.minute == 0)]
P["ret"] = P[f"exit_px_{C.XS_HORIZON}"] / P["entry_px"] - 1.0
grid = sorted(P["known_at_ts"].unique())
t_now, t_prev = grid[-1], grid[-2]

def leg_table(t):
    d = P[P.known_at_ts == t].set_index("symbol")
    d = d[~d.yhat.isna() & ~d.atr_pct.isna()]
    d["yhat_dm"] = d["yhat"] - d["yhat"].mean()
    d = d.sort_values("yhat_dm", ascending=False)
    d["rank"] = np.arange(1, len(d) + 1)
    d["leg"] = ""
    d.iloc[:N, d.columns.get_loc("leg")] = "LONG"
    d.iloc[-N:, d.columns.get_loc("leg")] = "SHORT"
    d["w"] = 0.0
    for leg, sgn in (("LONG", 1.0), ("SHORT", -1.0)):
        m = d.leg == leg
        inv = 1.0 / d.loc[m, "atr_pct"]
        d.loc[m, "w"] = sgn * HALF * inv / inv.sum()
    return d

d0, d1 = leg_table(t_prev), leg_table(t_now)
md(f"## B. Rebalance at {t_now:%Y-%m-%d %H:%M} UTC, equity {EQ:,.0f} USDT\n")
md("Step 1 — model output. `yhat` = predicted 8h return in ATR units for each of the 20 names (from the model trained on the 12 months before this month).")
md("Step 2 — demean: `yhat_dm = yhat − mean(yhat over the 20)`. Only the relative value matters.")
md(f"Step 3 — rank by yhat_dm. Top {N} = LONG, bottom {N} = SHORT, middle {20-2*N} = flat. No threshold.")
md(f"Step 4 — weights. Within a leg, `w_i = ±{HALF} × (1/atr_pct_i) / Σ(1/atr_pct_j)`; atr_pct = ATR14 on 15m bars ÷ price. Each leg sums to {HALF:.1f}× equity. Notional_i = w_i × equity.\n")
show = d1[["rank", "yhat", "yhat_dm", "atr_pct", "leg", "w"]].copy()
show["notional_usdt"] = (show["w"] * EQ).round(0)
show["entry_px"] = d1["entry_px"]
show.insert(0, "symbol", show.index)
tbl(show.reset_index(drop=True), 4)

md(f"Step 5 — orders. Previous book (from {t_prev:%H:%M}) vs new targets. Order_i = (w_new_i − w_old_i) × equity. Names that stay in the same leg are only topped up / trimmed, not re-traded.\n")
allsym = sorted(set(d0.index) | set(d1.index))
o = pd.DataFrame({"w_old": d0["w"].reindex(allsym).fillna(0.0), "w_new": d1["w"].reindex(allsym).fillna(0.0)})
o["delta_w"] = o.w_new - o.w_old
o["order_usdt"] = (o.delta_w * EQ).round(0)
o["side"] = np.where(o.delta_w > 0, "BUY", np.where(o.delta_w < 0, "SELL", "-"))
o = o[o.delta_w.abs() > 1e-12].sort_values("delta_w")
o.insert(0, "symbol", o.index)
tbl(o.reset_index(drop=True), 4)
turn = o.delta_w.abs().sum()
md(f"Turnover this rebalance = Σ|Δw| = {turn:.3f} of equity = {turn*EQ:,.0f} USDT traded.")
for name, bps in C.XS_COST_SCENARIOS.items():
    md(f"  fees if all {name} ({bps:+.1f} bps/side): {turn*bps/1e4*EQ:,.0f} USDT = {turn*bps/1e4*100:.3f}% of equity")
md()
md("Step 6 — execution. For each order: post-only limit at the 1-minute open price of the grid time, leave it 15 min; if not filled, cross as taker (IOC) at minute 15. Skip an order if its notional > 5% of the name's trailing 1-minute volume.")
md("Step 7 — hold 8 hours. Funding settlements inside the hold are paid/received per name: funding_i = w_i × rate_i (sign: a long pays positive funding).")
md("Step 8 — at the next grid time, repeat from Step 1. Nothing is closed explicitly; the next rebalance's orders are the difference.\n")

md("Realised P&L of this period (for the example only — in live trading this is what you see 8 hours later):")
d1["ret"] = P[P.known_at_ts == t_now].set_index("symbol")["ret"].reindex(d1.index)
d1["pnl_frac"] = d1["w"] * d1["ret"]
pl = d1[d1.leg != ""][["leg", "w", "entry_px", "ret", "pnl_frac"]].copy()
pl["exit_px"] = P[P.known_at_ts == t_now].set_index("symbol")[f"exit_px_{C.XS_HORIZON}"].reindex(pl.index)
pl["pnl_usdt"] = (pl.pnl_frac * EQ).round(0)
pl.insert(0, "symbol", pl.index)
tbl(pl.reset_index(drop=True)[["symbol", "leg", "w", "entry_px", "exit_px", "ret", "pnl_frac", "pnl_usdt"]], 5)
g = pl.pnl_frac.sum()
md(f"gross = Σ w_i × ret_i = {g:+.5f} of equity = {g*EQ:+,.0f} USDT;  net (maker_mix 1.5 bps) = {g - turn*1.5e-4:+.5f} = {(g - turn*1.5e-4)*EQ:+,.0f} USDT (funding not shown here).\n")

# ---------------------------------------------------------------------------
# C. last 12 rebalances, compact
# ---------------------------------------------------------------------------
md("## C. Last 12 rebalances (who was long / short)\n")
rows = []
w_prev = None
for t in grid[-12:]:
    d = leg_table(t)
    w = d["w"]
    turn = (w.reindex(allsym).fillna(0) - (w_prev.reindex(allsym).fillna(0) if w_prev is not None else 0)).abs().sum() if w_prev is not None else np.nan
    rows.append(dict(grid_time=f"{t:%m-%d %H:%M}", longs=" ".join(d[d.leg == "LONG"].index), shorts=" ".join(d[d.leg == "SHORT"].index), turnover=turn))
    w_prev = w
tbl(pd.DataFrame(rows), 3)
(C.RESULTS_DIR / "current_book.md").write_text("\n".join(out), encoding="utf-8")
print("wrote results/current_book.md")
