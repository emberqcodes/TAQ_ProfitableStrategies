"""
08_param_walkforward.py — nested (rolling) parameter selection for the 8h book, lookahead-free.

The model walk-forward (step 04) refits the model every month. The *book* parameters (names per side N,
model variant) were so far fixed over the whole history. This script re-selects them every month from the
trailing XS_PWF_LOOKBACK_MONTHS of realised out-of-sample results and applies the choice to the next month.
Because every candidate's month-m P&L comes from a model trained before m, and the choice for month m uses
only months < m, the resulting "adaptive" series is a fair out-of-sample track record of the *procedure*.

Candidates = XS_PWF_N (names per side) x every prediction file present in results/ for this horizon:
  xs_predictions_h480_tr12.parquet         (12-month train, identity on)
  xs_predictions_h480_tr12_nosym.parquet   (identity off)         if present
  xs_predictions_h480_tr6.parquet          (6-month train)        if present
Cost scenario for the selection and the report: XS_PWF_COST (default maker_mix, 1.5 bps/side).

Outputs: results/xs_pwf_report_h480.md, results/xs_pwf_equity_h480.png, results/xs_pwf_choices_h480.csv
"""
import importlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

x13 = importlib.import_module("05_backtest")
H = C.XS_HORIZON
HS = f"_h{H}" if H != 240 else ""
LOOK = getattr(C, "XS_PWF_LOOKBACK_MONTHS", 6)
NS = getattr(C, "XS_PWF_N", [3, 5, 7])
COST_NAME = getattr(C, "XS_PWF_COST", "maker_mix")
COST = C.XS_COST_SCENARIOS[COST_NAME]
MIN_IMPROVE = getattr(C, "XS_PWF_MIN_IMPROVE", 0.0)   # switch only if trailing Sharpe beats the incumbent by this much
out = []
def md(s=""): out.append(s); print(s)
def table(df, nd=4):
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f": d[c] = d[c].round(nd)
    md("| " + " | ".join(map(str, d.columns)) + " |"); md("|" + "---|" * len(d.columns))
    for _, r in d.iterrows(): md("| " + " | ".join(map(str, r.values)) + " |")
    md()

# ---------------------------------------------------------------------------
# candidate prediction files
# ---------------------------------------------------------------------------
variants = {}
TRM = getattr(C, "XS_TRAIN_MONTHS", C.TRAIN_MONTHS)
for tag, fn in [(f"tr{TRM}_sym", f"xs_predictions{HS}_tr{TRM}.parquet"), (f"tr{TRM}_nosym", f"xs_predictions{HS}_tr{TRM}_nosym.parquet"),
                ("tr6_sym", f"xs_predictions{HS}_tr6.parquet")]:
    if tag in variants or (C.RESULTS_DIR / fn) in variants.values():
        continue
    if (C.RESULTS_DIR / fn).exists():
        variants[tag] = C.RESULTS_DIR / fn
if not variants:
    raise SystemExit("no prediction files found for this horizon; run step 04 first")
print("model variants:", list(variants))

# ---------------------------------------------------------------------------
# per-period net for every candidate (same engine as step 05)
# ---------------------------------------------------------------------------
orig = x13.pred_file
series = {}
for tag, fp in variants.items():
    x13.pred_file = lambda name="xs_predictions", _fp=fp: _fp
    yh, ret, atr, fund = x13.load_panel()
    for n in NS:
        D = x13.run(yh, ret, atr, fund, n, 0.0, COST)
        series[(tag, n)] = D["net"]
        print(f"  {tag} N={n}: {len(D):,} periods, total {D['net'].sum():+.3f}")
x13.pred_file = orig
NET = pd.DataFrame(series).sort_index()
NET.columns = [f"{t}|N{n}" for t, n in NET.columns]
NET = NET.fillna(0.0)
month = NET.index.to_period("M")
months = sorted(month.unique())

def sharpe(x):
    d = x.groupby(x.index.floor("D")).sum()
    return float(d.mean() / d.std() * np.sqrt(365)) if len(d) > 5 and d.std() > 0 else np.nan

# ---------------------------------------------------------------------------
# rolling selection
# ---------------------------------------------------------------------------
choices, adaptive = [], []
incumbent = None
for i, m in enumerate(months):
    if i < LOOK:
        continue
    win = NET[(month >= months[i - LOOK]) & (month < m)]
    score = win.apply(sharpe)
    best = score.idxmax()
    if incumbent is not None and not np.isnan(score.get(incumbent, np.nan)) and score[best] - score[incumbent] < MIN_IMPROVE:
        best = incumbent
    incumbent = best
    seg = NET.loc[month == m, best]
    adaptive.append(seg)
    choices.append(dict(month=str(m), chosen=best, trailing_sharpe=round(float(score[best]), 2),
                        realised_month_net=round(float(seg.sum()), 4),
                        best_in_hindsight=NET.loc[month == m].sum().idxmax()))
ADAPT = pd.concat(adaptive)
CH = pd.DataFrame(choices)
CH.to_csv(C.RESULTS_DIR / f"xs_pwf_choices{HS}.csv", index=False)

# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
md(f"# Rolling parameter selection, {H//60}h book, cost = {COST_NAME} ({COST:+.1f} bps/side)\n")
md(f"Every month the (model variant, N) with the best trailing-{LOOK}-month Sharpe is used for the next month. "
   f"Candidates: {list(NET.columns)}. Evaluation window: {months[LOOK]} to {months[-1]} ({len(months) - LOOK} months).\n")
ev = NET[month >= months[LOOK]]
rows = [dict(candidate="ADAPTIVE (rolling choice)", total=ADAPT.sum(), sharpe=sharpe(ADAPT),
             max_dd=float((ADAPT.groupby(ADAPT.index.floor('D')).sum().cumsum() - ADAPT.groupby(ADAPT.index.floor('D')).sum().cumsum().cummax()).min()))]
for c in ev.columns:
    s = ev[c]; d = s.groupby(s.index.floor("D")).sum().cumsum()
    rows.append(dict(candidate=f"fixed {c}", total=s.sum(), sharpe=sharpe(s), max_dd=float((d - d.cummax()).min())))
md("## Adaptive vs every fixed choice, same evaluation window\n")
table(pd.DataFrame(rows).sort_values("sharpe", ascending=False))
md("If ADAPTIVE is not near the top, re-selecting parameters on a trailing window adds noise rather than value: "
   "keep the fixed reference setting.\n")
md("## How often the choice changed, and how often it was right\n")
md(f"switches: {(CH.chosen != CH.chosen.shift()).sum() - 1} in {len(CH)} months; "
   f"chosen == best-in-hindsight: {(CH.chosen == CH.best_in_hindsight).mean():.0%} of months; "
   f"share of months by choice: {CH.chosen.value_counts(normalize=True).round(2).to_dict()}\n")
md("## Per year\n")
yr = pd.DataFrame({"adaptive": ADAPT.groupby(ADAPT.index.year).sum()})
for c in ev.columns: yr[c] = ev[c].groupby(ev.index.year).sum()
yr.index.name = "year"
table(yr.reset_index())
md("## Month-by-month choices\n")
table(CH)
(C.RESULTS_DIR / f"xs_pwf_report{HS}.md").write_text("\n".join(out), encoding="utf-8")

plt.figure(figsize=(11, 5))
plt.plot(ADAPT.groupby(ADAPT.index.floor("D")).sum().cumsum(), label="adaptive", lw=2, color="k")
for c in ev.columns:
    plt.plot(ev[c].groupby(ev[c].index.floor("D")).sum().cumsum(), label=f"fixed {c}", lw=0.9, alpha=0.7)
plt.legend(fontsize=7); plt.grid(alpha=0.3); plt.title(f"{H//60}h book: rolling parameter choice vs fixed ({COST_NAME})")
plt.ylabel("cumulative net (fraction of equity)"); plt.tight_layout()
plt.savefig(C.RESULTS_DIR / f"xs_pwf_equity{HS}.png", dpi=130)
print("wrote", C.RESULTS_DIR / f"xs_pwf_report{HS}.md")
