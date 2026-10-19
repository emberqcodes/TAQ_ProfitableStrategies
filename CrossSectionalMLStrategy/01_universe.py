"""
Step 10 - Monthly liquidity universe, formed with no lookahead.

Scans every <SYMBOL>.parquet in RAW_OHLCV_DIR reading only timestamp + quote_volume, aggregates
to daily quote volume, and for each month start M ranks symbols by the MEDIAN daily quote volume
over the XS_LIQ_WINDOW_DAYS days ending the day before M. A symbol qualifies for month M if it
has >= XS_MIN_HISTORY_DAYS of data before M, its trailing median >= XS_MIN_DAILY_QUOTE_VOL, and
it is not in XS_EXCLUDE. Top XS_BOOK_SIZE -> in_book, top XS_TRAIN_POOL -> in_train.

Delisted names: a symbol whose data ends is simply absent from later months. Because the ranking
at M uses only data before M, a name that was liquid then and died later is included then.

Outputs:
  data_processed/xs/daily_qv.parquet      symbol, date, quote_vol
  data_processed/xs/universe.parquet      month, symbol, liq_rank, trailing_qv, in_book, in_train
  data_processed/xs/build_symbols.txt     symbols that are ever in_train (what step 02 builds)
  results/universe_report.txt
"""
import io
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import config as C

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows console safety

out = io.StringIO()


def p(*a):
    print(*a); print(*a, file=out)


files = sorted(C.RAW_OHLCV_DIR.glob("*.parquet"))
p(f"{len(files)} parquet files in {C.RAW_OHLCV_DIR}")

daily = []
first_last = []
for i, f in enumerate(files):
    sym = f.stem
    if sym in C.XS_EXCLUDE:
        continue
    try:
        t = pq.read_table(f, columns=["timestamp", "quote_volume"]).to_pandas()
    except Exception as e:  # noqa
        p(f"  !! {sym}: {e}"); continue
    if t.empty:
        continue
    ts = pd.to_datetime(t["timestamp"], utc=True)
    d = t.groupby(ts.dt.floor("D"))["quote_volume"].sum().rename("quote_vol").reset_index()
    d = d.rename(columns={"timestamp": "date"})
    d["symbol"] = sym
    daily.append(d)
    first_last.append(dict(symbol=sym, first=ts.min(), last=ts.max(), rows=len(t)))
    if (i + 1) % 50 == 0:
        print(f"  scanned {i + 1}/{len(files)}")

D = pd.concat(daily, ignore_index=True)
D.to_parquet(C.XS_DIR / "daily_qv.parquet", index=False)
FL = pd.DataFrame(first_last)
data_end = FL["last"].max()
FL["delisted"] = FL["last"] < data_end - pd.Timedelta(days=7)
odd = [s for s in FL.symbol if not s.isascii()]
p(f"non-ASCII symbol names: {odd}")
p(f"symbols with data: {len(FL)}   data end: {data_end}   delisted (ended >7d before end): {FL.delisted.sum()}")

# ---------------------------------------------------------------------------
# monthly ranking
# ---------------------------------------------------------------------------
piv = D.pivot(index="date", columns="symbol", values="quote_vol").sort_index()
months = pd.date_range(pd.Timestamp(C.XS_FIRST_MONTH, tz="UTC"), data_end.normalize(), freq="MS")
first_date = FL.set_index("symbol")["first"]

rows = []
for M in months:
    win = piv.loc[(piv.index >= M - pd.Timedelta(days=C.XS_LIQ_WINDOW_DAYS)) & (piv.index < M)]
    if win.empty:
        continue
    med = win.median(axis=0, skipna=True)
    ndays = win.notna().sum(axis=0)
    hist_ok = (M - first_date.reindex(med.index)).dt.days >= C.XS_MIN_HISTORY_DAYS
    ok = (med >= C.XS_MIN_DAILY_QUOTE_VOL) & (ndays >= C.XS_LIQ_WINDOW_DAYS * 0.8) & hist_ok
    ranked = med[ok].sort_values(ascending=False)
    for r, (sym, v) in enumerate(ranked.items(), start=1):
        rows.append(dict(month=M, symbol=sym, liq_rank=r, trailing_qv=v,
                         in_book=r <= C.XS_BOOK_SIZE, in_train=r <= C.XS_TRAIN_POOL))
U = pd.DataFrame(rows)
U.to_parquet(C.XS_DIR / "universe.parquet", index=False)

build = sorted(U.loc[U.in_train, "symbol"].unique())
(C.XS_DIR / "build_symbols.txt").write_text("\n".join(build), encoding="utf-8")

# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
p("")
p("qualifying symbols per month (eligible / in_train / in_book):")
g = U.groupby("month").agg(eligible=("symbol", "size"), train=("in_train", "sum"), book=("in_book", "sum"))
p(g.to_string())
p("")
p(f"symbols ever in_book : {U.loc[U.in_book, 'symbol'].nunique()}")
p(f"symbols ever in_train: {len(build)}  -> step 02 builds these")
p("")
book_delisted = U.loc[U.in_book, "symbol"].unique()
dl = FL.set_index("symbol").loc[book_delisted, "delisted"]
p(f"of the ever-in-book names, {dl.sum()} are delisted by the data end (survivorship coverage check)")
p("")
p("book composition, first month of each year:")
for M in months:
    if M.month == 1 or M == months[0]:
        b = U[(U.month == M) & U.in_book].sort_values("liq_rank")["symbol"].tolist()
        p(f"  {M.date()}: {', '.join(b)}")
p("")
p("monthly turnover of the book (names entering / leaving):")
prev = None
turn = []
for M, gm in U[U.in_book].groupby("month"):
    cur = set(gm["symbol"])
    if prev is not None:
        turn.append(dict(month=M.date(), enter=len(cur - prev), leave=len(prev - cur)))
    prev = cur
T = pd.DataFrame(turn)
p(f"  mean names changing per month: {T.enter.mean():.1f}   max: {T.enter.max()}")

(C.RESULTS_DIR / "universe_report.txt").write_text(out.getvalue(), encoding="utf-8")
print(f"\nwritten {C.XS_DIR / 'universe.parquet'} and {C.RESULTS_DIR / 'universe_report.txt'}")
