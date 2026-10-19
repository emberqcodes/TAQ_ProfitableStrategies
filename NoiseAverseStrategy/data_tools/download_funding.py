"""Download and organize Binance USD-M funding-rate data.

Symbols and date ranges are DERIVED FROM YOUR KLINES TREE, so funding
coverage matches your price data exactly — no silent gaps.

Sources:
  1. Binance Vision monthly zips (historical):
     https://data.binance.vision/data/futures/um/monthly/fundingRate/<SYM>/<SYM>-fundingRate-YYYY-MM.zip
  2. Optional --api-topup: current (unpublished) month via the public
     fapi endpoint, written as a converter-compatible zip.

Output layout mirrors yours:  <out-dir>/<SYMBOL>/<SYMBOL>-fundingRate-YYYY-MM.zip
Feed the result to binance_convert.py (same tree or its own tree).

Audit semantics (bias-relevant): a 404 BEFORE a symbol's first kline month
is benign (pre-listing). A 404 in the MIDDLE of the range is a hole that
would silently zero funding in backtests — flagged loudly.

Usage:
  python download_funding.py --klines-dir <...>/binance_ohlcv/1m/zips \
      --out-dir <...>/binance_funding/zips [--workers 8] [--api-topup]
"""
import argparse, io, json, os, re, time, zipfile
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
import datetime as dt
import pandas as pd

KLINE_RE = re.compile(r'^([A-Z0-9]+)-1m-(\d{4}-\d{2})(-\d{2})?\.zip$')
VISION = ('https://data.binance.vision/data/futures/um/monthly/'
          'fundingRate/{sym}/{sym}-fundingRate-{month}.zip')
FAPI = ('https://fapi.binance.com/fapi/v1/fundingRate'
        '?symbol={sym}&startTime={start}&limit=1000')


def discover_symbol_ranges(klines_dir):
    out = {}
    for root, _, files in os.walk(klines_dir):
        for fn in files:
            m = KLINE_RE.match(fn)
            if m:
                out.setdefault(m.group(1), set()).add(m.group(2))
    return {s: (min(ms), max(ms)) for s, ms in out.items()}


def month_range(first, last):
    y, mo = map(int, first.split('-'))
    y2, m2 = map(int, last.split('-'))
    ms = []
    while (y, mo) <= (y2, m2):
        ms.append(f'{y:04d}-{mo:02d}')
        mo += 1
        if mo == 13:
            y, mo = y + 1, 1
    return ms


def fetch(url, retries=3, timeout=30):
    for i in range(retries):
        try:
            with urlopen(Request(url, headers={'User-Agent': 'funding-dl'}),
                         timeout=timeout) as r:
                return r.read(), None
        except HTTPError as e:
            if e.code == 404:
                return None, 404
            time.sleep(1.5 * (i + 1))
            err = e.code
        except URLError as e:
            time.sleep(1.5 * (i + 1))
            err = str(e)
    return None, err


def download_symbol(sym, first, last, out_dir, api_topup):
    os.makedirs(os.path.join(out_dir, sym), exist_ok=True)
    now_month = dt.datetime.now(dt.UTC).strftime('%Y-%m')
    months = month_range(first, min(last, now_month))
    rows = []
    for m in months:
        dest = os.path.join(out_dir, sym, f'{sym}-fundingRate-{m}.zip')
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            rows.append((m, 'exists')); continue
        if m == now_month:
            rows.append((m, 'current-month (vision not yet published)')); continue
        data, err = fetch(VISION.format(sym=sym, month=m))
        if data:
            open(dest, 'wb').write(data)
            rows.append((m, 'ok'))
        else:
            rows.append((m, f'missing({err})'))
    if api_topup:
        try:
            start = int(dt.datetime.strptime(now_month + '-01', '%Y-%m-%d')
                        .replace(tzinfo=dt.UTC).timestamp() * 1000)
            data, err = fetch(FAPI.format(sym=sym, start=start))
            if data:
                recs = json.loads(data)
                if recs:
                    df = pd.DataFrame(dict(
                        calc_time=[r['fundingTime'] for r in recs],
                        funding_interval_hours=8,
                        last_funding_rate=[r['fundingRate'] for r in recs]))
                    dest = os.path.join(out_dir, sym,
                                        f'{sym}-fundingRate-{now_month}.zip')
                    with zipfile.ZipFile(dest, 'w') as z:
                        z.writestr(f'{sym}-fundingRate-{now_month}.csv',
                                   df.to_csv(index=False))
                    rows.append((now_month, f'api-topup({len(recs)} rows)'))
        except Exception as e:
            rows.append((now_month, f'api-topup-failed({e})'))
    return sym, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--klines-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--api-topup', action='store_true')
    a = ap.parse_args()

    ranges = discover_symbol_ranges(a.klines_dir)
    print(f'{len(ranges)} symbols discovered from klines tree')
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(download_symbol, s, f, l, a.out_dir, a.api_topup)
                for s, (f, l) in sorted(ranges.items())]
        results = [f.result() for f in futs]

    report, holes = [], []
    for sym, rows in results:
        got = sum(1 for _, st in rows if st in ('ok', 'exists')
                  or st.startswith('api-topup('))
        missing = [(m, st) for m, st in rows if st.startswith('missing')]
        # leading pre-listing 404s are benign; interior ones are holes
        interior = []
        seen_ok = False
        for m, st in rows:
            ok = st in ('ok', 'exists') or st.startswith('api-topup(')
            if ok:
                seen_ok = True
            elif seen_ok and st.startswith('missing') and not m.startswith(
                    dt.datetime.now(dt.UTC).strftime('%Y-%m')):
                interior.append(m)
        report.append(dict(symbol=sym, months_requested=len(rows),
                           months_obtained=got,
                           leading_gaps=len(missing) - len(interior),
                           interior_holes=';'.join(interior)))
        if interior:
            holes.append((sym, interior))
    rep = pd.DataFrame(report)
    rep.to_csv('funding_download_report.csv', index=False)
    print(f'done: {rep.months_obtained.sum()} months obtained across '
          f'{len(rep)} symbols (report: funding_download_report.csv)')
    if holes:
        print(f'\nWARNING — {len(holes)} symbols have INTERIOR funding holes '
              f'(these would silently zero funding in backtests):')
        for s, ms in holes[:20]:
            print(f'  {s}: {ms}')
    else:
        print('coverage audit clean: no interior funding holes.')


if __name__ == '__main__':
    main()
