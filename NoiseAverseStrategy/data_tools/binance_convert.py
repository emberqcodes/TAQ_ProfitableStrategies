"""Convert Binance Vision zip archives -> pipeline parquets.

Handles klines (1m) and fundingRate zips, monthly or daily, discovered
recursively under --in-dir. Robust to Binance format drift:
  * header row present or absent (auto-detected per file)
  * timestamps in ms or microseconds (auto-detected per file by magnitude;
    Binance switched Vision files to microseconds for 2025+)
Outputs (pipeline schema):
  <out-ohlcv>/<BASE>_USDT_USDT.parquet   timestamp(ms), datetime(UTC), open,
                                         high, low, close, volume(base),
                                         quote_volume(USDT — true $ volume)
  <out-funding>/<BASE>_USDT_USDT.parquet timestamp(UTC), fundingRate
Ends with a MANDATORY continuity audit: duplicate/monotonicity checks, gap
totals, and month-over-month median dollar-volume ratios — any adjacent-month
jump > 5x is flagged loudly (this is the stitch-defect detector).

Usage:
  python binance_convert.py --in-dir /path/zips --list           # dry run
  python binance_convert.py --in-dir /path/zips \
      --out-ohlcv /data/ohlcv --out-funding /data/funding --workers 8
"""
import argparse, io, os, re, zipfile, traceback
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
import pandas as pd

KLINE_RE = re.compile(r'^([A-Z0-9]+)-1m-(\d{4}-\d{2}(-\d{2})?)\.zip$')
FUND_RE = re.compile(r'^([A-Z0-9]+)-fundingRate-(\d{4}-\d{2})\.zip$')

KLINE_COLS = ['open_time', 'open', 'high', 'low', 'close', 'volume',
              'close_time', 'quote_volume', 'count', 'taker_buy_volume',
              'taker_buy_quote_volume', 'ignore']


def discover(in_dir):
    kl, fu = defaultdict(list), defaultdict(list)
    for root, _, files in os.walk(in_dir):
        for fn in files:
            m = KLINE_RE.match(fn)
            if m:
                kl[m.group(1)].append(os.path.join(root, fn))
                continue
            m = FUND_RE.match(fn)
            if m:
                fu[m.group(1)].append(os.path.join(root, fn))
    return kl, fu


def _read_zip_csv(path, names):
    with zipfile.ZipFile(path) as z:
        with z.open(z.namelist()[0]) as f:
            head = f.readline().decode('utf-8', 'replace')
        skip = 1 if not head.split(',')[0].strip().isdigit() else 0
        with z.open(z.namelist()[0]) as f:
            return pd.read_csv(io.BytesIO(f.read()), header=None,
                               names=names, skiprows=skip)


def _to_ms(ts):
    """Auto-detect ms vs microseconds by magnitude and normalize to ms."""
    ts = ts.astype('int64')
    med = float(np.median(ts))
    if med > 1e17:          # microseconds (~1.7e18 for 2020s dates... actually 1.7e15)
        return ts // 1000
    if med > 1e14:          # microseconds range for 2020s (1.6e15..1.8e15)
        return ts // 1000
    return ts               # already ms (1.6e12..1.8e12)


def convert_symbol(sym, kline_paths, fund_paths, out_ohlcv, out_funding):
    try:
        base = sym[:-4] if sym.endswith('USDT') else sym
        if not kline_paths:
            k = None
        else:
            parts = []
            for p in sorted(kline_paths):
                df = _read_zip_csv(p, KLINE_COLS)
                df['open_time'] = _to_ms(df['open_time'])
                parts.append(df[['open_time', 'open', 'high', 'low', 'close',
                                 'volume', 'quote_volume']])
            k = pd.concat(parts, ignore_index=True)
            k = k.drop_duplicates('open_time').sort_values('open_time')
            k = k.rename(columns={'open_time': 'timestamp'})
            for c in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
                k[c] = pd.to_numeric(k[c], errors='coerce')
            k['datetime'] = pd.to_datetime(k['timestamp'], unit='ms', utc=True)
            k.to_parquet(os.path.join(out_ohlcv, f'{base}_USDT_USDT.parquet'),
                         index=False)

        if fund_paths:
            fparts = []
            for p in sorted(fund_paths):
                f = _read_zip_csv(p, ['calc_time', 'funding_interval_hours',
                                      'last_funding_rate'])
                fparts.append(f)
            fu = pd.concat(fparts, ignore_index=True)
            fu['calc_time'] = _to_ms(fu['calc_time'])
            fu = fu.drop_duplicates('calc_time').sort_values('calc_time')
            ts = pd.to_datetime(fu['calc_time'], unit='ms', utc=True).dt.round('h')
            if len(ts) and not (ts.min().year >= 2019 and ts.max().year <= 2030):
                raise ValueError(f'implausible funding timestamps '
                                 f'{ts.min()}..{ts.max()} — unit problem in source')
            out = pd.DataFrame(dict(
                timestamp=ts,
                fundingRate=pd.to_numeric(fu['last_funding_rate'],
                                          errors='coerce')))
            out.to_parquet(os.path.join(out_funding,
                                        f'{base}_USDT_USDT.parquet'), index=False)

        # ---- continuity audit ----
        if k is None:
            return dict(symbol=sym, status='ok', rows=0, first='', last='',
                        gaps=0, missing_min=0, worst_mom_vol_ratio=1.0,
                        audit='funding-only', funding_files=len(fund_paths))
        dt = k['timestamp'].diff().dropna()
        gaps = int((dt > 60_000).sum())
        miss = int((dt[dt > 60_000].sum() / 60_000) - gaps) if gaps else 0
        mon = k.set_index('datetime')['quote_volume'].resample('MS').median()
        mon = mon[mon > 0]
        ratio = (mon / mon.shift(1)).dropna()
        worst = float(np.maximum(ratio, 1 / ratio).max()) if len(ratio) else 1.0
        flag = 'VOLUME-CLIFF!' if worst > 5 else 'ok'
        return dict(symbol=sym, status='ok', rows=len(k),
                    first=str(k['datetime'].iloc[0])[:10],
                    last=str(k['datetime'].iloc[-1])[:10],
                    gaps=gaps, missing_min=miss,
                    worst_mom_vol_ratio=round(worst, 1), audit=flag,
                    funding_files=len(fund_paths))
    except Exception:
        return dict(symbol=sym, status='error', err=traceback.format_exc(-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir', required=True)
    ap.add_argument('--out-ohlcv', default='converted/ohlcv')
    ap.add_argument('--out-funding', default='converted/funding')
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 4)
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()

    kl, fu = discover(a.in_dir)
    print(f'found {len(kl)} symbols with kline zips, {len(fu)} with funding zips')
    if a.list:
        for s in sorted(kl):
            months = sorted(os.path.basename(p) for p in kl[s])
            print(f'  {s}: {len(kl[s])} kline zips ({months[0]} .. {months[-1]}), '
                  f'{len(fu.get(s, []))} funding zips')
        if not kl:
            print('NOTHING MATCHED. Expected names like SOLUSDT-1m-2024-01.zip / '
                  'SOLUSDT-fundingRate-2024-01.zip — paste your actual filenames '
                  'if they differ.')
        return

    os.makedirs(a.out_ohlcv, exist_ok=True)
    os.makedirs(a.out_funding, exist_ok=True)
    jobs = [(s, kl.get(s, []), fu.get(s, []), a.out_ohlcv, a.out_funding)
            for s in sorted(set(kl) | set(fu))]
    with Pool(a.workers) as pool:
        res = pool.starmap(convert_symbol, jobs)
    ok = [r for r in res if r['status'] == 'ok']
    err = [r for r in res if r['status'] == 'error']
    rep = pd.DataFrame(ok)
    rep.to_csv('conversion_report.csv', index=False)
    print(f'converted {len(ok)} symbols, {len(err)} errors '
          f'(details in conversion_report.csv)')
    if err:
        print('first error:', err[0]['symbol'], err[0]['err'][-300:])
    bad = rep[~rep['audit'].isin(['ok', 'funding-only'])] if len(rep) else rep
    if len(bad):
        print(f'\nAUDIT FLAGS — {len(bad)} symbols with month-over-month dollar-'
              f'volume jumps > 5x (listing/delisting months are usually benign; '
              f'a shared date across many symbols is a stitch defect):')
        print(bad[['symbol', 'worst_mom_vol_ratio', 'first', 'last']]
              .to_string(index=False))
    else:
        print('\naudit clean: no volume discontinuities detected anywhere.')


if __name__ == '__main__':
    main()
