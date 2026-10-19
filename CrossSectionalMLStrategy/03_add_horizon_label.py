"""
Step 20 - Add a label for another horizon to the existing per-symbol feature files (no rebuild).

For H = XS_HORIZON (minutes), adds to data_processed/xs/feat/<SYM>.parquet:
  exit_ts_H  = known_at_ts + H
  exit_px_H  = 1m open at exit_ts_H          (from data_processed/xs/slim/<SYM>.parquet)
  y_H        = (exit_px_H - entry_px) / atr
Files that already have y_H are skipped. Features are untouched.
"""
import sys
import pandas as pd
import config as C

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
H = C.XS_HORIZON
FEAT_DIR, SLIM_DIR = C.XS_DIR / "feat", C.XS_DIR / "slim"

if __name__ == "__main__":
    files = sorted(FEAT_DIR.glob("*.parquet"))
    done = skipped = 0
    for i, fp in enumerate(files):
        d = pd.read_parquet(fp)
        if f"y_{H}" in d.columns:
            skipped += 1; continue
        s = pd.read_parquet(SLIM_DIR / f"{fp.stem}.parquet", columns=["timestamp", "open"])
        open_map = pd.Series(s["open"].to_numpy(), index=s["timestamp"])
        exit_ts = d["known_at_ts"] + pd.Timedelta(minutes=H)
        d[f"exit_ts_{H}"] = exit_ts
        d[f"exit_px_{H}"] = open_map.reindex(exit_ts).to_numpy()
        d[f"y_{H}"] = (d[f"exit_px_{H}"] - d["entry_px"]) / d["atr"]
        assert (d[f"exit_ts_{H}"] > d["known_at_ts"]).all()
        d.to_parquet(fp, index=False); done += 1
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(files)}")
    print(f"horizon {H}: labels added to {done} files, {skipped} already had them")
