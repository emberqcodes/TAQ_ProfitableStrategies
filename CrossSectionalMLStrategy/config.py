"""
THRESHOLD 8h cross-sectional book — replication configuration.

Only the settings the 8h pipeline (scripts 01-08) reads are here. Values are the ones used for the
results in Quant Report v0.2 / Operating Procedure v1.1. The two switches that change between runs are
marked  >>> RUN SWITCH <<<.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_OHLCV_DIR = Path(r"D:/quant/_maindata_updated/binance/ohlcv")      # <SYMBOL>.parquet, 1-minute bars
RAW_FUNDING_DIR = Path(r"D:/quant/_maindata_updated/binance/funding")  # <SYMBOL>.parquet, funding history

PROJECT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_DIR / "data_processed"
RESULTS_DIR = PROJECT_DIR / "results"
PROCESSED_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
XS_DIR = PROCESSED_DIR / "xs"
XS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Bars and features (features_lib.py, 02_build_features.py)
# ---------------------------------------------------------------------------
BTC_SYMBOL = "BTCUSDT"          # cross-asset features
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]   # only used by features_lib's stand-alone main(); the 8h pipeline builds from 01's list
BAR_MINUTES = 15
TRIM_HOURS_AFTER_LISTING = 48   # drop this many hours after each symbol's first bar
REQUIRE_FULL_BARS = True        # drop 15m bars that do not contain all 15 minutes
ATR_PERIOD = 14                 # in 15m bars (Wilder)
HORIZONS_MIN = [60, 120, 240]   # labels written by 02; the 8h label (480) is added by 03
RET_LAGS = [1, 2, 4, 8, 16, 96] # in 15m bars
VOL_SHORT = 4
VOL_LONG = 96                   # = 24h
VWAP_WINDOW = 96
RSI_PERIOD = 14
FUNDING_MEAN_WINDOW = 6         # settlements
LOOKAHEAD_TEST_SAMPLES = 25     # rows per tested symbol recomputed from truncated data (02 runs it on 3 symbols)

# ---------------------------------------------------------------------------
# Costs (bps per side)
# ---------------------------------------------------------------------------
FEE_TAKER_BPS = 3.0
FEE_MAKER_BPS = -1.0            # rebate
SLIPPAGE_BPS = 1.0              # assumed on taker fills

# ---------------------------------------------------------------------------
# Universe (01_universe.py)
# ---------------------------------------------------------------------------
XS_BOOK_SIZE = 20             # tradeable names per month (platform limit)
XS_TRAIN_POOL = 40            # names per month used to train the pooled model
XS_LIQ_WINDOW_DAYS = 30       # trailing window for the liquidity ranking (days strictly before the 1st)
XS_MIN_HISTORY_DAYS = 180
XS_MIN_DAILY_QUOTE_VOL = 5e6  # USDT median daily quote volume
XS_FIRST_MONTH = "2020-07"    # first month a universe is formed
XS_EXCLUDE = []               # e.g. ["USDCUSDT", "FDUSDUSDT"]

# ---------------------------------------------------------------------------
# Horizon, walk-forward (03, 04)
# ---------------------------------------------------------------------------
XS_HORIZON = 480              # minutes. 480 = the 8h book (240 would reproduce the 4h book)
XS_GRID_HOURS = 8             # rebalance grid (UTC); must equal XS_HORIZON / 60
XS_TRAIN_MONTHS = 12          # rolling training window (6 was tested: worse)
TRAIN_MONTHS = 12             # fallback if XS_TRAIN_MONTHS is absent
TEST_MONTHS = 1
XS_FIRST_TEST_MONTH = "2021-07"
XS_DEMEAN_TARGET = True       # train on y minus the cross-sectional mean of y at the same time
LABEL_CLIP_ATR = 10.0
XS_MIN_TRAIN_ROWS = 20000
LGB_PARAMS = dict(
    objective="huber", alpha=2.0,
    learning_rate=0.03, num_leaves=31, min_child_samples=500,
    feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1,
    lambda_l2=10.0, n_estimators=400, verbose=-1, n_jobs=-1, seed=7,
)
HOLDOUT_START = "2026-03-01"  # rows at/after this are never predicted while RUN_HOLDOUT is False

# >>> RUN SWITCH <<<  in-sample development: False.  Final run (reported): True.
RUN_HOLDOUT = True

# >>> RUN SWITCH <<<  run A (reported, identity on): True.  run B (ablation, outputs get _nosym): False.
XS_USE_SYMCAT = True

# ---------------------------------------------------------------------------
# Book backtest (05)
# ---------------------------------------------------------------------------
XS_MIN_NAMES = 10                 # need at least this many book names with predictions to trade a period
XS_N_LEGS = [3, 5, 7]             # names per side to test (5 is the reported setting)
XS_K_LEGS = [0.0, 0.25, 0.5]      # min |demeaned yhat| to take a name; only 0.0 is used (k>0 unbalances the legs)
XS_GROSS_LEVERAGE = 2.0
XS_COST_SCENARIOS = {"taker": 4.0, "maker_mix": 1.5, "maker": -1.0}   # bps per side
XS_REPORT_COST = "taker"
XS_RANDOM_SEEDS = 200
FRAGILITY_N = [1, 3, 5, 10, 20, 50]

# ---------------------------------------------------------------------------
# Maker-fill simulation (06)
# ---------------------------------------------------------------------------
XS_MAKER_N = 5
XS_MAKER_WAITS = [5, 15, 30]      # minutes a rebalance limit order is left working
XS_MAKER_THROUGH = [1.0, 2.0]     # bps of trade-through required to count a fill
XS_MAKER_FALLBACK_TAKER = True    # unfilled -> taker at window end

# ---------------------------------------------------------------------------
# Rolling parameter selection (08)
# ---------------------------------------------------------------------------
XS_PWF_LOOKBACK_MONTHS = 6
XS_PWF_N = [3, 5, 7]
XS_PWF_COST = "maker_mix"
XS_PWF_MIN_IMPROVE = 0.0
