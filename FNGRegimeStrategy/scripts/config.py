"""config.py -- central paths. Everything resolves relative to this file, so the
package works wherever you drop it. Layout:
    <pkg>/scripts/config.py   <- this file
    <pkg>/Data/{price,funding,supply,fear_greed}/
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data"
SUPPLY    = DATA / "supply"
PRICE     = DATA / "price"
FUNDING   = DATA / "funding"
FEARGREED = DATA / "fear_greed"
for d in (SUPPLY, PRICE, FUNDING, FEARGREED):
    d.mkdir(parents=True, exist_ok=True)
MANIFEST = ROOT / "strategy_manifest.json"

BACKTEST_START = "2020-10-01"
CONTRACTS = {
    "binance":  {"ETHUSDT": "ETHUSDT", "ETHUSDC": "ETHUSDC", "BTCUSDT": "BTCUSDT", "XRPUSDT": "XRPUSDT"},
    "coinbase": {"ETHUSD": "ETH-USD"},
}
