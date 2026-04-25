"""Tüm path sabitlerini tek bir yerden topla.

live_trading/
├── artifacts/v4_production/{COIN}/model.lgb + 4 JSON
├── code/
├── data_live/
│   ├── sentiment/{COIN}.parquet    # 29 kolon + date
│   ├── tech/{COIN}.parquet         # 41 teknik kolon + date
│   └── macro/macro.parquet         # 27 makro kolon + date
└── logs/
"""
from __future__ import annotations
from pathlib import Path

# Bu dosya code/ altında; bir yukarı çıkınca live_trading/ kökü
ROOT = Path(__file__).resolve().parent.parent

ARTIFACTS_ROOT = ROOT / "artifacts" / "v4_production"
DATA_LIVE      = ROOT / "data_live"
LOGS_DIR       = ROOT / "logs"

SENT_DIR  = DATA_LIVE / "sentiment"
TECH_DIR  = DATA_LIVE / "tech"
MACRO_DIR = DATA_LIVE / "macro"

# Backfill: eğitimde kullanılan geçmiş veriyi buradan okuruz (tek seferlik kopya)
# Canlıya geçtikten sonra bu referans kalkar.
# live_trading/ klasörü `data/` altındaysa otomatik bulur (Windows + Linux uyumlu).
# Override için LIVE_HISTORICAL_DATA_ROOT env var kullanılabilir.
import os
_env_override = os.environ.get("LIVE_HISTORICAL_DATA_ROOT")
if _env_override:
    HISTORICAL_DATA_ROOT = Path(_env_override).resolve()
else:
    # ROOT = live_trading/, parent = data/  (bizim repo düzeni)
    _candidate = ROOT.parent
    if (_candidate / "models" / "v2_sentiment_strategy").is_dir():
        HISTORICAL_DATA_ROOT = _candidate
    else:
        # Fallback: cowork mount path (sadece sandbox'ta anlamlı)
        HISTORICAL_DATA_ROOT = Path("/sessions/exciting-eloquent-goodall/mnt/data")

COINS = ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOT", "AVAX", "LINK", "LTC"]


def ensure_dirs() -> None:
    for d in (ARTIFACTS_ROOT, DATA_LIVE, LOGS_DIR, SENT_DIR, TECH_DIR, MACRO_DIR):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print("ROOT =", ROOT)
    print("ARTIFACTS_ROOT =", ARTIFACTS_ROOT)
    print("DATA_LIVE =", DATA_LIVE)
    print("Artifacts var?", ARTIFACTS_ROOT.exists())
    print("Coin klasörleri:", [c.name for c in ARTIFACTS_ROOT.glob("*") if c.is_dir()])
