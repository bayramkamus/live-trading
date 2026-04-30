"""A4: Model versiyon meta — artifact_hash + production_metadata.

Her coin icin model.lgb + feature_columns.json + production_metadata.json
SHA-256 ozeti hesaplanir. Production_metadata'daki created_at ve feature_set
sinyal kayitlarina yazilir.

Kullanim:
    from model_version import get_model_version
    mv = get_model_version("BTC")
    # {'artifact_hash': 'a1b2c3d4e5f6', 'created_at': '2026-04-25T...',
    #  'feature_set': 'all (sent+macro+tech)...'}
"""
from __future__ import annotations
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from paths import ARTIFACTS_ROOT


def _file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=64)
def get_model_version(coin: str) -> Dict[str, Optional[str]]:
    """Coin icin model artifact'larinin SHA + metadata'si.

    Hash = sha256(model.lgb || feature_columns.json) ilk 12 hex.
    Production metadata global olduğu için coin-bagimsiz alanlar onun icin
    okunur.
    """
    coin_dir = ARTIFACTS_ROOT / coin
    out: Dict[str, Optional[str]] = {
        "artifact_hash": None, "created_at": None, "feature_set": None,
    }
    if not coin_dir.exists():
        return out

    h = hashlib.sha256()
    for fname in ("model.lgb", "feature_columns.json"):
        p = coin_dir / fname
        if p.exists():
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
    out["artifact_hash"] = h.hexdigest()[:12]

    # Production metadata — uretim klasorunun ust dizininde
    meta_path = ARTIFACTS_ROOT / "production_metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            out["created_at"] = meta.get("created_at")
            out["feature_set"] = meta.get("feature_set")
        except Exception:
            pass

    return out


def get_global_version() -> Dict[str, Optional[str]]:
    """Tum coinlerin artifact_hash'lerinden tek bir global hash uret.
    Dashboard header'da 'aktif model surumu' gostermek icin."""
    h = hashlib.sha256()
    for coin in sorted([p.name for p in ARTIFACTS_ROOT.iterdir() if p.is_dir()]):
        mv = get_model_version(coin)
        if mv["artifact_hash"]:
            h.update(coin.encode())
            h.update(mv["artifact_hash"].encode())

    out = {"global_hash": h.hexdigest()[:12]}
    # Production metadata bilgilerini de geri gönder (BTC üzerinden okuruz)
    btc = get_model_version("BTC")
    out["created_at"]  = btc["created_at"]
    out["feature_set"] = btc["feature_set"]
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        for c in sys.argv[1:]:
            print(c, get_model_version(c))
    else:
        print(json.dumps(get_global_version(), indent=2))
        print()
        for coin in ["BTC","ETH","BNB","SOL","XRP","ADA","DOT","AVAX","LINK","LTC"]:
            mv = get_model_version(coin)
            print(f"{coin:5s} {mv['artifact_hash']}")
