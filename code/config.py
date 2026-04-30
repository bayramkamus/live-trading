"""Live trading configuration — single source of truth.

Tum sabitler env-aware: env'den oku, yoksa default. Modüller bu dosyadan okur.

Env override örnekleri (.github/workflows/daily.yml veya local .env):
    PAPER_BASE_EQUITY=10000
    PAPER_RISK_PCT=0.10
    MAX_POSITIONS=5
    FEE_BPS=10
    ALLOW_SHORTS=true
    MIN_CONFIDENCE=0.0
    DIR_MARGIN_DEFAULT=0.03
    HOLD_VETO_DEFAULT=0.05
    DIR_MARGIN_WEAK=0.06
    HOLD_VETO_WEAK=0.0
    WEAK_COINS=ADA,AVAX,DOT,ETH,LINK,LTC
    NEWS_MIN_LAST7D=3
    NEWS_STALE_DAYS_MAX=2
    FILL_OFFSET=1
"""
from __future__ import annotations
import os
from typing import Set


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on", "y", "t")


def _env_set(key: str, default: Set[str]) -> Set[str]:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return set(default)
    return {s.strip().upper() for s in raw.split(",") if s.strip()}


# ============================================================
# Paper broker
# ============================================================
BASE_EQUITY        = _env_float("PAPER_BASE_EQUITY", 10_000.0)
RISK_PCT_PER_TRADE = _env_float("PAPER_RISK_PCT",    0.10)
MAX_POSITIONS      = _env_int  ("MAX_POSITIONS",     5)
FEE_BPS            = _env_float("FEE_BPS",           10.0)
ALLOW_SHORTS       = _env_bool ("ALLOW_SHORTS",      True)


# ============================================================
# Sinyal kapilari (A2)
# ============================================================
# A2 gate'leri zaten 3-kapi + tier + haber yapiyor — ek confidence filtresi
# default'unda kapali. >0 verirsen p_buy/p_sell >= MIN_CONFIDENCE sarti eklenir.
MIN_CONFIDENCE = _env_float("MIN_CONFIDENCE", 0.0)

# 3-kapi varsayilanlari (strong tier — production f1_val >= 0.25 ve best_iter > 1)
DIR_MARGIN_DEFAULT = _env_float("DIR_MARGIN_DEFAULT", 0.03)
HOLD_VETO_DEFAULT  = _env_float("HOLD_VETO_DEFAULT",  0.05)

# Weak tier (production f1_val < 0.25 veya best_iter <= 1)
DIR_MARGIN_WEAK = _env_float("DIR_MARGIN_WEAK", 0.06)
HOLD_VETO_WEAK  = _env_float("HOLD_VETO_WEAK",  0.0)
WEAK_COINS      = _env_set  ("WEAK_COINS",
                              {"ADA", "AVAX", "DOT", "ETH", "LINK", "LTC"})


# ============================================================
# Haber kapsam gating (A2)
# ============================================================
NEWS_MIN_LAST7D     = _env_int("NEWS_MIN_LAST7D",     3)
NEWS_STALE_DAYS_MAX = _env_int("NEWS_STALE_DAYS_MAX", 2)


# ============================================================
# Sinyal/fill timing (A1)
# ============================================================
# T+1 fill: cron 02:10 UTC'de signal_date=T-2, fill_date=T-1 (her ikisi de
# fully-closed gun). 0 verirsen legacy same-day fill (lookahead riski).
FILL_OFFSET = _env_int("FILL_OFFSET", 1)


def dump() -> dict:
    """Tum aktif config'i dict olarak don — orchestrate startup'inda log icin."""
    return {
        "BASE_EQUITY":         BASE_EQUITY,
        "RISK_PCT_PER_TRADE":  RISK_PCT_PER_TRADE,
        "MAX_POSITIONS":       MAX_POSITIONS,
        "FEE_BPS":             FEE_BPS,
        "ALLOW_SHORTS":        ALLOW_SHORTS,
        "MIN_CONFIDENCE":      MIN_CONFIDENCE,
        "DIR_MARGIN_DEFAULT":  DIR_MARGIN_DEFAULT,
        "HOLD_VETO_DEFAULT":   HOLD_VETO_DEFAULT,
        "DIR_MARGIN_WEAK":     DIR_MARGIN_WEAK,
        "HOLD_VETO_WEAK":      HOLD_VETO_WEAK,
        "WEAK_COINS":          sorted(WEAK_COINS),
        "NEWS_MIN_LAST7D":     NEWS_MIN_LAST7D,
        "NEWS_STALE_DAYS_MAX": NEWS_STALE_DAYS_MAX,
        "FILL_OFFSET":         FILL_OFFSET,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(dump(), indent=2, default=str))
