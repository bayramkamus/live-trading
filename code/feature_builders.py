"""Canlı feature okuyucuları.

Read order:
  1) data_live/{sentiment,tech,macro} parquet'leri varsa oradan
  2) Yoksa HISTORICAL_DATA_ROOT'tan (backfill; yalnız ilk kurulum için)
"""
from __future__ import annotations
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from paths import (
    HISTORICAL_DATA_ROOT,
    SENT_DIR, TECH_DIR, MACRO_DIR,
)

NON_FEATURE_COLS = {
    "date", "close", "coin", "scenario",
    "w_s1", "w_s2", "w_s3",
    "fwd_return", "label",
}


def _normalize_dates(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return pd.DatetimeIndex(idx).normalize()


def _strip_tz(s: pd.Series) -> pd.Series:
    """Tz-aware bir datetime serisini UTC'ye çevirip tz-bilgisini düşürür."""
    s = pd.to_datetime(s)
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s.dt.normalize()


def _read_parquet_first(paths: list[Path]) -> pd.DataFrame | None:
    for p in paths:
        if p.exists():
            return pd.read_parquet(p)
    return None


# ---------- Sentiment ----------

def load_sentiment_features(coin: str) -> pd.DataFrame:
    """Canlı sentiment parquet'i yoksa V2 eğitim parquet'ine fall back et.

    Returns: date + 29 sentiment kolonu (+ ek kolonlar olabilir; inference align eder).
    """
    live = SENT_DIR / f"{coin}.parquet"
    if live.exists():
        df = pd.read_parquet(live)
    else:
        # Backfill: V2 eğitim parquet'i
        hist = HISTORICAL_DATA_ROOT / "models" / "v2_sentiment_strategy" / "features" / f"features_{coin}_S1.parquet"
        df = pd.read_parquet(hist)

    df["date"] = _strip_tz(df["date"])
    drop = [c for c in ["coin", "scenario", "w_s1", "w_s2", "w_s3", "close", "fwd_return", "label"] if c in df.columns]
    df = df.drop(columns=drop).sort_values("date").reset_index(drop=True)
    return df


# ---------- Macro ----------

def load_macro_features() -> pd.DataFrame:
    live = MACRO_DIR / "macro.parquet"
    hist = HISTORICAL_DATA_ROOT / "macro_features_1d.parquet"

    if live.exists():
        m = pd.read_parquet(live)
    else:
        m = pd.read_parquet(hist)

    # index ile tarih kolonu olarak da gelebilir — iki yolu da destekle
    if "date" not in m.columns:
        m.index = _normalize_dates(m.index)
        m = m.reset_index().rename(columns={"index": "date"})
        if "date" not in m.columns:
            m = m.rename(columns={m.columns[0]: "date"})
    m["date"] = _strip_tz(m["date"])
    return m.sort_values("date").reset_index(drop=True)


# ---------- Technical ----------

def load_technical_features(coin: str) -> pd.DataFrame:
    live = TECH_DIR / f"{coin}.parquet"
    hist = HISTORICAL_DATA_ROOT / coin / "features" / f"{coin}_technical_features_1d.parquet"

    if live.exists():
        t = pd.read_parquet(live)
    else:
        t = pd.read_parquet(hist)

    if "date" not in t.columns:
        t.index = _normalize_dates(t.index)
        t = t.reset_index().rename(columns={"index": "date"})
        if "date" not in t.columns:
            t = t.rename(columns={t.columns[0]: "date"})
    t["date"] = _strip_tz(t["date"])
    if "coin" in t.columns:
        t = t.drop(columns=["coin"])
    return t.sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    # Smoke
    for c in ["BTC", "ETH"]:
        s = load_sentiment_features(c)
        t = load_technical_features(c)
        print(f"{c}: sent {s.shape} son={s['date'].max().date()}, tech {t.shape} son={t['date'].max().date()}")
    m = load_macro_features()
    print(f"Macro: {m.shape} son={m['date'].max().date()}")
