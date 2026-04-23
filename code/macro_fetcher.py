"""Makro feature güncelleyici — FRED API ile gerçek veri.

Kaynaklar (hepsi ücretsiz, FRED API key yeterli):
  VIX   → FRED VIXCLS   (günlük, CBOE Volatility Index)
  DXY   → FRED DTWEXBGS (günlük, Trade-Weighted U.S. Dollar Index: Broad Goods & Services)
  IGREA → FRED WEI      (haftalık, Weekly Economic Index - Kilian IGREA proxy'si)

FRED API key: https://fredaccount.stlouisfed.org/apikey  (ücretsiz, anında verilir)
Env var:  FRED_API_KEY=...
Veya aşağıdaki API_KEYS bloğuna yaz.

Schema (28 kolon — eğitim verisiyle aynı):
    date,
    {vix,dxy,igrea}_raw_value,
    {…}_daily_change, _weekly_change,
    {…}_rolling_zscore_7, _rolling_zscore_30,
    {…}_lag_1, _lag_3, _lag_7, _lag_30
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from paths import MACRO_DIR, HISTORICAL_DATA_ROOT


# ============================================================
# CONFIG
# ============================================================

API_KEYS = {
    # https://fredaccount.stlouisfed.org/apikey (ücretsiz, anında alınır)
    "FRED_API_KEY": "5420d260fd001bb3c15728790a62fb74",
}

# FRED series IDs
FRED_SERIES = {
    "vix":   "VIXCLS",     # CBOE VIX, günlük
    "dxy":   "DTWEXBGS",   # Trade-Weighted USD (Broad, Goods & Services), günlük
    "igrea": "WEI",        # Weekly Economic Index (IGREA proxy), haftalık
}

MACRO_OUT = MACRO_DIR / "macro.parquet"
FRED_URL  = "https://api.stlouisfed.org/fred/series/observations"


def _get_fred_key() -> str:
    key = os.environ.get("FRED_API_KEY") or API_KEYS.get("FRED_API_KEY") or ""
    return key.strip()


# ============================================================
# FRED fetcher
# ============================================================

def fetch_fred_series(series_id: str,
                      start: str,
                      end: Optional[str] = None,
                      timeout: int = 30,
                      retries: int = 4) -> pd.DataFrame:
    """FRED'den bir seriyi çek — DataFrame[date, {series_id}].

    5xx hatalarında exponential backoff ile retry (FRED ara sıra 500 döndürür).
    start / end: 'YYYY-MM-DD'. end None ise bugün (UTC).
    """
    key = _get_fred_key()
    if not key:
        print(f"[fred] FRED_API_KEY yok; {series_id} atlandı")
        return pd.DataFrame(columns=["date", series_id])

    if end is None:
        end = pd.Timestamp.utcnow().date().isoformat()

    params = {
        "series_id": series_id,
        "api_key": key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }

    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(FRED_URL, params=params, timeout=timeout)
            # 5xx'leri retry et
            if 500 <= r.status_code < 600:
                last_err = f"{r.status_code} {r.reason}"
                backoff = 1.5 * (attempt + 1)
                print(f"[fred] {series_id} {last_err} — {backoff:.1f}s bekle, deneme {attempt+2}/{retries}")
                time.sleep(backoff)
                continue
            r.raise_for_status()
            js = r.json()
            obs = js.get("observations", [])
            if not obs:
                return pd.DataFrame(columns=["date", series_id])
            df = pd.DataFrame(obs)
            df["date"] = pd.to_datetime(df["date"])
            df[series_id] = pd.to_numeric(df["value"], errors="coerce")
            df = df[["date", series_id]].dropna(subset=[series_id])
            return df.sort_values("date").reset_index(drop=True)
        except requests.exceptions.RequestException as e:
            last_err = str(e)
            backoff = 1.5 * (attempt + 1)
            print(f"[fred] {series_id} hata: {last_err} — {backoff:.1f}s bekle, deneme {attempt+2}/{retries}")
            time.sleep(backoff)
        except Exception as e:
            last_err = str(e)
            break

    print(f"[fred] {series_id} kesin hata (retry bitti): {last_err}")
    return pd.DataFrame(columns=["date", series_id])


def fetch_all_raw(start: str = "2021-06-01",
                  end: Optional[str] = None) -> pd.DataFrame:
    """Üç kaynağı çek, günlük takvime forward-fill et.

    Returns: DataFrame[date, vix, dxy, igrea]  — günlük, tz-naive.
    """
    if end is None:
        end = pd.Timestamp.utcnow().date().isoformat()

    frames = {}
    for name, sid in FRED_SERIES.items():
        df = fetch_fred_series(sid, start, end)
        if df.empty:
            print(f"[fred] {name} ({sid}): 0 satır")
            continue
        df = df.rename(columns={sid: name})
        frames[name] = df
        print(f"[fred] {name} ({sid}): {len(df)} satır  "
              f"son={df['date'].max().date()}  değer={df[name].iloc[-1]:.3f}")
        time.sleep(0.15)  # polite

    if not frames:
        return pd.DataFrame(columns=["date", "vix", "dxy", "igrea"])

    # Günlük takvim üzerine hizala
    cal = pd.date_range(start=start, end=end, freq="D")
    out = pd.DataFrame({"date": cal})
    for name, df in frames.items():
        out = out.merge(df, on="date", how="left")

    # FRED serileri tatil/hafta sonu için NaN döndürür → forward fill
    for c in ("vix", "dxy", "igrea"):
        if c in out.columns:
            out[c] = out[c].ffill()

    return out


# ============================================================
# Transforms (27 kolon)
# ============================================================

def _add_transforms(out: pd.DataFrame, src: pd.Series, prefix: str) -> None:
    """Bir seriye şu 9 transform'u ekle:
        raw_value, daily_change, weekly_change,
        rolling_zscore_7, rolling_zscore_30,
        lag_1, lag_3, lag_7, lag_30
    """
    s = src.astype(float)

    out[f"{prefix}_raw_value"]     = s.values
    out[f"{prefix}_daily_change"]  = s.diff(1).values
    out[f"{prefix}_weekly_change"] = s.diff(7).values

    for w in (7, 30):
        roll = s.rolling(window=w, min_periods=2)
        mean = roll.mean()
        std  = roll.std(ddof=0).replace(0.0, np.nan)
        z = (s - mean) / std
        out[f"{prefix}_rolling_zscore_{w}"] = z.values

    for lag in (1, 3, 7, 30):
        out[f"{prefix}_lag_{lag}"] = s.shift(lag).values


def compute_macro_transforms(raw: pd.DataFrame) -> pd.DataFrame:
    """Ham (date, vix, dxy, igrea) → 28 kolonluk feature matrisi."""
    if raw.empty:
        return pd.DataFrame()

    raw = raw.sort_values("date").reset_index(drop=True)
    out = pd.DataFrame({"date": raw["date"].values})

    for prefix in ("vix", "dxy", "igrea"):
        if prefix not in raw.columns:
            # eksikse NaN kolonlar doldur
            s = pd.Series([np.nan] * len(raw))
        else:
            s = raw[prefix]
        _add_transforms(out, s, prefix)

    return out


# ============================================================
# update / backfill
# ============================================================

EXPECTED_COLS = [
    "date",
    "vix_raw_value", "vix_daily_change", "vix_weekly_change",
    "vix_rolling_zscore_7", "vix_rolling_zscore_30",
    "vix_lag_1", "vix_lag_3", "vix_lag_7", "vix_lag_30",
    "dxy_raw_value", "dxy_daily_change", "dxy_weekly_change",
    "dxy_rolling_zscore_7", "dxy_rolling_zscore_30",
    "dxy_lag_1", "dxy_lag_3", "dxy_lag_7", "dxy_lag_30",
    "igrea_raw_value", "igrea_daily_change", "igrea_weekly_change",
    "igrea_rolling_zscore_7", "igrea_rolling_zscore_30",
    "igrea_lag_1", "igrea_lag_3", "igrea_lag_7", "igrea_lag_30",
]


def _strip_tz(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s.dt.normalize()


def update_macro(lookback_days: int = 60) -> Path:
    """FRED'den son `lookback_days`'i çek, var olan parquet ile merge et.

    FRED key yoksa veya fetch başarısızsa, historical parquet'ı kopyalar
    (ilk kurulumda kullanışlı).
    """
    MACRO_DIR.mkdir(parents=True, exist_ok=True)

    # Mevcut parquet
    if MACRO_OUT.exists():
        existing = pd.read_parquet(MACRO_OUT)
    else:
        hist = HISTORICAL_DATA_ROOT / "macro_features_1d.parquet"
        if hist.exists():
            existing = pd.read_parquet(hist)
            if "date" not in existing.columns:
                existing = existing.reset_index().rename(
                    columns={existing.index.name or "index": "date"}
                )
        else:
            existing = pd.DataFrame(columns=EXPECTED_COLS)

    if "date" in existing.columns:
        existing["date"] = _strip_tz(existing["date"])

    key = _get_fred_key()
    if not key:
        print("[macro] FRED_API_KEY yok — mevcut parquet'ı koru (update yok)")
        if not MACRO_OUT.exists() and not existing.empty:
            existing.to_parquet(MACRO_OUT, index=False)
            print(f"[macro] historical kopyalandı → {MACRO_OUT}")
        return MACRO_OUT

    today = pd.Timestamp.utcnow().normalize()
    # Transform'lar için 40 gün warm-up gerekli (lag_30)
    warm = max(lookback_days, 40) + 10
    start = (today - pd.Timedelta(days=warm)).date().isoformat()
    end = today.date().isoformat()

    print(f"[macro] FRED çek: {start} → {end}")
    raw = fetch_all_raw(start=start, end=end)
    if raw.empty:
        print("[macro] FRED boş; mevcut parquet korundu")
        return MACRO_OUT

    new_feats = compute_macro_transforms(raw)
    if new_feats.empty:
        print("[macro] transform boş döndü; mevcut parquet korundu")
        return MACRO_OUT

    new_feats["date"] = _strip_tz(new_feats["date"])

    # Kolonları hizala
    for c in EXPECTED_COLS:
        if c not in new_feats.columns:
            new_feats[c] = np.nan
    new_feats = new_feats[EXPECTED_COLS]

    # Merge: yeni feature hesaplamaları eski günlerin üstüne YAZMASIN
    # (warm-up penceresinde eski rolling değerler daha doğru olabilir)
    if not existing.empty:
        # yalnız YENİ tarihleri al
        cutoff = existing["date"].max()
        fresh = new_feats[new_feats["date"] > cutoff]
        merged = pd.concat([existing, fresh], ignore_index=True)
    else:
        merged = new_feats

    merged = merged.drop_duplicates(subset=["date"], keep="last")
    merged = merged.sort_values("date").reset_index(drop=True)

    # Eğer bir seri bu seferki fetch'te hiç gelmediyse (örn. VIXCLS 500),
    # yeni günlerdeki NaN'leri son bilinen raw_value ile doldur. Change/z/lag
    # kolonları NaN kalır — bir sonraki başarılı update'te düzelir.
    for prefix in ("vix", "dxy", "igrea"):
        col = f"{prefix}_raw_value"
        if col in merged.columns:
            merged[col] = merged[col].ffill()

    # Sadece beklenen kolonları yaz
    for c in EXPECTED_COLS:
        if c not in merged.columns:
            merged[c] = np.nan
    merged = merged[EXPECTED_COLS]

    merged.to_parquet(MACRO_OUT, index=False)
    print(f"[macro] yazıldı → {MACRO_OUT.name}  "
          f"shape={merged.shape}  son={pd.to_datetime(merged['date']).max().date()}")
    return MACRO_OUT


def rebuild_full(start: str = "2021-06-01") -> Path:
    """Tüm tarihi FRED'den sıfırdan yeniden oluştur (historical'ın yerine geçer)."""
    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    key = _get_fred_key()
    if not key:
        raise RuntimeError("FRED_API_KEY yok — rebuild yapılamıyor")

    end = pd.Timestamp.utcnow().date().isoformat()
    print(f"[macro] FULL REBUILD: {start} → {end}")
    raw = fetch_all_raw(start=start, end=end)
    feats = compute_macro_transforms(raw)
    feats["date"] = _strip_tz(feats["date"])

    for c in EXPECTED_COLS:
        if c not in feats.columns:
            feats[c] = np.nan
    feats = feats[EXPECTED_COLS].sort_values("date").reset_index(drop=True)
    feats.to_parquet(MACRO_OUT, index=False)
    print(f"[macro] yazıldı → {MACRO_OUT.name}  shape={feats.shape}")
    return MACRO_OUT


def backfill_from_historical() -> None:
    """macro.parquet yoksa, HISTORICAL_DATA_ROOT'tan kopyala (ilk kurulum)."""
    if MACRO_OUT.exists():
        print(f"{MACRO_OUT.name} zaten var, atla")
        return
    hist = HISTORICAL_DATA_ROOT / "macro_features_1d.parquet"
    if not hist.exists():
        print(f"historical yok: {hist}")
        return
    df = pd.read_parquet(hist)
    if "date" not in df.columns:
        df = df.reset_index().rename(columns={df.index.name or "index": "date"})
    df["date"] = _strip_tz(df["date"])
    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(MACRO_OUT, index=False)
    print(f"macro backfill → {MACRO_OUT.name} ({df.shape})")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if args and args[0] == "backfill":
        backfill_from_historical()
    elif args and args[0] == "rebuild":
        rebuild_full()
    elif args and args[0] == "test":
        # FRED bağlantı testi
        df = fetch_fred_series("VIXCLS",
                               start=(pd.Timestamp.utcnow() - pd.Timedelta(days=30)).date().isoformat())
        print(df.tail())
    else:
        p = update_macro()
        if p.exists():
            df = pd.read_parquet(p)
            print(f"\n{p.name}: {df.shape}  "
                  f"son={pd.to_datetime(df['date']).max().date()}")
            print("son 5 VIX:")
            print(df[["date", "vix_raw_value", "vix_daily_change",
                      "vix_rolling_zscore_7"]].tail())
