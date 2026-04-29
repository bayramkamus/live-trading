"""Binance günlük OHLCV çekici + 42 teknik feature hesaplayıcı.

Akış:
  1) fetch_klines(sym, interval, start_ms, end_ms)  # Binance birinci, CC fallback
  2) compute_technical_features(ohlcv_1d, ohlcv_4h, btc_close_series)
  3) update_coin(coin)  → data_live/tech/{COIN}.parquet (historical ile concat)

Birinci kaynak: Binance public API (anahtar gerekmiyor, 1200 istek/dk).
Fallback: CryptoCompare (CRYPTOCOMPARE_KEY env var, 50k istek/ay free tier).
Fallback gerekçesi: Binance ABD-IP'den geo-blocked (HTTP 451), GitHub Actions
runner'ları US-based — Binance fail ederse CC'ye düşüyoruz ki workflow patlamasın.
"""
from __future__ import annotations

import argparse
import os
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import requests

from paths import TECH_DIR, COINS, HISTORICAL_DATA_ROOT, DATA_LIVE

BINANCE_BASE = "https://api.binance.com/api/v3/klines"
CC_BASE      = "https://min-api.cryptocompare.com/data/v2"
OHLCV_DIR    = DATA_LIVE / "ohlcv"  # execute.py close fiyatını buradan okur
BINANCE_SYMBOLS = {
    "BTC":  "BTCUSDT", "ETH":  "ETHUSDT", "BNB":  "BNBUSDT",
    "SOL":  "SOLUSDT", "XRP":  "XRPUSDT", "ADA":  "ADAUSDT",
    "DOT":  "DOTUSDT", "AVAX": "AVAXUSDT", "LINK": "LINKUSDT",
    "LTC":  "LTCUSDT",
}


# ============================================================
# 1. BINANCE KLINES
# ============================================================

def fetch_binance_klines(symbol: str, interval: str,
                        start_ms: int, end_ms: Optional[int] = None,
                        limit: int = 1000) -> pd.DataFrame:
    """Binance /api/v3/klines — sayfalı 1000-bar'lık çekim.

    interval: "1d", "4h", "1h"
    Dönüş: date/open_time, open, high, low, close, volume (float)
    """
    rows: List[list] = []
    cursor = start_ms
    if end_ms is None:
        end_ms = int(time.time() * 1000)

    while cursor < end_ms:
        params = {
            "symbol": symbol, "interval": interval,
            "startTime": cursor, "endTime": end_ms,
            "limit": limit,
        }
        try:
            r = requests.get(BINANCE_BASE, params=params, timeout=15)
            if r.status_code != 200:
                warnings.warn(f"Binance {symbol} {interval} status={r.status_code}: {r.text[:200]}")
                break
            batch = r.json()
        except Exception as e:
            warnings.warn(f"Binance fetch hata {symbol} {interval}: {e}")
            break
        if not batch:
            break
        rows.extend(batch)
        last_open = batch[-1][0]
        # bir sonraki sayfanın başlangıcı = son bar open + 1 ms
        cursor = last_open + 1
        if len(batch) < limit:
            break
        time.sleep(0.15)  # rate-limit saygısı

    if not rows:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "n_trades", "tbav", "tqav", "ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(None)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["open_time", "open", "high", "low", "close", "volume"]]


# ============================================================
# 1b. CRYPTOCOMPARE FALLBACK
# ============================================================

def fetch_cryptocompare_ohlcv(symbol: str, interval: str,
                               start_ms: int, end_ms: Optional[int] = None,
                               api_key: Optional[str] = None) -> pd.DataFrame:
    """CryptoCompare histoday/histohour fallback for when Binance is blocked.

    Output schema matches fetch_binance_klines exactly:
        open_time (UTC, tz-naive), open, high, low, close, volume

    Volume = "volumefrom" (base asset volume), same convention as Binance.
    Pair: prefer USDT, fall back to USD if USDT not available.
    """
    if api_key is None:
        api_key = os.environ.get("CRYPTOCOMPARE_KEY", "")

    if interval == "1d":
        endpoint, aggregate = "histoday", 1
    elif interval == "4h":
        endpoint, aggregate = "histohour", 4
    elif interval == "1h":
        endpoint, aggregate = "histohour", 1
    else:
        raise ValueError(f"Bilinmeyen interval: {interval}")

    # Binance sembolünü ayır: BTCUSDT → BTC + USDT
    if symbol.endswith("USDT"):
        fsym, tsym = symbol[:-4], "USDT"
    elif symbol.endswith("USD"):
        fsym, tsym = symbol[:-3], "USD"
    else:
        fsym, tsym = symbol, "USDT"

    if end_ms is None:
        end_ms = int(time.time() * 1000)
    end_ts = end_ms // 1000
    start_ts = start_ms // 1000

    url = f"{CC_BASE}/{endpoint}"
    headers = {}
    if api_key:
        headers["authorization"] = f"Apikey {api_key}"

    rows: List[dict] = []
    cursor_to = end_ts
    tsym_tried_usd = False
    safety = 20  # max paging rounds

    while cursor_to > start_ts and safety > 0:
        safety -= 1
        params = {
            "fsym": fsym, "tsym": tsym,
            "limit": 2000,
            "toTs": cursor_to,
            "aggregate": aggregate,
        }
        try:
            r = requests.get(url, params=params, headers=headers, timeout=20)
            if r.status_code != 200:
                warnings.warn(
                    f"CryptoCompare {fsym}/{tsym} {interval} status={r.status_code}: {r.text[:200]}"
                )
                break
            payload = r.json()
        except Exception as e:
            warnings.warn(f"CryptoCompare fetch hata {fsym}/{tsym} {interval}: {e}")
            break

        if payload.get("Response") != "Success":
            msg = payload.get("Message", "")[:200]
            # USDT yoksa USD'ye düş (BNB gibi bazı coinler)
            if tsym == "USDT" and not tsym_tried_usd:
                tsym_tried_usd = True
                tsym = "USD"
                continue
            warnings.warn(f"CryptoCompare {fsym}/{tsym} response failed: {msg}")
            break

        data = payload.get("Data", {}).get("Data", [])
        if not data:
            break

        # In-range filter
        for b in data:
            t = b.get("time", 0)
            if start_ts <= t <= end_ts:
                rows.append(b)

        oldest_in_batch = data[0].get("time", 0)
        if oldest_in_batch <= start_ts:
            break
        cursor_to = oldest_in_batch - 1
        time.sleep(0.15)

    if not rows:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows).drop_duplicates(subset=["time"]).sort_values("time")
    df["open_time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(None)
    df["volume"] = df.get("volumefrom", 0.0)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["open_time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


_INTERVAL_DELTA = {
    "1d": pd.Timedelta(days=1),
    "4h": pd.Timedelta(hours=4),
    "1h": pd.Timedelta(hours=1),
}


def _drop_incomplete_candles(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """In-progress (henuz kapanmamis) mum cubugunu at - lookahead onleme."""
    if df.empty or "open_time" not in df.columns:
        return df
    delta = _INTERVAL_DELTA.get(interval)
    if delta is None:
        return df
    now = pd.Timestamp.utcnow().tz_localize(None)
    out = df[pd.to_datetime(df["open_time"]) + delta <= now].copy()
    dropped = len(df) - len(out)
    if dropped > 0:
        print(f"[incomplete-candle] {interval}: {dropped} satir atildi")
    return out.reset_index(drop=True)


def fetch_klines(symbol: str, interval: str,
                 start_ms: int, end_ms: Optional[int] = None,
                 limit: int = 1000) -> pd.DataFrame:
    """Binance birinci, fallback CryptoCompare; in-progress mum cikarilir."""
    df = fetch_binance_klines(symbol, interval, start_ms, end_ms, limit)
    if df.empty:
        print(f"[fallback] {symbol} {interval} -> CryptoCompare")
        df = fetch_cryptocompare_ohlcv(symbol, interval, start_ms, end_ms)
    return _drop_incomplete_candles(df, interval)


# ============================================================
# 2. TEKNİK GÖSTERGELER — stdlib + numpy
# ============================================================

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig  = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([high - low,
                    (high - prev_close).abs(),
                    (low  - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mu = series.rolling(window, min_periods=window//2).mean()
    sd = series.rolling(window, min_periods=window//2).std()
    return (series - mu) / sd.replace(0, np.nan)


def _drawdown(close: pd.Series, window: int = 30) -> pd.Series:
    roll_max = close.rolling(window, min_periods=1).max()
    return (close - roll_max) / roll_max


# ============================================================
# 3. 42 TEKNİK KOLON
# ============================================================

def compute_technical_features(daily: pd.DataFrame,
                               four_hour: Optional[pd.DataFrame] = None,
                               btc_daily: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Günlük OHLCV + 4h OHLCV + BTC günlük kapanış → 42 kolon.

    daily columns: open_time (date), open, high, low, close, volume
    four_hour: aynı kolonlar, 4h granülü; None ise 4h feature'ları NaN kalır.
    btc_daily: BTC'nin günlük OHLCV'si; coin==BTC ise kendisi verilebilir.
    """
    if daily.empty:
        return pd.DataFrame()

    d = daily.sort_values("open_time").reset_index(drop=True).copy()
    d = d.rename(columns={"open_time": "date"})
    d["date"] = d["date"].dt.normalize()
    close = d["close"]
    high  = d["high"]
    low   = d["low"]
    vol   = d["volume"]

    # --- returns ---
    d["return_1d"]  = close.pct_change(1)
    d["return_3d"]  = close.pct_change(3)
    d["return_7d"]  = close.pct_change(7)
    d["return_14d"] = close.pct_change(14)

    # --- volatility ---
    ret = np.log(close / close.shift(1))
    d["rolling_volatility_7"]  = ret.rolling(7,  min_periods=3).std() * np.sqrt(365)
    d["rolling_volatility_14"] = ret.rolling(14, min_periods=5).std() * np.sqrt(365)
    d["rolling_volatility_30"] = ret.rolling(30, min_periods=10).std() * np.sqrt(365)

    # --- momentum ---
    d["rsi_14"] = _rsi(close, 14)
    macd, sig, hist = _macd(close)
    d["macd"]        = macd
    d["macd_signal"] = sig
    d["macd_hist"]   = hist

    # --- trend (SMA/EMA) ---
    d["sma_7"]  = close.rolling(7,  min_periods=1).mean()
    d["sma_20"] = close.rolling(20, min_periods=1).mean()
    d["sma_50"] = close.rolling(50, min_periods=1).mean()
    d["ema_12"] = close.ewm(span=12, adjust=False).mean()
    d["ema_26"] = close.ewm(span=26, adjust=False).mean()

    # --- bollinger (20, 2σ) ---
    mid = d["sma_20"]
    sd  = close.rolling(20, min_periods=5).std()
    d["bollinger_upper"] = mid + 2 * sd
    d["bollinger_lower"] = mid - 2 * sd
    d["bollinger_width"] = (d["bollinger_upper"] - d["bollinger_lower"]) / mid

    # --- ATR ---
    d["atr_14"] = _atr(high, low, close, 14)

    # --- volume ---
    d["volume_change"]     = vol.pct_change()
    d["volume_zscore_30"]  = _zscore(vol, 30)

    # --- drawdown / intraday ---
    d["drawdown_30"]        = _drawdown(close, 30)
    d["intraday_return"]    = (close - d["open"]) / d["open"]
    d["intraday_volatility"] = (high - low) / d["open"]
    d["high_low_range"]     = (high - low) / close

    # --- 4h feature'lar ---
    for c in ("volume_sum_4h", "volume_zscore_4h",
              "last_4h_return", "positive_4h_candle_ratio"):
        d[c] = np.nan

    if four_hour is not None and not four_hour.empty:
        fh = four_hour.sort_values("open_time").copy()
        fh["date"] = fh["open_time"].dt.normalize()
        fh["candle_return"] = fh["close"].pct_change()
        fh["is_pos"] = (fh["candle_return"] > 0).astype(int)

        agg = fh.groupby("date").agg(
            volume_sum_4h=("volume", "sum"),
            last_4h_return=("candle_return", "last"),
            positive_4h_candle_ratio=("is_pos", "mean"),
        ).reset_index()
        agg["volume_zscore_4h"] = _zscore(agg["volume_sum_4h"], 30)

        # Merge into daily by date
        d = d.drop(columns=["volume_sum_4h", "volume_zscore_4h",
                            "last_4h_return", "positive_4h_candle_ratio"])
        d = d.merge(agg, on="date", how="left")

    # --- BTC-göreli kolonlar ---
    btc_cols = ["btc_return_1d", "btc_return_3d", "btc_return_7d",
                "btc_volatility_7d", "btc_volatility_30d",
                "btc_rsi_14", "btc_macd",
                "btc_trend_above_sma_20", "btc_trend_above_sma_50",
                "btc_volume_zscore", "btc_drawdown_30d"]
    for c in btc_cols:
        d[c] = np.nan

    if btc_daily is not None and not btc_daily.empty:
        bt = btc_daily.sort_values("open_time").copy()
        bt["date"] = bt["open_time"].dt.normalize()
        bc = bt["close"]
        br = np.log(bc / bc.shift(1))
        btc = pd.DataFrame({
            "date": bt["date"],
            "btc_return_1d": bc.pct_change(1),
            "btc_return_3d": bc.pct_change(3),
            "btc_return_7d": bc.pct_change(7),
            "btc_volatility_7d":  br.rolling(7,  min_periods=3).std() * np.sqrt(365),
            "btc_volatility_30d": br.rolling(30, min_periods=10).std() * np.sqrt(365),
            "btc_rsi_14": _rsi(bc, 14),
            "btc_macd":   _macd(bc)[0],
            "btc_trend_above_sma_20": (bc > bc.rolling(20, min_periods=1).mean()).astype(int),
            "btc_trend_above_sma_50": (bc > bc.rolling(50, min_periods=1).mean()).astype(int),
            "btc_volume_zscore": _zscore(bt["volume"], 30),
            "btc_drawdown_30d": _drawdown(bc, 30),
        })
        d = d.drop(columns=btc_cols)
        d = d.merge(btc, on="date", how="left")

    # Çıkış: date + sabit kolon sırası (model feature_columns ile uyumlu olmak için önemli değil,
    # inference.py zaten `tech_` prefix'i ekleyip dict.get ile alıyor)
    return d


# ============================================================
# 4. UPDATE
# ============================================================

def _last_known_date(p: Path) -> Optional[pd.Timestamp]:
    if not p.exists():
        return None
    df = pd.read_parquet(p, columns=["date"])
    if df.empty:
        return None
    return pd.to_datetime(df["date"]).max()


def update_coin(coin: str, lookback_days: int = 30,
                btc_daily_cached: Optional[pd.DataFrame] = None) -> Path:
    """Son lookback_days'ı Binance'tan çek → 42 feature → historical ile concat."""
    out = TECH_DIR / f"{coin}.parquet"
    sym = BINANCE_SYMBOLS[coin]

    # Mevcut veri
    if out.exists():
        existing = pd.read_parquet(out)
    else:
        hist = HISTORICAL_DATA_ROOT / coin / "features" / f"{coin}_technical_features_1d.parquet"
        existing = pd.read_parquet(hist)
        if "date" not in existing.columns:
            existing = existing.reset_index().rename(columns={existing.columns[0]: "date"})
    existing["date"] = pd.to_datetime(existing["date"])
    if getattr(existing["date"].dt, "tz", None) is not None:
        existing["date"] = existing["date"].dt.tz_convert("UTC").dt.tz_localize(None)

    last = existing["date"].max()
    # Warm-up için 60 gün geriye git (SMA 50 + lookback buffer)
    start = (last - pd.Timedelta(days=60)).to_pydatetime().replace(tzinfo=timezone.utc)
    start_ms = int(start.timestamp() * 1000)

    # Yeni günleri çek (1d + 4h) — Binance birinci, CC fallback
    new_d = fetch_klines(sym, "1d", start_ms)
    if new_d.empty:
        print(f"[{coin}] hiçbir kaynaktan veri gelmedi, mevcut dosya korundu")
        existing.to_parquet(out, index=False)
        return out

    new_4h = fetch_klines(sym, "4h", start_ms)

    # Ham günlük OHLCV'yi de data_live/ohlcv/{coin}_1d.parquet'a yaz
    # execute.py close fiyatı için buradan okuyor
    try:
        OHLCV_DIR.mkdir(parents=True, exist_ok=True)
        ohlcv_out = OHLCV_DIR / f"{coin}_1d.parquet"
        raw = new_d.copy()
        raw = raw.rename(columns={"open_time": "date"})
        raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
        if ohlcv_out.exists():
            old_raw = pd.read_parquet(ohlcv_out)
            old_raw["date"] = pd.to_datetime(old_raw["date"]).dt.normalize()
            raw = pd.concat([old_raw, raw], ignore_index=True)
            raw = raw.drop_duplicates(subset=["date"], keep="last").sort_values("date")
        raw.reset_index(drop=True).to_parquet(ohlcv_out, index=False)
    except Exception as e:
        warnings.warn(f"[{coin}] ham OHLCV yazılamadı: {e}")

    # BTC paralelde — ya çağrıldığıyla cache ya yeniden çek
    if btc_daily_cached is None:
        btc_daily_cached = fetch_klines("BTCUSDT", "1d", start_ms)

    feat_new = compute_technical_features(new_d, new_4h, btc_daily_cached)

    # Eski veriyi last-60d'den öncesiyle koru, yeni satırları ekle
    keep_old = existing[existing["date"] < (last - pd.Timedelta(days=60))]
    # Concat + drop_duplicates (date bazında, keep=last yeni hesabı tutar)
    merged = pd.concat([keep_old, feat_new], ignore_index=True)
    merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)

    if "coin" in existing.columns and "coin" not in merged.columns:
        merged["coin"] = coin

    merged.to_parquet(out, index=False)
    print(f"[{coin}] yazıldı: {out.name} shape={merged.shape} "
          f"son={pd.to_datetime(merged['date']).max().date()}")
    return out


def update_all(lookback_days: int = 30) -> None:
    """Tüm coinler — BTC feature'ları için BTC'yi bir kere çek ve cache'le."""
    print("[ohlcv] BTC günlük veri çekiliyor (cache)...")
    start = (pd.Timestamp.utcnow() - pd.Timedelta(days=365)).to_pydatetime().replace(tzinfo=timezone.utc)
    btc_cache = fetch_klines("BTCUSDT", "1d", int(start.timestamp() * 1000))

    for c in COINS:
        try:
            update_coin(c, lookback_days, btc_daily_cached=btc_cache)
        except Exception as e:
            print(f"[{c}] HATA: {e}")


def backfill_from_historical() -> None:
    for c in COINS:
        out = TECH_DIR / f"{c}.parquet"
        if out.exists():
            print(f"[{c}] zaten var, atla")
            continue
        hist = HISTORICAL_DATA_ROOT / c / "features" / f"{c}_technical_features_1d.parquet"
        df = pd.read_parquet(hist)
        if "date" not in df.columns:
            df = df.reset_index().rename(columns={df.columns[0]: "date"})
        df.to_parquet(out, index=False)
        print(f"[{c}] backfill → {out.name} ({df.shape})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["update", "backfill", "test"], nargs="?", default="update")
    ap.add_argument("--coin", default="BTC")
    ap.add_argument("--lookback", type=int, default=30)
    args = ap.parse_args()
    if args.cmd == "backfill":
        backfill_from_historical()
    elif args.cmd == "test":
        start = int((pd.Timestamp.utcnow() - pd.Timedelta(days=10)).timestamp() * 1000)
        d  = fetch_klines(BINANCE_SYMBOLS[args.coin], "1d", start)
        fh = fetch_klines(BINANCE_SYMBOLS[args.coin], "4h", start)
        bt = fetch_klines("BTCUSDT", "1d", start)
        feat = compute_technical_features(d, fh, bt)
        print(feat.tail(3).T)
    else:
        update_all(args.lookback)
