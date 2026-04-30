"""Sinyal → broker emri köprüsü.

Kullanım:
    from execute import execute_signals
    trades = execute_signals(signals_df, as_of_date=None, broker=None)

signals_df en az şu kolonlara sahip olmalı:
    coin, signal (+1/0/-1) VEYA signal_int
    p_buy, p_sell  (opsiyonel — confidence filter için)
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from paths import TECH_DIR, DATA_LIVE, COINS
from broker import BrokerAdapter, Trade
from paper_broker import PaperBroker
# A3: tek kaynak config.py
from config import MIN_CONFIDENCE


# ============================================================
# Fiyat kaynağı
# ============================================================

def _read_tech_close(coin: str, target_date: pd.Timestamp,
                     strict: bool = False) -> Optional[float]:
    """data_live/ohlcv/{coin}_1d.parquet -> target_date close.
    strict=True: tam target_date eslemesi sart (T+1 fill icin).
    strict=False: en yakin gecmis ile fallback (legacy)."""
    ohlcv_dir = DATA_LIVE / "ohlcv"
    p = ohlcv_dir / f"{coin}_1d.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if "close" not in df.columns:
        return None
    dc = pd.to_datetime(df["date"])
    if getattr(dc.dt, "tz", None) is not None:
        dc = dc.dt.tz_convert("UTC").dt.tz_localize(None)
    dc = dc.dt.normalize()
    df = df.assign(date=dc)
    row = df.loc[df["date"] == target_date]
    if row.empty:
        if strict:
            return None
        past = df.loc[df["date"] <= target_date]
        if past.empty:
            return None
        row = past.tail(1)
    return float(row["close"].iloc[0])


def _fallback_close_from_tech(coin: str, as_of_date: pd.Timestamp) -> Optional[float]:
    """Tech feature parquet'te close doğrudan yoksa, sma_7 veya ema_12 ile
    yaklaşık bir değer döndüremeyiz. Bu fonksiyon None döndürüp fiyatsız bırakır."""
    return None


def build_price_map(coins: list[str], fill_date,
                    strict: bool = True) -> Dict[str, float]:
    """fill_date close'unu OHLCV parquet'ten oku - fill price.
    strict=True (default): tam fill_date sart, gelecek/yok ise coin atlanir.
    strict=False: en yakin gecmis fallback (debug/replay)."""
    fill_date = pd.Timestamp(fill_date).normalize()
    out: Dict[str, float] = {}
    for c in coins:
        px = _read_tech_close(c, fill_date, strict=strict)
        if px is not None:
            out[c] = float(px)
    return out


# ============================================================
# Execute
# ============================================================

def _signal_int(row: pd.Series) -> int:
    if "signal_int" in row and pd.notna(row["signal_int"]):
        return int(row["signal_int"])
    s = str(row.get("signal", "HOLD")).upper()
    return {"BUY": 1, "SELL": -1, "HOLD": 0}.get(s, 0)


def _passes_confidence(row: pd.Series, min_conf: float) -> bool:
    if min_conf <= 0:
        return True
    sig = _signal_int(row)
    if sig == +1:
        return float(row.get("p_buy", 1.0)) >= min_conf
    if sig == -1:
        return float(row.get("p_sell", 1.0)) >= min_conf
    return True


def execute_signals(signals_df: pd.DataFrame,
                    as_of_date: Optional[str] = None,
                    fill_date: Optional[str] = None,
                    broker: Optional[BrokerAdapter] = None,
                    min_confidence: float = MIN_CONFIDENCE,
                    prices: Optional[Dict[str, float]] = None,
                    save: bool = True) -> list[Trade]:
    """Sinyal tablosu -> broker.step().
    Lookahead-safe: as_of_date = sinyal gunu (T), fill_date = doldurma gunu (T+1)."""
    if as_of_date is None:
        as_of_date = pd.Timestamp.utcnow().date().isoformat()
    if fill_date is None:
        fill_date = as_of_date  # legacy same-day

    if broker is None:
        broker = PaperBroker.load_or_init()

    sig_map: Dict[str, int] = {}
    for _, row in signals_df.iterrows():
        coin = str(row["coin"])
        sig_map[coin] = _signal_int(row) if _passes_confidence(row, min_confidence) else 0

    # Fill price = fill_date close (strict)
    if prices is None:
        prices = build_price_map(list(sig_map.keys()), fill_date, strict=True)

    missing = [c for c in sig_map if c not in prices and sig_map[c] != 0]
    if missing:
        print(f"[execute] fill_date={fill_date} fiyati yok -> sinyaller atlandi: {missing}")
        for c in missing:
            sig_map[c] = 0

    # Trade tarihi = fill_date (lookahead onleme)
    trades = broker.step(sig_map, prices, date=str(fill_date))

    if save:
        broker.save()

    # konsol özet
    print(f"[execute] {as_of_date} — {len(trades)} emir işlendi")
    for t in trades:
        pnl = f" pnl={t.realized_pnl:+.2f}" if t.realized_pnl else ""
        print(f"  {t.side:11s} {t.coin:4s} qty={t.qty:.6f} @ {t.price:.4f}"
              f"  fee={t.fee:.4f}{pnl}")

    summary = broker.summary(prices) if isinstance(broker, PaperBroker) else {}
    if summary:
        print(f"[execute] cash={summary['cash']}  equity={summary['equity']}  "
              f"open={summary['n_open']}")

    return trades


# ============================================================
# CLI
# ============================================================

def _cli() -> None:
    import argparse, json

    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="YYYY-MM-DD")
    ap.add_argument("--signals", default=None,
                    help="CSV path (coin,signal[,p_buy,p_sell])")
    ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="state/trades yazma")
    args = ap.parse_args()

    if args.signals:
        df = pd.read_csv(args.signals)
    else:
        # orchestrate'ın yazdığı en son signals_*.csv'yi oku
        from paths import LOGS_DIR
        if args.date:
            p = LOGS_DIR / f"signals_{args.date}.csv"
        else:
            cands = sorted(LOGS_DIR.glob("signals_*.csv"))
            if not cands:
                raise SystemExit("signals_*.csv bulunamadı; --signals ver")
            p = cands[-1]
        print(f"[execute] signals: {p}")
        df = pd.read_csv(p)

    as_of = args.date or (
        pd.to_datetime(df["as_of_date"].iloc[0]).date().isoformat()
        if "as_of_date" in df.columns else None
    )

    broker = PaperBroker.load_or_init()
    trades = execute_signals(df, as_of_date=as_of, broker=broker,
                             min_confidence=args.min_conf,
                             save=not args.dry_run)
    print(json.dumps(broker.summary(build_price_map(list(df["coin"]), as_of or "")),
                     indent=2))


if __name__ == "__main__":
    _cli()
