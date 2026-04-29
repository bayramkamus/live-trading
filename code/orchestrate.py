"""Gunluk orchestrator - tum pipeline'i sirali calistirir.

Akis:
  1) OHLCV + teknik feature guncelle  (ohlcv_fetcher.update_all)
  2) Makro guncelle                    (macro_fetcher.update_macro)
  3) Haber + sentiment guncelle        (sentiment_pipeline.update_all)
  4) 10 coin icin sinyal uret           (inference.predict_signal_for_date)
     - Default as_of_date: en son fully-closed gun - fill_offset (varsayilan 1)
       cron 02:10 UTC: signal_date=T-2, fill_date=T-1 (her ikisi de bilinen gun)
     - Eksik feature (sentiment/macro/tech) -> signal HOLD, reason=missing_features
  5) logs/signals_YYYY-MM-DD.csv yaz
  6) (ops.) sinyalleri PaperBroker'a uygula (--execute)
     - Fill price = fill_date gunu close (strict, lookahead yok)

Calistirma:
  python3 orchestrate.py                       # default T-2 sinyal, T-1 fill
  python3 orchestrate.py --date 2026-04-20     # belirli signal_date
  python3 orchestrate.py --fill-offset 0       # legacy same-day fill
  python3 orchestrate.py --execute             # paper broker'a uygula
  python3 orchestrate.py --skip-update --execute
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from paths import LOGS_DIR, COINS, DATA_LIVE
import ohlcv_fetcher
import macro_fetcher
import sentiment_pipeline
import inference


def _latest_complete_day_utc() -> pd.Timestamp:
    """En son fully-closed UTC gun = (now - 1 gun).normalize().
    Cron 02:10 UTC'de calisirken bu T-1 (yesterday) doner."""
    return (pd.Timestamp.utcnow().tz_localize(None).normalize()
            - pd.Timedelta(days=1))


def _gate_missing_features(out: dict) -> tuple[dict, str | None]:
    """has_* False ise signal HOLD'a dusur + reason dondur."""
    if out.get("has_sent") and out.get("has_macro") and out.get("has_tech"):
        return out, None
    missing = [k for k in ("sent", "macro", "tech") if not out.get(f"has_{k}")]
    out = dict(out)
    out["signal"] = "HOLD"
    out["signal_int"] = 0
    return out, f"missing_features:{','.join(missing)}"


def run(as_of_date: str | None = None, skip_update: bool = False,
        execute: bool = False, min_confidence: float = 0.0,
        fill_offset: int = 1) -> pd.DataFrame:
    """Sinyal uret + (ops.) execute.

    as_of_date:  sinyalin kullandigi feature gunu (T)
    fill_date:   doldurma fiyat gunu (T + fill_offset)
    fill_offset: default 1 (T+1, lookahead-safe). 0 = legacy same-day."""
    last_complete = _latest_complete_day_utc()
    if as_of_date is None:
        signal_date = (last_complete - pd.Timedelta(days=fill_offset)).normalize()
        as_of_date = signal_date.date().isoformat()

    target = pd.Timestamp(as_of_date).normalize()
    fill_date = (target + pd.Timedelta(days=fill_offset)).normalize()

    print(f"[orchestrate] signal_date={target.date()}  "
          f"fill_date={fill_date.date()}  fill_offset={fill_offset}  "
          f"skip_update={skip_update}")

    if not skip_update:
        print("-> OHLCV + teknik guncelle...")
        ohlcv_fetcher.update_all()
        print("-> Makro guncelle...")
        macro_fetcher.update_macro()
        print("-> Sentiment guncelle...")
        sentiment_pipeline.update_all()

    print(f"-> Sinyaller uretiliyor (as_of={target.date()})...")
    rows = []
    errors = []
    n_gated = 0
    for coin in COINS:
        try:
            out = inference.predict_signal_for_date(coin, target)
            out, reason = _gate_missing_features(out)
            out["reason"] = reason or "ok"
            if reason:
                n_gated += 1
                print(f"  [{coin}] {reason} -> HOLD")
            rows.append(out)
        except Exception as e:
            errors.append({"coin": coin, "error": str(e)})
            print(f"  [{coin}] HATA: {e}")

    if not rows:
        raise RuntimeError("Hic sinyal uretilemedi")

    df = pd.DataFrame(rows)
    df["signal_date"] = target.date().isoformat()
    df["fill_date"] = fill_date.date().isoformat()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = LOGS_DIR / f"signals_{target.date().isoformat()}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSinyaller yazildi: {out_csv}")
    show_cols = ["coin","signal","p_sell","p_hold","p_buy","buy_th","sell_th","horizon","reason"]
    print(df[[c for c in show_cols if c in df.columns]].to_string(index=False))
    summary = df["signal"].value_counts().to_dict()
    print(f"\nOzet: {summary}  (gated: {n_gated} coin)")

    if errors:
        err_path = LOGS_DIR / f"errors_{target.date().isoformat()}.json"
        err_path.write_text(json.dumps(errors, indent=2))
        print(f"Hatalar: {err_path}")

    if execute:
        try:
            from execute import execute_signals
            from paper_broker import PaperBroker
            broker = PaperBroker.load_or_init()
            print(f"\n-> PaperBroker execute (fill_date={fill_date.date()})...")
            execute_signals(df, as_of_date=target.date().isoformat(),
                            fill_date=fill_date.date().isoformat(),
                            broker=broker, min_confidence=min_confidence,
                            save=True)
        except Exception as e:
            print(f"[execute] atlandi: {e}")

    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None,
                    help="signal as_of_date YYYY-MM-DD (default: orchestrate auto)")
    ap.add_argument("--fill-offset", type=int, default=1,
                    help="fill_date = as_of_date + N gun (default 1=T+1; 0=same-day)")
    ap.add_argument("--skip-update", action="store_true",
                    help="Veri fetcher'lari atla")
    ap.add_argument("--execute", action="store_true",
                    help="Sinyalleri PaperBroker'a uygula")
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="p_buy/p_sell min guven (0=filtre yok)")
    args = ap.parse_args()
    run(args.date, args.skip_update, execute=args.execute,
        min_confidence=args.min_conf, fill_offset=args.fill_offset)
