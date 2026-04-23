"""Günlük orchestrator — tüm pipeline'ı sıralı çalıştırır.

Akış:
  1) OHLCV + teknik feature güncelle  (ohlcv_fetcher.update_all)
  2) Makro güncelle                    (macro_fetcher.update_macro)
  3) Haber + sentiment güncelle        (sentiment_pipeline.update_all)
  4) 10 coin için bugünkü sinyal üret  (inference.predict_signal_for_date)
  5) logs/signals_YYYY-MM-DD.csv yaz
  6) (ops.) sinyalleri PaperBroker'a uygula (--execute)

Çalıştırma:
  python3 orchestrate.py                       # bugün için
  python3 orchestrate.py --date 2026-04-20     # belirli tarih
  python3 orchestrate.py --execute             # sinyalleri paper broker'a uygula
  python3 orchestrate.py --skip-update --execute

Cron örneği (UTC):
  0 2 * * *  cd /path/to/live_trading/code && python3 orchestrate.py --execute
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from paths import LOGS_DIR, COINS
import ohlcv_fetcher
import macro_fetcher
import sentiment_pipeline
import inference


def run(as_of_date: str | None = None, skip_update: bool = False,
        execute: bool = False, min_confidence: float = 0.0) -> pd.DataFrame:
    if as_of_date is None:
        as_of_date = pd.Timestamp.utcnow().date().isoformat()
    target = pd.Timestamp(as_of_date)

    print(f"[orchestrate] tarih: {target.date()}  skip_update={skip_update}")

    if not skip_update:
        print("→ OHLCV + teknik güncelle...")
        ohlcv_fetcher.update_all()
        print("→ Makro güncelle...")
        macro_fetcher.update_macro()
        print("→ Sentiment güncelle...")
        sentiment_pipeline.update_all()

    print(f"→ Sinyaller üretiliyor ({target.date()})...")
    rows = []
    errors = []
    for coin in COINS:
        try:
            out = inference.predict_signal_for_date(coin, target)
            rows.append(out)
        except Exception as e:
            errors.append({"coin": coin, "error": str(e)})
            print(f"  [{coin}] HATA: {e}")

    if not rows:
        raise RuntimeError("Hiç sinyal üretilemedi")

    df = pd.DataFrame(rows)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = LOGS_DIR / f"signals_{target.date().isoformat()}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSinyaller yazıldı: {out_csv}")
    print(df[["coin","signal","p_sell","p_hold","p_buy","buy_th","sell_th","horizon"]].to_string(index=False))
    print(f"\nÖzet: {df['signal'].value_counts().to_dict()}")

    if errors:
        err_path = LOGS_DIR / f"errors_{target.date().isoformat()}.json"
        err_path.write_text(json.dumps(errors, indent=2))
        print(f"Hatalar: {err_path}")

    if execute:
        try:
            from execute import execute_signals
            from paper_broker import PaperBroker
            broker = PaperBroker.load_or_init()
            print("\n→ PaperBroker execute...")
            execute_signals(df, as_of_date=target.date().isoformat(),
                            broker=broker, min_confidence=min_confidence,
                            save=True)
        except Exception as e:
            print(f"[execute] atlandı: {e}")

    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: bugün UTC)")
    ap.add_argument("--skip-update", action="store_true",
                    help="Veri fetcher'ları atla, yalnız sinyal üret")
    ap.add_argument("--execute", action="store_true",
                    help="Sinyalleri PaperBroker'a uygula")
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="p_buy/p_sell min güven (0 → filtre yok)")
    args = ap.parse_args()
    run(args.date, args.skip_update, execute=args.execute,
        min_confidence=args.min_conf)
