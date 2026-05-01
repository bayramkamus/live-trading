"""Günlük otomasyon — GitHub Actions ya da cron tarafından çağrılır.

Sıra:
  1) ohlcv_fetcher.update_all       → tech + ham OHLCV (parquet + DB)
  2) macro_fetcher.update_macro     → macro (parquet + DB)
  3) sentiment_pipeline.update_all  → sentiment (parquet + DB)
  4) inference + execute_signals    → signals + trades + equity (DB)
  5) db.prune_old(days=30, cache_days=60)

Kullanım (yerel veya CI):
    python scripts/daily_run.py
    python scripts/daily_run.py --date 2026-04-23
    python scripts/daily_run.py --skip-update --no-execute     # sadece re-inference
    python scripts/daily_run.py --dry-run                      # DB yazma, log et

Env değişkenleri:
    FRED_API_KEY       — macro_fetcher için
    NEWSAPI_KEY        — (opsiyonel, kod içinde default var)
    PAPER_BASE_EQUITY  — broker başlangıç sermayesi (default 10000)
    PAPER_RISK_PCT     — işlem başına equity yüzdesi (default 0.10)
    MIN_CONFIDENCE     — p_buy/p_sell filtresi (default 0.0)

Çıktılar:
    data_live/app.db    — tüm kalıcı veri
    logs/daily_YYYY-MM-DD.log
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# code/ paketini PYTHONPATH'e ekle
ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from paths import LOGS_DIR, COINS, DATA_LIVE, TECH_DIR, SENT_DIR, MACRO_DIR  # noqa: E402
from db import DB                                                              # noqa: E402
import config as cfg                                                           # noqa: E402


def _log(msg: str, log_file: Optional[Path] = None) -> None:
    ts = datetime.utcnow().isoformat(timespec="seconds")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if log_file is not None:
        try:
            with log_file.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


def _sync_caches_to_db(db: DB) -> dict:
    """Parquet cache'lerini DB'ye mirror et (son 60 gün).

    Not: pd.Timestamp.utcnow() pandas 2.x'te tz-aware UTC dondurur. Parquet
    dosyalarindaki 'date' kolonu tz-naive olabilir, kiyaslamada hata olmasin
    diye cutoff'u tz_localize(None) ile naive yapariz.
    """
    counts = {"ohlcv": 0, "tech": 0, "sent": 0, "macro": 0}
    cutoff = (pd.Timestamp.utcnow().tz_localize(None).normalize()
              - pd.Timedelta(days=60))

    # OHLCV
    ohlcv_dir = DATA_LIVE / "ohlcv"
    if ohlcv_dir.exists():
        for p in sorted(ohlcv_dir.glob("*_1d.parquet")):
            coin = p.stem.replace("_1d", "")
            try:
                df = pd.read_parquet(p)
                df["date"] = pd.to_datetime(df["date"])
                df = df[df["date"] >= cutoff]
                db.write_ohlcv(coin, df)
                counts["ohlcv"] += len(df)
            except Exception as e:
                _log(f"[sync] ohlcv {coin}: {e}")

    # Tech
    for p in sorted(TECH_DIR.glob("*.parquet")):
        coin = p.stem
        try:
            df = pd.read_parquet(p)
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["date"] >= cutoff]
            db.write_features("tech_cache", coin, df)
            counts["tech"] += len(df)
        except Exception as e:
            _log(f"[sync] tech {coin}: {e}")

    # Sent
    for p in sorted(SENT_DIR.glob("*.parquet")):
        coin = p.stem
        try:
            df = pd.read_parquet(p)
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["date"] >= cutoff]
            db.write_features("sent_cache", coin, df)
            counts["sent"] += len(df)
        except Exception as e:
            _log(f"[sync] sent {coin}: {e}")

    # Macro
    mp = MACRO_DIR / "macro.parquet"
    if mp.exists():
        try:
            df = pd.read_parquet(mp)
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["date"] >= cutoff]
            db.write_features("macro_cache", None, df)
            counts["macro"] += len(df)
        except Exception as e:
            _log(f"[sync] macro: {e}")

    return counts


def _sync_broker_to_db(db: DB) -> None:
    """PaperBroker'ın JSON/CSV'sini DB'ye yansıt."""
    import paper_broker as pb_mod

    # trades.csv'yi okuyup yeni trade'leri ekle (last id'e kadar seen)
    last_trade_id = db.get_meta("last_broker_trade_row")
    last_seen = int(last_trade_id) if last_trade_id and last_trade_id.isdigit() else 0

    if pb_mod.TRADES_FILE.exists():
        df = pd.read_csv(pb_mod.TRADES_FILE)
        new = df.iloc[last_seen:]
        for _, r in new.iterrows():
            db.append_trade({
                "date": str(r["date"]), "coin": str(r["coin"]),
                "side": str(r["side"]), "qty": float(r["qty"]),
                "price": float(r["price"]), "fee": float(r["fee"]),
                "gross": float(r["gross"]),
                "realized_pnl": float(r.get("realized_pnl", 0) or 0),
                "note": str(r.get("note", "")),
            })
        db.set_meta("last_broker_trade_row", str(len(df)))

    # equity.csv → son 7 gün DB'ye yansıt
    if pb_mod.EQUITY_FILE.exists():
        eq = pd.read_csv(pb_mod.EQUITY_FILE)
        for _, r in eq.iterrows():
            db.append_equity(str(r["date"]), float(r["cash"]),
                             float(r["positions_value"]),
                             float(r["equity"]), int(r["n_open"]))

    # positions: state.json
    if pb_mod.STATE_FILE.exists():
        js = json.loads(pb_mod.STATE_FILE.read_text())
        positions = list(js.get("positions", {}).values())
        db.replace_positions(positions)


def run(as_of_date: Optional[str] = None,
        skip_update: bool = False,
        execute: bool = True,
        dry_run: bool = False,
        fill_offset: int = cfg.FILL_OFFSET) -> int:
    """0 dönerse başarılı, >0 ise hatalı çıkış kodu.

    fill_offset: T+1 fill için varsayılan 1.
                 0 verirsen same-day (legacy) fill — debug için.
    """

    # Log dosyası — orchestrate.run() içeriden as_of_date'i çözer (None → T-1-fill_offset)
    run_tag = as_of_date or pd.Timestamp.utcnow().date().isoformat()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"daily_{run_tag}.log"

    _log(f"=== daily_run  run_tag={run_tag}  fill_offset={fill_offset}  "
         f"skip_update={skip_update}  execute={execute}  dry_run={dry_run} ===", log_file)

    errors = []

    # 1) fetcher'lar
    if not skip_update:
        import ohlcv_fetcher, macro_fetcher, sentiment_pipeline  # noqa
        for step_name, fn in [
            ("ohlcv", lambda: ohlcv_fetcher.update_all()),
            ("macro", lambda: macro_fetcher.update_macro()),
            ("sentiment", lambda: sentiment_pipeline.update_all()),
        ]:
            try:
                _log(f"→ {step_name} update...", log_file)
                fn()
            except Exception as e:
                msg = f"[{step_name}] HATA: {e}\n{traceback.format_exc()}"
                _log(msg, log_file)
                errors.append({"step": step_name, "error": str(e)})

    # 2) inference — orchestrate.run() signal_date/fill_date'i kendi seçer
    import orchestrate  # noqa
    df = pd.DataFrame()
    signal_date = run_tag
    try:
        _log("→ inference + execute...", log_file)
        df = orchestrate.run(as_of_date=as_of_date,   # None → orchestrate çözer
                             skip_update=True,
                             execute=execute,
                             min_confidence=cfg.MIN_CONFIDENCE,
                             fill_offset=fill_offset,
                             replay=False)
        if not df.empty and "signal_date" in df.columns:
            signal_date = str(df["signal_date"].iloc[0])
    except Exception as e:
        _log(f"[inference] HATA: {e}\n{traceback.format_exc()}", log_file)
        errors.append({"step": "inference", "error": str(e)})

    # A4: replay = kullanici --date <past> ile MANUEL gecmis tarih verdi
    # Default cron run'inda as_of_date None'dur (orchestrate kendi cozer T-2'ye);
    # bu durum replay sayilmaz, main DB'ye yazilir.
    today_utc = pd.Timestamp.utcnow().tz_localize(None).normalize().date().isoformat()
    is_replay = bool(as_of_date) and as_of_date < today_utc

    # 3) DB'ye yansit
    if not dry_run and not is_replay:
        try:
            with DB() as db:
                db.init()
                # signals + decisions
                if not df.empty:
                    n = db.write_signals(df, as_of_date=signal_date)
                    _log(f"[db] signals upsert: {n}  signal_date={signal_date}", log_file)
                    # A4: per-coin broker_decisions
                    n_dec = 0
                    for _, r in df.iterrows():
                        db.write_decision(
                            date=signal_date,
                            coin=str(r["coin"]),
                            raw_signal=str(r.get("raw_signal", r.get("signal", "HOLD"))),
                            raw_signal_int=int(r.get("raw_signal_int", r.get("signal_int", 0)) or 0),
                            final_action=str(r.get("signal", "HOLD")),
                            reason=str(r.get("reason", "ok")),
                            model_version=str(r["model_version"]) if pd.notna(r.get("model_version")) else None,
                        )
                        n_dec += 1
                    _log(f"[db] broker_decisions upsert: {n_dec}", log_file)
                # cache mirror
                counts = _sync_caches_to_db(db)
                _log(f"[db] cache mirror: {counts}", log_file)
                # broker
                if execute:
                    _sync_broker_to_db(db)
                    _log("[db] broker state sync edildi", log_file)
                # retention
                pruned = db.prune_old(days=30, cache_days=60)
                _log(f"[db] retention: {pruned}", log_file)
                # meta
                db.set_meta("last_run", datetime.utcnow().isoformat(timespec="seconds"))
                db.set_meta("last_date", signal_date)
                # A4: aktif model versiyonu
                try:
                    sys.path.insert(0, str(CODE))
                    from model_version import get_global_version
                    gv = get_global_version()
                    db.set_meta("last_model_version", gv.get("global_hash") or "")
                    db.set_meta("last_model_created_at", gv.get("created_at") or "")
                except Exception:
                    pass
        except Exception as e:
            _log(f"[db] HATA: {e}\n{traceback.format_exc()}", log_file)
            errors.append({"step": "db", "error": str(e)})
    elif is_replay:
        _log(f"[db] REPLAY mode (signal_date={signal_date} < {today_utc}) — main DB yazilmadi", log_file)

    # 4) özet
    if errors:
        _log(f"=== HATA: {len(errors)} adım başarısız ===", log_file)
        (LOGS_DIR / f"errors_{run_tag}.json").write_text(
            json.dumps(errors, indent=2, ensure_ascii=False))
        return 1

    _log("=== OK ===", log_file)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None,
                    help="signal as_of_date YYYY-MM-DD (default: orchestrate auto)")
    ap.add_argument("--fill-offset", type=int, default=cfg.FILL_OFFSET,
                    help="fill_date = as_of_date + N gun (default 1=T+1; 0=same-day)")
    ap.add_argument("--skip-update", action="store_true",
                    help="Veri fetcher'lari atla")
    ap.add_argument("--no-execute", action="store_true",
                    help="PaperBroker'a uygulama")
    ap.add_argument("--dry-run", action="store_true",
                    help="DB / broker yazma; sadece signals_*.csv ve log uret")
    args = ap.parse_args()

    rc = run(as_of_date=args.date,
             skip_update=args.skip_update,
             execute=not args.no_execute,
             dry_run=args.dry_run,
             fill_offset=args.fill_offset)
    sys.exit(rc)
