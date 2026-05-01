"""Tek seferlik retroaktif replay.

Mevcut data_live/app.db'deki signals tablosundaki p_buy/p_hold/p_sell olasiliklarina
yeni A2 (flat tier-az) kurallarini uygulayip $10K baslangic sermayeli izole bir
paper broker ile simulate eder. Mevcut canli state (data_live/broker/) DOKUNULMAZ.

Output: data_live/replay_<UTCISO>/
    decisions.csv      — her (signal_date, coin) icin A2 sonucu + debug alanlari
    trades.csv         — simulasyondaki tum doldurulmalar (BUY_OPEN, SELL_CLOSE, ...)
    equity.csv         — gunluk equity snapshot
    state_final.json   — son broker state (acik pozisyonlar)
    summary.json       — ozet metrikler
    summary.md         — insan-okur ozet (ayni summary.json'i markdown'da)

Kullanim:
    python scripts/replay_broker_from_signals.py
    python scripts/replay_broker_from_signals.py --start 2026-04-23 --end 2026-04-29
    python scripts/replay_broker_from_signals.py --base-equity 10000 --no-news-gate
    python scripts/replay_broker_from_signals.py --out data_live/replay_test/

Kurallar (flat — coin tier-az):
    SIGNAL_MARGIN  = 0.03   p_dir - p_other_dir >= bu olmali
    HOLD_TOLERANCE = 0.05   p_hold - p_dir <= bu olmali

Aksi halde HOLD (reason: blocked_margin_* / blocked_hold_veto_* / below_threshold).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

# Paths
ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from paths import DATA_LIVE, COINS  # noqa: E402

DB_PATH = DATA_LIVE / "app.db"
OHLCV_DIR = DATA_LIVE / "ohlcv"

# === Kurallar (flat) ===
SIGNAL_MARGIN  = 0.03
HOLD_TOLERANCE = 0.05

# Haber gating (A2'deki ayni)
NEWS_MIN_LAST7D     = 3
NEWS_STALE_DAYS_MAX = 2


# ==========================================================================
# A2 sinyal kurali — flat (tier-az) + debug alanlari
# ==========================================================================

def evaluate_signal_flat(p_sell: float, p_hold: float, p_buy: float,
                          buy_th: float, sell_th: float) -> dict:
    """3-kapi flat kural. Donus: dict(final_signal, reason, signal_margin, hold_gap)."""
    # BUY denemesi
    if p_buy >= buy_th and p_buy > p_sell:
        margin = p_buy - p_sell
        gap = p_hold - p_buy
        if margin < SIGNAL_MARGIN:
            return {"final_signal": "HOLD", "reason": "blocked_margin_buy",
                    "signal_margin": round(margin, 4), "hold_gap": round(gap, 4)}
        if gap > HOLD_TOLERANCE:
            return {"final_signal": "HOLD", "reason": "blocked_hold_veto_buy",
                    "signal_margin": round(margin, 4), "hold_gap": round(gap, 4)}
        return {"final_signal": "BUY", "reason": "buy",
                "signal_margin": round(margin, 4), "hold_gap": round(gap, 4)}
    # SELL denemesi
    if p_sell >= sell_th and p_sell > p_buy:
        margin = p_sell - p_buy
        gap = p_hold - p_sell
        if margin < SIGNAL_MARGIN:
            return {"final_signal": "HOLD", "reason": "blocked_margin_sell",
                    "signal_margin": round(margin, 4), "hold_gap": round(gap, 4)}
        if gap > HOLD_TOLERANCE:
            return {"final_signal": "HOLD", "reason": "blocked_hold_veto_sell",
                    "signal_margin": round(margin, 4), "hold_gap": round(gap, 4)}
        return {"final_signal": "SELL", "reason": "sell",
                "signal_margin": round(margin, 4), "hold_gap": round(gap, 4)}
    # Esik alti
    return {"final_signal": "HOLD", "reason": "below_threshold",
            "signal_margin": 0.0, "hold_gap": 0.0}


# ==========================================================================
# Haber gating
# ==========================================================================

def check_news_gate(coin: str, signal_date: pd.Timestamp) -> Optional[str]:
    """Sentiment parquet'inde 7-gun haber kapsami yetersiz mi?
    Donus: None (ok) | 'low_news_count(...)' | 'stale_news(...)' | 'no_sentiment_row'
    """
    sent_path = DATA_LIVE / "sentiment" / f"{coin}.parquet"
    if not sent_path.exists():
        return "no_sentiment_row"
    try:
        sent = pd.read_parquet(sent_path)
        sent["date"] = pd.to_datetime(sent["date"]).dt.tz_localize(None).dt.normalize()
        row = sent.loc[sent["date"] == signal_date]
        if row.empty:
            return "no_sentiment_row"
        r = row.iloc[0]
        total_raw = r.get("total_news_count", 0)
        days_raw = r.get("days_since_news", pd.NA)
        total = 0.0 if pd.isna(total_raw) else float(total_raw)
        days_since = 999.0 if pd.isna(days_raw) else float(days_raw)
        if total < NEWS_MIN_LAST7D:
            return f"low_news_count({int(total)}<{NEWS_MIN_LAST7D})"
        if days_since > NEWS_STALE_DAYS_MAX:
            return f"stale_news({int(days_since)}d>{NEWS_STALE_DAYS_MAX})"
        return None
    except Exception as e:
        return f"news_gate_err:{e}"


# ==========================================================================
# OHLCV fill price — yerel oncelikli, eksikse Binance fetch
# ==========================================================================

def get_fill_price(coin: str, fill_date: pd.Timestamp,
                   strict: bool = True) -> Optional[float]:
    """fill_date gunu close — once yerel parquet, sonra Binance fetch."""
    p = OHLCV_DIR / f"{coin}_1d.parquet"
    if p.exists():
        try:
            df = pd.read_parquet(p)
            if "close" in df.columns and "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize() \
                    if pd.api.types.is_datetime64tz_dtype(df["date"]) \
                    else pd.to_datetime(df["date"]).dt.normalize()
                row = df.loc[df["date"] == fill_date]
                if not row.empty:
                    return float(row["close"].iloc[0])
                if not strict:
                    past = df.loc[df["date"] <= fill_date]
                    if not past.empty:
                        return float(past.tail(1)["close"].iloc[0])
        except Exception as e:
            print(f"  [warn] yerel parquet okuma {coin}: {e}")

    # Yerel'de yok — Binance fetch
    try:
        from ohlcv_fetcher import fetch_klines, BINANCE_SYMBOLS
        sym = BINANCE_SYMBOLS.get(coin)
        if not sym:
            return None
        start_ms = int(fill_date.timestamp() * 1000) - 86400 * 2 * 1000  # -2 gun
        end_ms   = int(fill_date.timestamp() * 1000) + 86400 * 1 * 1000  # +1 gun
        kl = fetch_klines(sym, "1d", start_ms, end_ms)
        if kl.empty:
            return None
        kl["open_time"] = pd.to_datetime(kl["open_time"]).dt.tz_localize(None).dt.normalize() \
            if pd.api.types.is_datetime64tz_dtype(kl["open_time"]) \
            else pd.to_datetime(kl["open_time"]).dt.normalize()
        match = kl.loc[kl["open_time"] == fill_date]
        if not match.empty:
            return float(match["close"].iloc[0])
        if not strict:
            past = kl.loc[kl["open_time"] <= fill_date]
            if not past.empty:
                return float(past.tail(1)["close"].iloc[0])
    except Exception as e:
        print(f"  [warn] fetch {coin} {fill_date.date()}: {e}")
    return None


# ==========================================================================
# Replay simulasyonu — izole PaperBroker
# ==========================================================================

def run_replay(start: Optional[str] = None,
               end: Optional[str] = None,
               base_equity: float = 10000.0,
               no_news_gate: bool = False,
               out_dir: Optional[Path] = None,
               db_path: Optional[Path] = None) -> Path:
    db_path = db_path or DB_PATH
    if not db_path.exists():
        raise SystemExit(f"DB yok: {db_path}")

    # === 1) Output dizini ===
    if out_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        out_dir = DATA_LIVE / f"replay_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[replay] output: {out_dir}")

    # === 2) Signals oku ===
    where = []
    params: list = []
    if start:
        where.append("date >= ?"); params.append(start)
    if end:
        where.append("date <= ?"); params.append(end)
    where_clause = ("WHERE " + " AND ".join(where)) if where else ""

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    sigs = pd.read_sql_query(
        f"SELECT date, coin, signal AS raw_signal, signal_int AS raw_signal_int, "
        f"p_sell, p_hold, p_buy, buy_th, sell_th, horizon "
        f"FROM signals {where_clause} ORDER BY date ASC, coin ASC",
        conn, params=params,
    )
    conn.close()
    if sigs.empty:
        raise SystemExit("Signals tablosu bos veya pencere disi.")
    sigs["date"] = pd.to_datetime(sigs["date"]).dt.normalize()
    print(f"[replay] {len(sigs)} signal · {sigs['date'].min().date()} → {sigs['date'].max().date()}")

    # === 3) Per-row A2 + haber gate ===
    decisions = []
    for _, r in sigs.iterrows():
        ev = evaluate_signal_flat(r["p_sell"], r["p_hold"], r["p_buy"],
                                    r["buy_th"], r["sell_th"])
        # Haber gating — sadece sinyal aktif iken (final BUY/SELL ise)
        news_block = None
        if not no_news_gate and ev["final_signal"] != "HOLD":
            news_block = check_news_gate(r["coin"], r["date"])
            if news_block:
                ev["final_signal"] = "HOLD"
                ev["reason"] = news_block

        decisions.append({
            "signal_date": r["date"].date().isoformat(),
            "fill_date":   (r["date"] + pd.Timedelta(days=1)).date().isoformat(),
            "coin": r["coin"],
            "raw_signal": r["raw_signal"],
            "final_signal": ev["final_signal"],
            "reason": ev["reason"],
            "signal_margin": ev["signal_margin"],
            "hold_gap": ev["hold_gap"],
            "p_buy": round(r["p_buy"], 4),
            "p_hold": round(r["p_hold"], 4),
            "p_sell": round(r["p_sell"], 4),
            "buy_th": r["buy_th"], "sell_th": r["sell_th"],
            "horizon": int(r["horizon"]) if pd.notna(r["horizon"]) else None,
        })

    dec_df = pd.DataFrame(decisions)
    dec_df.to_csv(out_dir / "decisions.csv", index=False)
    print(f"[replay] decisions.csv yazildi: {len(dec_df)} satir")

    # === 4) Paper broker simulate ===
    # Izole replay state
    from paper_broker import PaperBroker
    state_file = out_dir / "broker" / "state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    pb = PaperBroker.load_or_init(state_file=state_file, base_equity=base_equity)

    # Tarihsel sirayla isle: her tarih icin {coin: signal_int} ve {coin: price}
    dates = sorted(dec_df["signal_date"].unique())
    print(f"[replay] {len(dates)} gun sirayla simulate ediliyor...")

    fill_log = []  # her gun icin price map detay
    for sd in dates:
        day_decs = dec_df[dec_df["signal_date"] == sd]
        fill_date = pd.Timestamp(sd) + pd.Timedelta(days=1)
        # Sinyal map'i: BUY=+1 SELL=-1 HOLD=0
        sig_map = {}
        price_map = {}
        for _, d in day_decs.iterrows():
            fs = d["final_signal"]
            sig_map[d["coin"]] = {"BUY": 1, "SELL": -1, "HOLD": 0}[fs]
            px = get_fill_price(d["coin"], fill_date, strict=True)
            if px is None:
                # T+1 yoksa son available'a düş
                px = get_fill_price(d["coin"], fill_date, strict=False)
            if px is not None:
                price_map[d["coin"]] = px
            fill_log.append({"signal_date": sd, "fill_date": fill_date.date().isoformat(),
                             "coin": d["coin"], "fill_price": px})

        # broker.step() — fill_date tarihiyle
        trades = pb.step(sig_map, price_map, date=fill_date.date().isoformat())
        if trades:
            for t in trades:
                pnl = f" pnl={t.realized_pnl:+.2f}" if t.realized_pnl else ""
                print(f"  {fill_date.date()} {t.side:11s} {t.coin:5s} qty={t.qty:.6f} @ {t.price:.4f}{pnl}")

    pb.save()
    pd.DataFrame(fill_log).to_csv(out_dir / "fill_log.csv", index=False)
    # broker'in trades.csv ve equity.csv'sini out_dir altina kopyala (zaten orada — broker/)
    # PaperBroker.load_or_init iste state_file'a gore broker/ alt dir kullaniyor
    # trades.csv ve equity.csv pb.trades_file ve pb.equity_file'da
    import shutil
    if pb.trades_file != out_dir / "trades.csv":
        shutil.copy2(pb.trades_file, out_dir / "trades.csv")
    if pb.equity_file != out_dir / "equity.csv":
        shutil.copy2(pb.equity_file, out_dir / "equity.csv")

    # === 5) Ozet ===
    trades_df = pd.read_csv(out_dir / "trades.csv") if (out_dir / "trades.csv").exists() else pd.DataFrame()
    equity_df = pd.read_csv(out_dir / "equity.csv") if (out_dir / "equity.csv").exists() else pd.DataFrame()
    state = json.loads(state_file.read_text())

    # FIFO roundtrip eslestirmesi
    rt = match_roundtrips(trades_df) if not trades_df.empty else pd.DataFrame()

    # Acik pozisyon mark-to-market
    mark_prices = {}
    last_fill = pd.Timestamp(dates[-1]) + pd.Timedelta(days=1)
    for coin in state.get("positions", {}).keys():
        px = get_fill_price(coin, last_fill, strict=False)
        if px is not None:
            mark_prices[coin] = px
    unrealized = 0.0
    for c, p in state.get("positions", {}).items():
        mp = mark_prices.get(c, p["avg_price"])
        unrealized += (mp - p["avg_price"]) * p["qty"]
    final_equity = state["cash"] + sum(p["qty"] * mark_prices.get(c, p["avg_price"])
                                        for c, p in state.get("positions", {}).items())

    raw_counts = dec_df["raw_signal"].value_counts().to_dict()
    final_counts = dec_df["final_signal"].value_counts().to_dict()
    reason_counts = dec_df["reason"].value_counts().to_dict()
    n_gated = (dec_df["raw_signal"] != dec_df["final_signal"]).sum()

    summary = {
        "period": {"start": str(sigs["date"].min().date()),
                    "end":   str(sigs["date"].max().date()),
                    "days":  int(sigs["date"].nunique())},
        "signals": {
            "total": int(len(dec_df)),
            "raw":   {k: int(v) for k, v in raw_counts.items()},
            "final": {k: int(v) for k, v in final_counts.items()},
            "n_gated": int(n_gated),
            "reason_breakdown": {k: int(v) for k, v in reason_counts.items()},
        },
        "trades": {
            "total_fills": int(len(trades_df)),
            "by_side": trades_df["side"].value_counts().to_dict() if not trades_df.empty else {},
        },
        "roundtrips": {
            "closed": int(len(rt)),
            "wins":   int((rt["pnl"] > 0).sum()) if not rt.empty else 0,
            "losses": int((rt["pnl"] < 0).sum()) if not rt.empty else 0,
            "win_rate_pct": round(100 * (rt["pnl"] > 0).sum() / max(len(rt), 1), 2) if not rt.empty else 0.0,
            "total_realized_pnl": round(float(rt["pnl"].sum()), 2) if not rt.empty else 0.0,
            "avg_win":  round(float(rt.loc[rt["pnl"] > 0, "pnl"].mean()), 2) if not rt.empty and (rt["pnl"] > 0).any() else 0.0,
            "avg_loss": round(float(rt.loc[rt["pnl"] < 0, "pnl"].mean()), 2) if not rt.empty and (rt["pnl"] < 0).any() else 0.0,
            "profit_factor": (round(float(rt.loc[rt["pnl"] > 0, "pnl"].sum() /
                                         abs(rt.loc[rt["pnl"] < 0, "pnl"].sum())), 2)
                              if not rt.empty and (rt["pnl"] < 0).any() and rt.loc[rt["pnl"] < 0, "pnl"].sum() != 0
                              else None),
            "avg_holding_days": round(float(rt["days"].mean()), 2) if not rt.empty else 0.0,
        },
        "open_positions": [
            {"coin": c, "qty": round(p["qty"], 6), "avg_price": p["avg_price"],
             "mark": mark_prices.get(c), "opened_at": p["opened_at"]}
            for c, p in state.get("positions", {}).items()
        ],
        "pnl": {
            "starting_equity": base_equity,
            "ending_cash":     round(state["cash"], 2),
            "unrealized_at_end": round(unrealized, 2),
            "ending_equity":   round(final_equity, 2),
            "total_return_pct": round(100 * (final_equity - base_equity) / base_equity, 2),
        },
        "config": {
            "SIGNAL_MARGIN":  SIGNAL_MARGIN,
            "HOLD_TOLERANCE": HOLD_TOLERANCE,
            "no_news_gate": no_news_gate,
            "base_equity": base_equity,
        },
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # Markdown ozet
    md = render_summary_md(summary, dec_df, rt, trades_df)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")

    print("\n" + "=" * 70)
    print(md)
    print("=" * 70)
    print(f"\n[replay] tum ciktilar: {out_dir}")
    return out_dir


# ==========================================================================
# FIFO roundtrip (ayni paper_broker formati)
# ==========================================================================

def match_roundtrips(tr: pd.DataFrame) -> pd.DataFrame:
    if tr.empty:
        return pd.DataFrame(columns=["coin","side","open_date","close_date","qty",
                                       "entry","exit","days","pnl","pnl_pct"])
    tr2 = tr.copy()
    tr2["date"] = pd.to_datetime(tr2["date"], errors="coerce")
    tr2 = tr2.sort_values(["coin", "date"]).reset_index(drop=True)
    from collections import deque, defaultdict
    long_q  = defaultdict(deque)
    short_q = defaultdict(deque)
    rows = []

    def _close(queue, direction, coin, qty, px, dt):
        rem = qty
        while rem > 1e-12 and queue:
            o_dt, o_q, o_px = queue[0]
            take = min(o_q, rem)
            if direction == "LONG":
                pnl = (px - o_px) * take
                pct = 100*(px/o_px - 1) if o_px else 0
            else:
                pnl = (o_px - px) * take
                pct = 100*(o_px/px - 1) if px else 0
            try:
                days = max((dt.normalize() - o_dt.normalize()).days, 0) \
                    if pd.notna(dt) and pd.notna(o_dt) else 0
            except Exception:
                days = 0
            rows.append({
                "coin": coin, "side": direction,
                "open_date": o_dt.date().isoformat() if pd.notna(o_dt) else "",
                "close_date": dt.date().isoformat() if pd.notna(dt) else "",
                "qty": round(take, 6),
                "entry": o_px, "exit": px,
                "days": int(days),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pct, 3),
            })
            queue[0][1] = o_q - take
            if queue[0][1] <= 1e-12:
                queue.popleft()
            rem -= take

    for _, r in tr2.iterrows():
        side = str(r.get("side","")).upper().strip()
        coin = str(r["coin"]); q = float(r.get("qty",0) or 0); px = float(r.get("price",0) or 0); dt = r["date"]
        if q <= 0: continue
        if side == "BUY_OPEN":  long_q[coin].append([dt, q, px])
        elif side == "SELL_OPEN": short_q[coin].append([dt, q, px])
        elif side == "SELL_CLOSE": _close(long_q[coin], "LONG",  coin, q, px, dt)
        elif side == "BUY_COVER":  _close(short_q[coin], "SHORT", coin, q, px, dt)
    return pd.DataFrame(rows)


# ==========================================================================
# Markdown ozet
# ==========================================================================

def render_summary_md(summary: dict, dec_df: pd.DataFrame,
                       rt: pd.DataFrame, tr: pd.DataFrame) -> str:
    s = summary
    lines = [
        f"# Replay raporu — {s['period']['start']} → {s['period']['end']} ({s['period']['days']} gun)",
        "",
        f"**Kural:** SIGNAL_MARGIN={s['config']['SIGNAL_MARGIN']}, "
        f"HOLD_TOLERANCE={s['config']['HOLD_TOLERANCE']}, "
        f"news_gate={'OFF' if s['config']['no_news_gate'] else 'ON'}",
        f"**Baslangic sermayesi:** ${s['config']['base_equity']:,.2f}",
        "",
        "## Sinyal istatistikleri",
        f"- Toplam: {s['signals']['total']}",
        f"- Ham (orijinal DB): {s['signals']['raw']}",
        f"- Final (A2 sonrasi): {s['signals']['final']}",
        f"- Gated (raw≠final): **{s['signals']['n_gated']}**",
        "",
        "**Reason kirilim:**",
    ]
    for k, v in sorted(s['signals']['reason_breakdown'].items(), key=lambda x: -x[1]):
        lines.append(f"- `{k}`: {v}")
    lines += [
        "",
        "## Trade istatistikleri",
        f"- Toplam fill: **{s['trades']['total_fills']}**",
        f"- Side dagilimi: {s['trades']['by_side']}",
        "",
        "## Kapanmis pozisyonlar (FIFO roundtrip)",
        f"- Kapanan: **{s['roundtrips']['closed']}**",
        f"- Win/Loss: {s['roundtrips']['wins']}W / {s['roundtrips']['losses']}L "
        f"(rate: {s['roundtrips']['win_rate_pct']}%)",
        f"- Toplam realized PnL: **${s['roundtrips']['total_realized_pnl']:+,.2f}**",
        f"- Ortalama kazanc / kayip: ${s['roundtrips']['avg_win']:+,.2f} / ${s['roundtrips']['avg_loss']:+,.2f}",
        f"- Profit factor: {s['roundtrips']['profit_factor']}",
        f"- Ortalama tutma: {s['roundtrips']['avg_holding_days']} gun",
        "",
        "## Acik pozisyonlar (simulasyon sonu)",
    ]
    if s['open_positions']:
        for op in s['open_positions']:
            lines.append(f"- {op['coin']} qty={op['qty']:.6f} @ ${op['avg_price']:.4f} "
                         f"(mark=${op['mark']}) opened {op['opened_at']}")
    else:
        lines.append("- (yok)")
    lines += [
        "",
        "## Net PnL",
        f"- Starting equity: ${s['pnl']['starting_equity']:,.2f}",
        f"- Ending cash: ${s['pnl']['ending_cash']:,.2f}",
        f"- Unrealized at end: ${s['pnl']['unrealized_at_end']:+,.2f}",
        f"- **Ending equity: ${s['pnl']['ending_equity']:,.2f}** ({s['pnl']['total_return_pct']:+.2f}%)",
    ]
    return "\n".join(lines)


# ==========================================================================
# CLI
# ==========================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None, help="signal_date >= YYYY-MM-DD")
    ap.add_argument("--end",   default=None, help="signal_date <= YYYY-MM-DD")
    ap.add_argument("--base-equity", type=float, default=10000.0)
    ap.add_argument("--no-news-gate", action="store_true",
                    help="Haber kapsam gating'i kapat")
    ap.add_argument("--out", default=None,
                    help="Output dizini (default: data_live/replay_<UTC-ISO>/)")
    ap.add_argument("--db", default=None,
                    help="Alternatif DB yolu (default: data_live/app.db)")
    args = ap.parse_args()

    out = Path(args.out) if args.out else None
    db_p = Path(args.db) if args.db else None
    run_replay(start=args.start, end=args.end,
               base_equity=args.base_equity,
               no_news_gate=args.no_news_gate,
               out_dir=out, db_path=db_p)
