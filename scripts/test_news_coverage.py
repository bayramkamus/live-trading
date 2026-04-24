"""10 coin × 3 kaynak haber kapsama testi.

Yerel olarak:
    python scripts/test_news_coverage.py
    python scripts/test_news_coverage.py --days 7
    python scripts/test_news_coverage.py --days 14 --samples 2

Çıktı: her coin için kaynaklardan kaç haber + dedupe sonrası toplam.
API anahtarı: sentiment_pipeline.py'daki API_KEYS sözlüğü kullanılıyor
(NEWSAPI_KEY hard-coded). Env var override: NEWSAPI_KEY=...
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Windows PowerShell `>` redirect'inde cp1254 (Turkish Windows) codec'i unicode
# ok/emoji karakterlerini yutamayip cokuyor. stdout/stderr'i UTF-8'e sabitle.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd

# code/ paketini PYTHONPATH'e ekle
ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from paths import COINS  # noqa: E402
from sentiment_pipeline import (                                # noqa: E402
    fetch_news_cryptocompare,
    fetch_news_rss,
    fetch_news_newsapi,
)


def _sample_titles(df: pd.DataFrame, n: int = 2) -> str:
    if df.empty:
        return ""
    titles = df["title"].dropna().astype(str).str.strip().head(n).tolist()
    return " | ".join(t[:70] for t in titles)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7,
                    help="kaç gün geriye bak (default: 7)")
    ap.add_argument("--samples", type=int, default=1,
                    help="her kaynaktan kaç başlık göster (default: 1)")
    ap.add_argument("--coins", default="",
                    help="virgüllü liste (default: tüm 10 coin)")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Her kaynak için debug log (HTTP status, count vs)")
    args = ap.parse_args()

    until = datetime.utcnow().date()
    since = until - timedelta(days=args.days)
    since_s, until_s = since.isoformat(), until.isoformat()

    coins = [c.strip().upper() for c in args.coins.split(",") if c.strip()] \
        if args.coins else list(COINS)

    print(f"\n=== Haber kapsama testi ===")
    print(f"Donem: {since_s} -> {until_s}  ({args.days} gun)")
    print(f"Coinler: {coins}")
    print(f"Kaynaklar: CryptoCompare (keysiz) + RSS (6 feed) + NewsAPI (key var)")
    print()

    rows = []
    for coin in coins:
        print(f"-> {coin} ...", flush=True)
        try:
            cc  = fetch_news_cryptocompare(coin, since_s, until_s, verbose=args.verbose)
        except Exception as e:
            print(f"  CC HATA: {e}"); cc = pd.DataFrame()
        try:
            rss = fetch_news_rss(coin, since_s, until_s, verbose=args.verbose)
        except Exception as e:
            print(f"  RSS HATA: {e}"); rss = pd.DataFrame()
        try:
            na  = fetch_news_newsapi(coin, since_s, until_s, verbose=args.verbose)
        except Exception as e:
            print(f"  NA HATA: {e}"); na = pd.DataFrame()

        # Deduped birleşim — pipeline'ın gerçek sonucu
        frames = [f for f in (cc, rss, na) if not f.empty]
        if frames:
            merged = pd.concat(frames, ignore_index=True)
            if "url" in merged.columns:
                merged = merged.drop_duplicates(subset=["url"])
            total = len(merged)
            # günde ortalama
            try:
                days_span = (pd.to_datetime(merged["date"]).nunique()
                             if not merged.empty else 0)
            except Exception:
                days_span = 0
        else:
            merged = pd.DataFrame()
            total = 0
            days_span = 0

        rows.append({
            "coin":      coin,
            "CC":        len(cc),
            "RSS":       len(rss),
            "NewsAPI":   len(na),
            "TOPLAM":    total,
            "gün/coin":  days_span,
            "ort/gün":   round(total / args.days, 1),
            "örnek_başlık": _sample_titles(
                merged if not merged.empty else
                (cc if not cc.empty else (rss if not rss.empty else na)),
                n=args.samples,
            )[:100],
        })
        print(f" CC={len(cc):4d}  RSS={len(rss):4d}  NA={len(na):4d}  -> dedupe {total}")

    df = pd.DataFrame(rows)

    print("\n" + "=" * 90)
    print("ÖZET TABLO")
    print("=" * 90)
    print(df.drop(columns=["örnek_başlık"]).to_string(index=False))

    print("\n" + "=" * 90)
    print("ÖRNEK BAŞLIKLAR")
    print("=" * 90)
    for _, r in df.iterrows():
        print(f"[{r['coin']:5s}] {r['örnek_başlık']}")

    # Kalite sinyali
    print("\n" + "=" * 90)
    print("KALİTE ANALİZİ")
    print("=" * 90)
    weak = df[df["TOPLAM"] < 3]
    ok   = df[(df["TOPLAM"] >= 3) & (df["TOPLAM"] < 10)]
    good = df[df["TOPLAM"] >= 10]
    print(f"  [GOOD] Iyi kapsama (>=10 haber):  {list(good['coin'])}")
    print(f"  [MID]  Orta kapsama (3-9 haber):  {list(ok['coin'])}")
    print(f"  [WEAK] Zayif kapsama (<3 haber):  {list(weak['coin'])}")

    # Kaynak başına toplam
    print(f"\n  Kaynak toplamları:")
    print(f"    CryptoCompare: {df['CC'].sum():5d} haber")
    print(f"    RSS:           {df['RSS'].sum():5d} haber")
    print(f"    NewsAPI:       {df['NewsAPI'].sum():5d} haber")
    print(f"    Dedupe sonra:  {df['TOPLAM'].sum():5d} haber")
    print()

    # Dosyaya yaz
    out = ROOT / "logs" / f"news_coverage_{until_s}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"CSV rapor: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
