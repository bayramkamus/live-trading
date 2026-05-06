"""TradingView Ideas — erişim & parse POC.

Bu script TV /symbols/{COIN}USD/ideas/ sayfalarına HTTP erişimi mümkün mü,
sayfanın yapısını parse edebiliyor muyuz, ölçer. Lokal Windows ve
GitHub Actions Linux ortamlarında çalışır.

Test ettiği şeyler:
  - HTTP status (200 = OK, 403 = Cloudflare blok, 429 = rate limit)
  - Sayfa boyutu (bot blok sayfası genellikle <5KB, gerçek sayfa >100KB)
  - Cloudflare challenge işareti (sayfada "checking your browser" var mı)
  - Idea card sayısı — 3 farklı parser stratejisi:
        1) embedded JSON (window.initialState veya başka init script)
        2) JSON-LD Article schema markup
        3) CSS selectors (TV bunları sık değiştirir, son çare)
  - Sample title (parse başarısı doğrulaması)

Output:
  - Stdout: insan-okur özet tablosu
  - tv_poc_results.json: CI/GHA artifact uyumlu detaylı sonuç
  - --save-html → html-snapshots/{COIN}.html (debug için)

Kullanım (lokal):
    cd C:\\Users\\bayra\\Desktop\\trade\\data
    python live_trading/scripts/test_tradingview_scrape.py
    python live_trading/scripts/test_tradingview_scrape.py --coins BTC ETH SOL --save-html

GHA için: .github/workflows/tv-scrape-poc.yml
    gh workflow run tv-scrape-poc.yml -f coins='BTC ETH SOL'
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    print("WARN: beautifulsoup4 yok — sadece JSON heuristics çalışır")
    print("      Yükle: pip install beautifulsoup4")


COINS_DEFAULT = ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOT", "AVAX", "LINK", "LTC"]

# Realistic browser headers (Chrome 131 Windows). TV'nin Cloudflare'i bot
# tespitinde User-Agent + Sec-Ch-Ua + Accept-Language üçlüsüne bakar.
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not A(Brand";v="24"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

# Cloudflare challenge sayfası tipik işaretleri
CF_MARKERS = (
    "checking your browser",
    "cf-browser-verification",
    "challenge-platform",
    "just a moment...",
    "cf-chl-bypass",
    "ray id:",  # CF blok sayfasının altında genellikle var
)


def fetch_listing(coin: str, timeout: int = 20) -> dict:
    """TV ideas listing sayfasını çek, status + raw HTML döner."""
    url = f"https://www.tradingview.com/symbols/{coin}USD/ideas/"
    rec = {
        "coin": coin,
        "url": url,
        "fetched_at": datetime.utcnow().isoformat() + "Z",
        "status": None,
        "size_bytes": None,
        "size_kb": None,
        "cloudflare_challenge": False,
        "error": None,
        "_html": None,  # underscore: JSON output'tan filtrelenir
    }
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        rec["status"] = r.status_code
        rec["size_bytes"] = len(r.content)
        rec["size_kb"] = round(len(r.content) / 1024, 1)
        rec["_html"] = r.text
        # CF challenge tespiti — küçük sayfa + tipik markerlar
        body_lower = r.text.lower()
        rec["cloudflare_challenge"] = any(m in body_lower for m in CF_MARKERS)
    except requests.exceptions.SSLError as e:
        rec["error"] = f"SSL: {e}"
    except requests.exceptions.ConnectionError as e:
        rec["error"] = f"Conn: {e}"
    except requests.exceptions.Timeout:
        rec["error"] = f"Timeout after {timeout}s"
    except requests.RequestException as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def parse_ideas_idea_card(html: str) -> Optional[dict]:
    """Strateji 1 (PRIMARY): TV'nin React component class adı — ideaCard-{HASH}.

    Class hash deploy başına değişir ama 'ideaCard-' prefix'i sabit.
    Her card içinde title elementi 'title-{HASH}' ya da yakın bir class taşır;
    fallback olarak <a> link metnini kullanırız.
    """
    if not html or not HAS_BS4:
        return None
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.select('[class*="ideaCard-"]')
    if len(cards) < 3:
        return None

    titles = []
    for c in cards[:30]:
        # 1) title-{HASH} class'lı element
        t = c.select_one('[class*="title-"]')
        # 2) fallback: ideaCard içindeki ilk anchor
        if t is None:
            t = c.find("a")
        if t is None:
            continue
        txt = t.get_text(strip=True)[:250]
        if 10 <= len(txt) <= 300:
            titles.append(txt)

    if len(titles) >= 3:
        return {
            "strategy": "idea_card_class",
            "idea_count": len(cards),
            "sample_titles": titles[:5],
        }
    return None


def parse_ideas_json_ld(html: str) -> Optional[dict]:
    """Strateji 2: JSON-LD schema.org Article markup."""
    if not html or not HAS_BS4:
        return None
    soup = BeautifulSoup(html, "html.parser")
    articles = []
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(script.string or script.text or "{}")
        except (json.JSONDecodeError, AttributeError):
            continue
        items = data if isinstance(data, list) else [data]
        for it in items:
            if isinstance(it, dict) and it.get("@type") in ("Article", "NewsArticle", "BlogPosting"):
                articles.append(it)
    if len(articles) >= 3:
        return {
            "strategy": "json_ld",
            "idea_count": len(articles),
            "sample_titles": [a.get("headline", "")[:200] for a in articles[:5]],
        }
    return None


def parse_ideas_css(html: str) -> Optional[dict]:
    """Strateji 3: CSS selector kombinasyonları (en kırılgan, son çare)."""
    if not html or not HAS_BS4:
        return None
    soup = BeautifulSoup(html, "html.parser")
    # TV idea kartları için yaygın selector'lar — periodik olarak değişiyor
    candidates = [
        ('div[data-qa-id="idea-card"]', "h3, h4, [data-qa-id='idea-title'] a"),
        ('article[data-name="idea-card"]', "h2, h3, a"),
        ('[class*="ideaCard-"]', '[class*="title-"], h3'),
        ('article', "h2, h3"),
        ('div.tv-widget-idea', ".tv-widget-idea__title"),
    ]
    for card_sel, title_sel in candidates:
        try:
            cards = soup.select(card_sel)
        except Exception:
            continue
        if len(cards) >= 3:
            titles = []
            for c in cards[:5]:
                try:
                    t = c.select_one(title_sel)
                except Exception:
                    t = None
                if t is None:
                    continue
                txt = t.get_text(strip=True)[:200]
                if 10 <= len(txt) <= 250:
                    titles.append(txt)
            if len(titles) >= 3:
                return {
                    "strategy": f"css:{card_sel}",
                    "idea_count": len(cards),
                    "sample_titles": titles[:5],
                }
    return None


def parse_ideas(html: str) -> dict:
    """3 stratejiyi sırayla dene, ilk çalışanı döner.

    Sıra önemli: idea_card_class en güvenilir (TV'nin gerçek React class'ı),
    JSON-LD ikinci (genelde sayfa-seviye Article markup), CSS son çare.
    """
    out = {"strategy": "none", "idea_count": 0, "sample_titles": []}
    if not html:
        return out
    for parser in (parse_ideas_idea_card, parse_ideas_json_ld, parse_ideas_css):
        result = parser(html)
        if result:
            return result
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="TradingView Ideas POC scraper")
    p.add_argument("--coins", nargs="+", default=COINS_DEFAULT[:3],
                   help=f"Default ilk 3: {COINS_DEFAULT[:3]}. Tümü: --coins {' '.join(COINS_DEFAULT)}")
    p.add_argument("--save-html", action="store_true",
                   help="Ham HTML'i html-snapshots/ klasörüne kaydet (debug)")
    p.add_argument("--rate-limit", type=float, default=2.0,
                   help="İstekler arası saniye (default 2.0, agresif değil)")
    p.add_argument("--timeout", type=int, default=20)
    p.add_argument("--out", type=Path, default=Path("tv_poc_results.json"),
                   help="JSON sonuç dosyası (GHA artifact uyumlu)")
    args = p.parse_args()

    print(f"=== TradingView Ideas POC ===")
    print(f"  Tarih: {datetime.utcnow().isoformat()}Z")
    print(f"  Ortam: {sys.platform}  Python {sys.version.split()[0]}")
    print(f"  bs4: {'evet' if HAS_BS4 else 'YOK (pip install beautifulsoup4)'}")
    print(f"  Coinler: {args.coins}")
    print(f"  Rate limit: {args.rate_limit}s arası")
    print()

    snap_dir = Path("html-snapshots") if args.save_html else None
    if snap_dir:
        snap_dir.mkdir(exist_ok=True)

    results = []
    for i, coin in enumerate(args.coins):
        if i > 0:
            time.sleep(args.rate_limit)
        print(f"[{coin}] {('https://www.tradingview.com/symbols/' + coin + 'USD/ideas/').ljust(60)}")
        fetch = fetch_listing(coin, timeout=args.timeout)
        parse = parse_ideas(fetch.get("_html") or "")

        # HTML snapshot
        if snap_dir and fetch.get("_html"):
            snap_path = snap_dir / f"{coin}.html"
            snap_path.write_text(fetch["_html"], encoding="utf-8")

        rec = {**fetch, **parse}
        rec.pop("_html", None)
        results.append(rec)

        # Renk + ikon
        if rec.get("error"):
            icon = "ERROR"
        elif rec.get("cloudflare_challenge"):
            icon = "BLOCKED"
        elif rec.get("status") != 200:
            icon = f"HTTP {rec.get('status')}"
        elif rec.get("idea_count", 0) >= 3:
            icon = "OK"
        else:
            icon = "PARSE_FAIL"
        print(f"     [{icon:11s}] HTTP {rec.get('status'):>4}  "
              f"{rec.get('size_kb'):>7}KB  CF={'Y' if rec.get('cloudflare_challenge') else 'n'}  "
              f"strategy={rec.get('strategy'):<15s} ideas={rec.get('idea_count'):>3}")
        for t in rec.get("sample_titles", [])[:2]:
            print(f"        - {t[:100]}")
        if rec.get("error"):
            print(f"        ERROR: {rec['error']}")

    # JSON output
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nJSON sonuç: {args.out.resolve()}")
    if snap_dir:
        print(f"HTML snapshots: {snap_dir.resolve()}")

    # Final özet
    n_total = len(results)
    n_ok = sum(1 for r in results
               if r.get("status") == 200
               and not r.get("cloudflare_challenge")
               and r.get("idea_count", 0) >= 3)
    n_cf = sum(1 for r in results if r.get("cloudflare_challenge"))
    n_http_err = sum(1 for r in results if r.get("status") and r.get("status") >= 400)
    n_parse_fail = sum(1 for r in results
                       if r.get("status") == 200
                       and not r.get("cloudflare_challenge")
                       and r.get("idea_count", 0) < 3)

    print(f"\n=== ÖZET ===")
    print(f"  Toplam        : {n_total}")
    print(f"  Tam başarı    : {n_ok}  ({100*n_ok//n_total if n_total else 0}%)")
    print(f"  CF challenge  : {n_cf}")
    print(f"  HTTP error    : {n_http_err}")
    print(f"  Parse fail    : {n_parse_fail}  (HTTP 200 ama idea sayısı <3)")

    print(f"\n=== KARAR ===")
    if n_ok == n_total:
        print(f"  Katman 1 (saf HTTP) GEÇERLİ — production'a güvenle entegre edilebilir.")
        return 0
    elif n_ok >= n_total * 0.7:
        print(f"  Katman 1 KISMEN ÇALIŞIYOR — retry + cache ile production'a alınabilir,")
        print(f"  ama bazı coin'lerde TV verisi olmayacak.")
        return 0
    elif n_cf >= n_total * 0.5:
        print(f"  CF agresif blok ediyor — Katman 2 (Playwright) veya Katman 3 (BrightData) gerekli.")
        return 1
    elif n_parse_fail >= n_total * 0.5:
        print(f"  HTTP geliyor ama parser bulamıyor — TV layout değişmiş, parser_ideas_css güncellenmeli.")
        print(f"  --save-html ile HTML'i dump et, manuel selektör çıkar.")
        return 1
    else:
        print(f"  Karışık sinyal — JSON dosyasını incele, coin başına detaya bak.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
