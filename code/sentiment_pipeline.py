"""Canlı sentiment pipeline — haber → NLP head → günlük feature parquet.

Akış:
  1) fetch_news(coin, since, until)  → CryptoPanic / NewsAPI / RSS
  2) NLPEnsemble.score_batch(texts)  → FinBERT + CryptoBERT + J-Hartmann
                                      combined_score = J-Hartmann (winner of V4
                                      NLP combo ablation on S1). fb/cb scores
                                      logged for diagnostics only.
  3) aggregate_daily(scored, coin)   → günlük tek skor + ratio + count
  4) add_lag_rolling(daily)          → lag 1/2/3, rolling 3/7, news rolling
  5) update_coin(coin)               → eksik günler: fetch → score → aggregate → historical ile concat → parquet

Anahtarlar aşağıdaki API_KEYS bloğunda. Env var da set'liyse env öncelikli.
En az bir API anahtarı gerekli; ikisi de yoksa sadece historical yazılır.
"""
from __future__ import annotations

import os
import time
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests

try:
    from bs4 import BeautifulSoup
    HAS_BS4_LIVE = True
except ImportError:
    HAS_BS4_LIVE = False

from paths import SENT_DIR, COINS, HISTORICAL_DATA_ROOT


# ============================================================
# KAYNAK ÖNCELİĞİ
# ============================================================
# 1) CryptoCompare News API — ÜCRETSİZ, anahtar gerekmiyor, coin-tagged.
#    Tek zayıflık: sadece son ~50-200 haber. Günlük çalıştırırsak yeterli.
# 2) RSS — sınırsız, ücretsiz, son ~50 haber/kaynak; coin-tagging yok,
#    client-side keyword filtre.
# 3) NewsAPI — 100 req/gün, 30 gün geçmiş. Büyük backfill için.
# 4) CryptoPanic — KAPALI (ücretsiz tier bitti).

API_KEYS = {
    # https://newsapi.org/   (ücretsiz tier: günde 100 req, 30 gün geçmiş)
    "NEWSAPI_KEY":       "1c1ed95bb65a46cdbb61033ff14a1197",

    # https://min-api.cryptocompare.com  (ücretsiz tier: 100k req/ay, key'siz
    # tier 3k/gün ve anonim IP'lerden bazen engelleniyor — key ile daha güvenli)
    "CRYPTOCOMPARE_KEY": "f4e58b5095263d9bcad030bdd345e99f71af25e276d551978ed77824f7a98a17",

    # Şu an kapalı — ileri ki kullanım için yer tutucu
    "CRYPTOPANIC_TOKEN": "",
}

# Kaynak açma/kapama bayrakları
USE_CRYPTOCOMPARE = True   # önerilen birincil kaynak
USE_RSS           = True   # fallback/supplement (sınırsız)
USE_NEWSAPI       = True   # anahtar varken açık
USE_CRYPTOPANIC   = False  # ücretsiz tier bitti — kapalı
USE_TRADINGVIEW   = True   # POC kanıtladı (5/5 başarı, GHA Linux + lokal)
TRADINGVIEW_FETCH_BODIES = False  # True → her idea'nın detail page body'sini de çek (yavaş)

# ============================================================
# SECTION ETIKETLEMESI
# ============================================================
# Historical V2 training datasıyla uyumlu olması için her kaynak bir
# section'a atanır. aggregate_daily section başına ayrı (avg_score,
# pos/neu/neg ratio, news_count) hesaplar; combined_score IC-weighted
# karışımdır (training-time t3_section_weights ile uyumlu).
#
#   section 1 — RSS (Coindesk, Cointelegraph, Decrypt vb. editörlü kaynaklar)
#   section 2 — Haber API'leri (CryptoCompare, NewsAPI, CryptoPanic)
#   section 3 — TradingView Ideas (community trader analizleri)
SOURCE_SECTION = {
    "cryptocompare":     2,
    "newsapi":           2,
    "cryptopanic":       2,
    # RSS feedleri — domain bazlı tüm RSS kaynakları section_1'e gider
    "coindesk.com":      1,
    "cointelegraph.com": 1,
    "decrypt.co":        1,
    "bitcoinmagazine.com": 1,
    "cryptoslate.com":   1,
    "theblock.co":       1,
    # TradingView Ideas
    "tradingview_ideas": 3,
}


def _section_for_source(source: str) -> int:
    """source string'inden section numarasını çıkar. Bilinmeyen → 2 (haber default)."""
    if not source:
        return 2
    s = source.lower().strip()
    if s in SOURCE_SECTION:
        return SOURCE_SECTION[s]
    # RSS kaynakları domain.com formatında geliyor — substring match
    for key, sec in SOURCE_SECTION.items():
        if key in s:
            return sec
    return 2  # default haber

# RSS kaynakları — crypto odaklı, coin adı ile client-side filtre
RSS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
    "https://bitcoinmagazine.com/.rss/full/",
    "https://cryptoslate.com/feed/",
    "https://www.theblock.co/rss.xml",
]

# NLP cihaz + batch
NLP_DEVICE     = "cpu"   # "cuda" (GPU varsa), "cpu", veya None (otomatik)
NLP_BATCH_SIZE = 16


def _get_key(name: str) -> str:
    """Env var > API_KEYS dict. Placeholder değerleri boş kabul edilir."""
    v = os.environ.get(name) or API_KEYS.get(name, "")
    if not v or v.startswith("PASTE_YOUR_"):
        return ""
    return v


# ============================================================
# 1. HABER ÇEKİMİ
# ============================================================

# CoinGecko ID eşlemesi (bazı API'ler sembol değil tam isim ister)
COIN_FULLNAME = {
    "BTC": "bitcoin",   "ETH": "ethereum",  "BNB": "binancecoin",
    "SOL": "solana",    "XRP": "ripple",    "ADA": "cardano",
    "DOT": "polkadot",  "AVAX": "avalanche","LINK": "chainlink",
    "LTC": "litecoin",
}


def fetch_news_cryptopanic(coin: str, since: str, until: str,
                           max_pages: int = 5) -> pd.DataFrame:
    """CryptoPanic /api/v1/posts/ — public access news.

    API sadece en son ~200 haberi döner, tarih filtresi zayıf;
    o yüzden since/until bizzat client tarafında filtrelenir.
    """
    token = _get_key("CRYPTOPANIC_TOKEN")
    if not token:
        return pd.DataFrame()

    url = "https://cryptopanic.com/api/v1/posts/"
    params = {
        "auth_token": token,
        "currencies": coin,
        "public": "true",
        "kind": "news",
    }
    rows: List[dict] = []
    try:
        for page in range(1, max_pages + 1):
            params["page"] = page
            r = requests.get(url, params=params, timeout=15)
            if r.status_code != 200:
                break
            js = r.json()
            results = js.get("results", [])
            if not results:
                break
            for it in results:
                created = it.get("created_at", "")[:10]
                if not created:
                    continue
                rows.append({
                    "date":   created,
                    "title":  it.get("title", "") or "",
                    "body":   it.get("description", "") or "",
                    "source": (it.get("source") or {}).get("title", "cryptopanic"),
                    "url":    it.get("url", ""),
                })
            # En eski dönen tarih hedefin gerisindeyse durduralım
            if results and results[-1].get("created_at", "")[:10] < since:
                break
            time.sleep(0.3)  # rate-limit saygısı
    except Exception as e:
        warnings.warn(f"CryptoPanic hata: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    df = df[(df["date"] >= since) & (df["date"] <= until)]
    return df.reset_index(drop=True)


# NewsAPI için crypto domain whitelist — coin sembol ambiguity'sini elimine eder.
# (ör. "ADA" → NVIDIA Ada Lovelace, "DOT" → emlak, "LINK" → NFL...)
NEWSAPI_CRYPTO_DOMAINS = ",".join([
    "coindesk.com",       "cointelegraph.com",  "decrypt.co",
    "theblock.co",        "bitcoinmagazine.com", "cryptoslate.com",
    "cryptobriefing.com", "u.today",             "ambcrypto.com",
    "cryptonews.com",     "cryptopotato.com",    "newsbtc.com",
    "beincrypto.com",     "bitcoinist.com",      "coinjournal.net",
    "cryptodaily.co.uk",  "livecoin.com",        "crypto.news",
])


def fetch_news_newsapi(coin: str, since: str, until: str,
                       page_size: int = 100,
                       verbose: bool = False) -> pd.DataFrame:
    """NewsAPI /v2/everything — 100 req/day free tier.

    İki eşzamanlı filtre:
      1) Sorgu: "{fullname}" AND (crypto OR cryptocurrency OR blockchain)
         → "Ada", "Dot", "Link" gibi ambiguous sembollerin başka alanlara
           kaymasını engeller.
      2) domains whitelist: sadece crypto sitelerinden çek.
    Bu ikisiyle NewsAPI artık alakasız NFL/emlak/grafik-kartı haberleri çekmez.
    """
    key = _get_key("NEWSAPI_KEY")
    if not key:
        if verbose: print("  [NewsAPI] key yok, atla")
        return pd.DataFrame()

    fullname = COIN_FULLNAME.get(coin, coin.lower())
    # "bitcoin" AND (crypto OR cryptocurrency OR blockchain OR token)
    # Fullname zaten çoğunlukla tek başına yeterli (bitcoin, ethereum...);
    # ama mid-cap için (cardano, polkadot, chainlink) NewsAPI relevance skoru
    # yine kripto-dışı sonuç verebiliyordu. "AND crypto" + domain whitelist sağlam çözüm.
    q = f'"{fullname}" AND (crypto OR cryptocurrency OR blockchain OR token)'
    params = {
        "q": q,
        "from": since,
        "to": until,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "domains": NEWSAPI_CRYPTO_DOMAINS,
        "apiKey": key,
    }
    rows: List[dict] = []
    try:
        r = requests.get("https://newsapi.org/v2/everything", params=params, timeout=15)
        if r.status_code != 200:
            if verbose: print(f"  [NewsAPI] HTTP {r.status_code}: {r.text[:150]}")
            return pd.DataFrame()
        js = r.json()
        articles = js.get("articles", [])
        if verbose: print(f"  [NewsAPI] {coin}: {len(articles)} article")
        for it in articles:
            published = (it.get("publishedAt") or "")[:10]
            if not published:
                continue
            # Başlık + body'de coin adının geçtiğinden emin ol (ikinci güvenlik)
            title = it.get("title", "") or ""
            body  = it.get("description", "") or ""
            blob = (title + " " + body).lower()
            # Crypto/coin kelimesi en az bir kere geçmeli
            if fullname.lower() not in blob and coin.lower() not in blob:
                continue
            rows.append({
                "date":   published,
                "title":  title,
                "body":   body,
                "source": (it.get("source") or {}).get("name", "newsapi"),
                "url":    it.get("url", ""),
            })
    except Exception as e:
        warnings.warn(f"NewsAPI hata: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[(df["date"] >= since) & (df["date"] <= until)]
    return df.reset_index(drop=True)


# ---------- CryptoCompare News API (ÜCRETSİZ, coin-tagged) ----------

def fetch_news_cryptocompare(coin: str, since: str, until: str,
                             max_pages: int = 5,
                             verbose: bool = False) -> pd.DataFrame:
    """CryptoCompare /data/v2/news/ — key opsiyonel ama önerilir.

    categories=BTC,ETH... coin-tagged filtre yapar. Her istek son ~50 haber;
    lTs (timestamp) ile sayfa-sayfa geçmişe doğru gidebiliriz.

    Key yoksa anonim tier çalışır (günde ~3k req, anonim IP'ler bazen
    engelleniyor); key varsa aylık 100k req ve stabil.
    """
    url = "https://min-api.cryptocompare.com/data/v2/news/"
    cat_map = {"BTC": "BTC", "ETH": "ETH", "BNB": "BNB", "SOL": "SOL",
               "XRP": "XRP", "ADA": "ADA", "DOT": "DOT", "AVAX": "AVAX",
               "LINK": "LINK", "LTC": "LTC"}
    cat = cat_map.get(coin, coin)

    rows: List[dict] = []
    lTs = None
    since_ts = int(pd.Timestamp(since).timestamp())

    headers = {"User-Agent": "Mozilla/5.0 (crypto-sentiment-bot/1.0)"}
    cc_key = _get_key("CRYPTOCOMPARE_KEY")
    if cc_key:
        # CryptoCompare resmi format: "Apikey {key}"
        headers["authorization"] = f"Apikey {cc_key}"
        if verbose:
            print(f"  [CC] {coin}: API key ile cagriliyor (authed tier)")
    elif verbose:
        print(f"  [CC] {coin}: key YOK, anonim tier (3k req/gun limitli)")

    try:
        for page in range(max_pages):
            params = {"lang": "EN", "categories": cat}
            if lTs is not None:
                params["lTs"] = lTs
            r = requests.get(url, params=params, timeout=15, headers=headers)
            if r.status_code != 200:
                if verbose:
                    print(f"  [CC] {coin} p{page}: HTTP {r.status_code} — {r.text[:120]}")
                break
            js = r.json()
            data = js.get("Data", []) or []
            if verbose:
                print(f"  [CC] {coin} p{page}: {len(data)} haber (Message={js.get('Message','?')[:40]})")
            if not data:
                break
            for it in data:
                ts = int(it.get("published_on", 0))
                if ts == 0:
                    continue
                d = pd.to_datetime(ts, unit="s").date().isoformat()
                rows.append({
                    "date":   d,
                    "title":  it.get("title", "") or "",
                    "body":   it.get("body", "") or "",
                    "source": it.get("source", "cryptocompare"),
                    "url":    it.get("url", ""),
                })
            min_ts = min(int(it.get("published_on", 0)) for it in data)
            if min_ts < since_ts:
                break
            lTs = min_ts - 1
            time.sleep(0.5)  # rate limit saygısı
    except Exception as e:
        warnings.warn(f"CryptoCompare hata: {e}")
        if verbose: print(f"  [CC] {coin} EXCEPTION: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if df.empty:
        if verbose: print(f"  [CC] {coin}: dönen row sayısı 0")
        return df
    before_filter = len(df)
    df = df[(df["date"] >= since) & (df["date"] <= until)]
    if verbose and len(df) < before_filter:
        print(f"  [CC] {coin}: tarih filtresi {before_filter} → {len(df)}")
    return df.drop_duplicates("url").reset_index(drop=True)


# ---------- RSS (ücretsiz, sınırsız) ----------

def fetch_news_rss(coin: str, since: str, until: str,
                   verbose: bool = False) -> pd.DataFrame:
    """RSS feedlerini tarayıp coin adı geçen haberleri döndürür.

    Önce feedparser varsa onu dener (daha sağlam Atom/ns handling);
    yoksa stdlib xml.etree ile parse eder.
    """
    from email.utils import parsedate_to_datetime

    fullname = COIN_FULLNAME.get(coin, coin.lower())
    needles = [fullname.lower(), coin.lower()]  # BTC, ETH gibi kısalar için case-insensitive sub

    # Gerçekçi User-Agent — Coindesk/Cointelegraph bot koruması için
    UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
    headers = {"User-Agent": UA, "Accept": "application/rss+xml, application/xml, text/xml, */*"}

    try:
        import feedparser  # opsiyonel, requirements'ta varsa daha iyi
        _has_fp = True
    except ImportError:
        _has_fp = False

    rows: List[dict] = []
    for feed_url in RSS_FEEDS:
        try:
            r = requests.get(feed_url, timeout=15, headers=headers)
            if r.status_code != 200:
                if verbose:
                    print(f"  [RSS] {coin} {feed_url.split('/')[2]}: HTTP {r.status_code}")
                continue

            items_raw = []

            if _has_fp:
                fp = feedparser.parse(r.content)
                for e in fp.entries:
                    items_raw.append({
                        "title": (e.get("title") or "").strip(),
                        "body":  ((e.get("summary") or e.get("description") or
                                   (e.get("content", [{}])[0] if e.get("content") else {}).get("value", "")) or ""),
                        "link":  e.get("link") or "",
                        "pub":   e.get("published") or e.get("updated") or "",
                    })
            else:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(r.content)
                items = root.findall(".//item") or root.findall(".//{http://www.w3.org/2005/Atom}entry")
                for it in items:
                    title_el = it.find("title") or it.find("{http://www.w3.org/2005/Atom}title")
                    # content:encoded bazı feedlerde asıl body
                    desc_el  = (it.find("{http://purl.org/rss/1.0/modules/content/}encoded")
                                or it.find("description")
                                or it.find("summary")
                                or it.find("{http://www.w3.org/2005/Atom}summary")
                                or it.find("{http://www.w3.org/2005/Atom}content"))
                    pub_el   = (it.find("pubDate") or it.find("published")
                                or it.find("{http://www.w3.org/2005/Atom}published")
                                or it.find("{http://www.w3.org/2005/Atom}updated"))
                    link_el  = it.find("link") or it.find("{http://www.w3.org/2005/Atom}link")

                    title = (title_el.text or "") if title_el is not None else ""
                    desc  = (desc_el.text  or "") if desc_el  is not None else ""
                    if link_el is not None:
                        link = link_el.text or link_el.get("href", "") or ""
                    else:
                        link = ""
                    pub_txt = (pub_el.text or "") if pub_el is not None else ""
                    items_raw.append({"title": title, "body": desc, "link": link, "pub": pub_txt})

            if verbose:
                print(f"  [RSS] {coin} {feed_url.split('/')[2]}: {len(items_raw)} item (pre-filter)")

            matched = 0
            for it in items_raw:
                blob = (it["title"] + " " + it["body"]).lower()
                if not any(n in blob for n in needles):
                    continue
                try:
                    dt = parsedate_to_datetime(it["pub"]) if it["pub"] else None
                    d = dt.date().isoformat() if dt else ""
                except Exception:
                    d = ""
                if not d:
                    continue
                rows.append({
                    "date":   d,
                    "title":  it["title"],
                    "body":   it["body"],
                    "source": feed_url.split("/")[2],
                    "url":    it["link"],
                })
                matched += 1
            if verbose and matched:
                print(f"  [RSS] {coin} {feed_url.split('/')[2]}: {matched} eşleşti")
        except Exception as e:
            if verbose:
                print(f"  [RSS] {coin} {feed_url.split('/')[2]}: EXCEPTION {e}")
            warnings.warn(f"RSS [{feed_url}] hata: {e}")
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[(df["date"] >= since) & (df["date"] <= until)]
    return df.drop_duplicates("url").reset_index(drop=True)


# ---------- TradingView Ideas (community trader analizleri) ----------

# TV ideas listing — bot için realistik headers. POC (test_tradingview_scrape.py)
# GHA Linux runner'ında 5/5 başarılı, Cloudflare blok yok.
_TV_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not A(Brand";v="24"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Upgrade-Insecure-Requests": "1",
}


def fetch_news_tradingview(coin: str, since: str, until: str,
                           fetch_bodies: bool = False,
                           verbose: bool = False) -> pd.DataFrame:
    """TradingView Ideas listing'i scrape et.

    URL: https://www.tradingview.com/symbols/{coin}USD/ideas/
    Parser: [class*="ideaCard-"] (POC ile doğrulandı, ~23 idea/coin döner)

    fetch_bodies=True olursa her idea'nın detail sayfasını da çeker (yavaşlatır
    ama sliding window FB+CB için zengin metin sağlar). False (default) snippet
    yeterli kabul eder.

    Hata durumlarında (CF blok, network, parser fail) boş DataFrame döner —
    pipeline'ı kırmaz.
    """
    if not HAS_BS4_LIVE:
        warnings.warn("[TV] beautifulsoup4 yok — TV scraper atlandı")
        return pd.DataFrame()

    url = f"https://www.tradingview.com/symbols/{coin}USD/ideas/"
    try:
        r = requests.get(url, headers=_TV_HEADERS, timeout=20, allow_redirects=True)
    except requests.RequestException as e:
        if verbose:
            print(f"  [TV] {coin}: request fail {e}")
        return pd.DataFrame()

    if r.status_code != 200:
        if verbose:
            print(f"  [TV] {coin}: HTTP {r.status_code}")
        return pd.DataFrame()

    body_lower = r.text.lower()
    cf_markers = ("checking your browser", "cf-browser-verification", "just a moment...")
    if any(m in body_lower for m in cf_markers):
        if verbose:
            print(f"  [TV] {coin}: Cloudflare challenge — skip")
        return pd.DataFrame()

    soup = BeautifulSoup(r.text, "html.parser")
    cards = soup.select('[class*="ideaCard-"]')
    if verbose:
        print(f"  [TV] {coin}: {len(cards)} idea card parse edildi")

    rows = []
    for c in cards:
        # Title: "title-{HASH}" class veya ilk anchor
        title_el = c.select_one('[class*="title-"]') or c.find("a")
        if title_el is None:
            continue
        title = title_el.get_text(strip=True)
        if len(title) < 10:
            continue

        # Body snippet: "paragraph-{HASH}" veya 'p' tag'i
        body_el = c.select_one('[class*="paragraph-"]') or c.find("p")
        body = body_el.get_text(strip=True) if body_el else ""

        # URL: ideaCard içinden href
        url_el = c.find("a", href=True)
        idea_url = url_el["href"] if url_el else ""
        if idea_url and not idea_url.startswith("http"):
            idea_url = "https://www.tradingview.com" + idea_url

        # Date: TV "time" element veya datetime attribute
        time_el = c.find("time") or c.select_one('[class*="time-"]')
        date_str = ""
        if time_el is not None:
            # datetime attribute genellikle ISO format
            dt_attr = time_el.get("datetime") or time_el.get("title") or time_el.get_text(strip=True)
            try:
                dt = pd.to_datetime(dt_attr, errors="coerce", utc=True)
                if pd.notna(dt):
                    date_str = dt.date().isoformat()
            except Exception:
                pass
        # Fallback: bugünün tarihi (TV trending'i genelde son 24-48 saat)
        if not date_str:
            date_str = pd.Timestamp.utcnow().date().isoformat()

        rows.append({
            "date": date_str,
            "title": title[:300],
            "body": body[:2000],
            "source": "tradingview_ideas",
            "url": idea_url,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df[(df["date"] >= since) & (df["date"] <= until)]
    df = df.drop_duplicates("url").reset_index(drop=True)

    # Detay body fetch — opsiyonel, FB+CB sliding window için tam metin
    if fetch_bodies and not df.empty:
        bodies_full = []
        for u in df["url"].tolist():
            try:
                rd = requests.get(u, headers=_TV_HEADERS, timeout=15)
                if rd.status_code == 200:
                    sd = BeautifulSoup(rd.text, "html.parser")
                    art = sd.find("article") or sd.select_one('[class*="ideaContent-"]')
                    full = art.get_text(" ", strip=True)[:5000] if art else ""
                    bodies_full.append(full)
                else:
                    bodies_full.append("")
                time.sleep(1.0)  # rate limit
            except Exception:
                bodies_full.append("")
        # Snippet + full body birleşimi
        df["body"] = [
            (full if len(full) > len(snip) else snip)
            for snip, full in zip(df["body"], bodies_full)
        ]

    return df


# ---------- Merkezi dispatcher ----------

def fetch_news(coin: str, since: str, until: str) -> pd.DataFrame:
    """Aktif kaynakları sırayla çağırır, sonuçları birleştirir.

    Sıra: CryptoCompare → RSS → NewsAPI → TradingView (USE_* bayrakları kontrol).
    Dönüş kolonları: date, title, body, source, url, section.
    section atama _section_for_source()'tan gelir; SOURCE_SECTION sözlüğüne bak.
    URL tekrarları birleştirme sırasında temizlenir.
    """
    frames = []

    if USE_CRYPTOCOMPARE:
        cc = fetch_news_cryptocompare(coin, since, until)
        if not cc.empty:
            frames.append(cc)

    if USE_RSS:
        rss = fetch_news_rss(coin, since, until)
        if not rss.empty:
            frames.append(rss)

    if USE_NEWSAPI:
        na = fetch_news_newsapi(coin, since, until)
        if not na.empty:
            frames.append(na)

    if USE_CRYPTOPANIC:
        cp = fetch_news_cryptopanic(coin, since, until)
        if not cp.empty:
            frames.append(cp)

    if USE_TRADINGVIEW:
        tv = fetch_news_tradingview(coin, since, until,
                                    fetch_bodies=TRADINGVIEW_FETCH_BODIES)
        if not tv.empty:
            frames.append(tv)

    if not frames:
        return pd.DataFrame(columns=["date", "title", "body", "source", "url", "section"])

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    df["_text"] = (df["title"].fillna("") + " " + df["body"].fillna("")).str.strip()
    df = df[df["_text"].str.len() > 10].drop(columns=["_text"]).reset_index(drop=True)

    # Section etiketi — historical V2 schema ile uyumlu (section 1/2/3)
    df["section"] = df["source"].apply(_section_for_source).astype("int8")
    return df


# ============================================================
# 2. NLP ENSEMBLE
# ============================================================

class NLPEnsemble:
    """FinBERT + CryptoBERT + J-Hartmann → combined_score.

    Model yüklemesi lazy: ilk score_batch çağrısında yapılır.
    Her model için score = P(positive) - P(negative) ∈ [-1, +1].
    combined_score = jh-only (V4 NLP combo ablation winner — S1 portfolio
    Sharpe +1.38, return +183%, alpha +193 pp). fb/cb skorları diagnostics
    için yine kayda geçer ama combined'da kullanılmaz.

    J-Hartmann 7 duygu döner; biz bunları pos/neg/neu kovalarına haritalarız:
        joy, surprise          → pos
        anger, sadness, fear,
        disgust                → neg
        neutral                → neu
    """

    # J-Hartmann etiketleri → pos/neg/neu grupları
    _JH_POS = {"joy", "surprise"}
    _JH_NEG = {"anger", "sadness", "fear", "disgust"}
    _JH_NEU = {"neutral"}

    def __init__(self, device: Optional[str] = None, batch_size: Optional[int] = None):
        # Öncelik: parametre > env var > dosya üstündeki NLP_DEVICE/NLP_BATCH_SIZE
        self.device = device or os.environ.get("NLP_DEVICE") or NLP_DEVICE
        self.batch_size = int(batch_size
                              or os.environ.get("NLP_BATCH_SIZE")
                              or NLP_BATCH_SIZE)
        self.fb = None   # ProsusAI/finbert
        self.cb = None   # ElKulako/cryptobert
        self.jh = None   # j-hartmann/emotion-english-distilroberta-base

    # ---------- lazy model loader ----------
    def _load(self):
        if self.fb is not None:
            return
        try:
            import torch
            from transformers import pipeline
        except ImportError as e:
            raise RuntimeError(
                "NLPEnsemble için transformers + torch gerekli.\n"
                "Yükleme: pip install transformers torch sentencepiece"
            ) from e

        # Device seçimi (0 = ilk GPU, -1 = CPU)
        if self.device == "cuda" or (self.device is None and torch.cuda.is_available()):
            dev = 0
        else:
            dev = -1

        print(f"[NLP] Modeller yükleniyor (device={'cuda:0' if dev == 0 else 'cpu'})...")
        t0 = time.time()
        # framework="pt" → transformers TensorFlow/Keras yolu denemesin.
        # Sistemde TF yüklüyse ve framework belirtilmezse TF branch seçebiliyor.
        self.fb = pipeline("sentiment-analysis",
                           model="ProsusAI/finbert",
                           framework="pt", device=dev,
                           truncation=True, max_length=256)
        self.cb = pipeline("sentiment-analysis",
                           model="ElKulako/cryptobert",
                           framework="pt", device=dev,
                           truncation=True, max_length=256)
        self.jh = pipeline("text-classification",
                           model="j-hartmann/emotion-english-distilroberta-base",
                           top_k=None, framework="pt", device=dev,
                           truncation=True, max_length=256)
        print(f"[NLP] Hazır ({time.time()-t0:.1f}s)")

    # ---------- label harita ----------
    @staticmethod
    def _probs_finbert(out) -> Dict[str, float]:
        """FinBERT tekli çıktı {label,score} → pos/neg/neu."""
        # FinBERT tek label döner (en yüksek); pos/neg/neu dışı olmadığından:
        lbl = out["label"].lower()
        s = float(out["score"])
        if lbl == "positive": return {"positive": s, "negative": 1 - s, "neutral": 0.0}
        if lbl == "negative": return {"positive": 0.0, "negative": s, "neutral": 1 - s}
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

    @staticmethod
    def _probs_cryptobert(out) -> Dict[str, float]:
        """CryptoBERT — Bullish / Bearish / Neutral."""
        lbl = out["label"].lower()
        s = float(out["score"])
        if lbl.startswith("bullish"): return {"positive": s, "negative": (1-s)/2, "neutral": (1-s)/2}
        if lbl.startswith("bearish"): return {"positive": (1-s)/2, "negative": s, "neutral": (1-s)/2}
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

    @classmethod
    def _probs_jhartmann(cls, out_list) -> Dict[str, float]:
        """J-Hartmann 7 duygu listesi → pos/neg/neu agregesi."""
        pos = neg = neu = 0.0
        for o in out_list:
            lbl = o["label"].lower()
            sc = float(o["score"])
            if lbl in cls._JH_POS: pos += sc
            elif lbl in cls._JH_NEG: neg += sc
            elif lbl in cls._JH_NEU: neu += sc
        # Normalize güvenlik
        tot = pos + neg + neu
        if tot > 0:
            pos, neg, neu = pos/tot, neg/tot, neu/tot
        return {"positive": pos, "negative": neg, "neutral": neu}

    @staticmethod
    def _to_score(probs: Dict[str, float]) -> float:
        return float(probs.get("positive", 0.0) - probs.get("negative", 0.0))

    # ---------- public: batch skorlama ----------
    def score_batch(self, texts: List[str]) -> pd.DataFrame:
        """Her metin için (score_fb, score_cb, score_jh, combined_score) döner.

        Boş metinler atılmaz — NaN ile doldurulur (caller reindex için kullanır).
        """
        if not texts:
            return pd.DataFrame(columns=["score_fb", "score_cb", "score_jh", "combined_score"])

        self._load()
        bs = self.batch_size

        # Truncate çok uzun metinleri (tokenizer + max_length zaten kesiyor ama
        # Python str slice daha hızlı ve bellek dostu)
        texts_clean = [(t or "")[:1000] if (t or "").strip() else "neutral." for t in texts]

        fb_out = self.fb(texts_clean, batch_size=bs)
        cb_out = self.cb(texts_clean, batch_size=bs)
        jh_out = self.jh(texts_clean, batch_size=bs)

        fb_scores, cb_scores, jh_scores = [], [], []
        for fb, cb, jh in zip(fb_out, cb_out, jh_out):
            fb_scores.append(self._to_score(self._probs_finbert(fb)))
            cb_scores.append(self._to_score(self._probs_cryptobert(cb)))
            jh_scores.append(self._to_score(self._probs_jhartmann(jh)))

        fb_arr = np.asarray(fb_scores)
        cb_arr = np.asarray(cb_scores)
        jh_arr = np.asarray(jh_scores)
        # Winner of V4 NLP combo ablation (S1) = jhartmann-only.
        # combined_score deploys jh-only so live sentiment matches the
        # production v4 model's training distribution. fb/cb scores are
        # still logged for diagnostics + future ablations.
        combined = jh_arr

        return pd.DataFrame({
            "score_fb":       fb_arr,
            "score_cb":       cb_arr,
            "score_jh":       jh_arr,
            "combined_score": combined,
        })


# ============================================================
# 3. GÜNLÜK AGGREGATE
# ============================================================

# Bir haberin pos/neu/neg "yönü" için eşik (training ile tutarlı)
SCORE_POS_THRESH = +0.05
SCORE_NEG_THRESH = -0.05


def _direction(score: float) -> str:
    if score >  SCORE_POS_THRESH: return "pos"
    if score < SCORE_NEG_THRESH: return "neg"
    return "neu"


# ─────────────────────────────────────────────────────────────────────────
# Section weights (V2 t3_section_weights.csv'den okunur; coin-spesifik S5)
# ─────────────────────────────────────────────────────────────────────────

SECTION_WEIGHTS_CSV = "section_weights.csv"   # data_live/ altında aranır


def _load_section_weights_for_coin(coin: str, scenario: str = "S5") -> dict[int, float]:
    """t3 csv'den (coin, scenario) ağırlıklarını çek. Bulunamazsa 1/3 fallback.

    Production için coin-spesifik S5 (en yeni rejim) önerilir — production
    modelinin eğitim cut-off (2025-12-31) ile S5 train/val penceresi uyumlu.
    """
    try:
        from paths import HISTORICAL_DATA_ROOT
    except ImportError:
        HISTORICAL_DATA_ROOT = Path(".")

    # Önce live (vendored) konumuna bak, sonra historical
    candidates = [
        SENT_DIR.parent / SECTION_WEIGHTS_CSV,                                   # data_live/section_weights.csv
        HISTORICAL_DATA_ROOT / "models" / "v2_sentiment_strategy" / SECTION_WEIGHTS_CSV,
    ]
    csv_path = next((p for p in candidates if p.is_file()), None)
    if csv_path is None:
        warnings.warn(f"[{coin}] section_weights.csv bulunamadı, 1/3 fallback")
        return {1: 1/3, 2: 1/3, 3: 1/3}

    df = pd.read_csv(csv_path)
    row = df[(df["coin"] == coin) & (df["scenario"] == scenario)]
    if row.empty:
        warnings.warn(f"[{coin}/{scenario}] section_weights.csv'de yok, 1/3 fallback")
        return {1: 1/3, 2: 1/3, 3: 1/3}

    r = row.iloc[0]
    return {1: float(r["w_s1"]), 2: float(r["w_s2"]), 3: float(r["w_s3"])}


def aggregate_daily(scored: pd.DataFrame, coin: str,
                    fill_dates: Optional[pd.DatetimeIndex] = None,
                    weights_scenario: str = "S5") -> pd.DataFrame:
    """(date × haber) → (date) seviyesinde sentiment features (section-aware).

    Girdi kolonları:
        date, combined_score, _dir (opsiyonel — yoksa SCORE_*_THRESH'den hesaplar),
        section (1/2/3 — `_section_for_source` ile etiketlenmiş olmalı)

    Çıktı şeması (V2 historical training ile birebir uyumlu, 22 kolon):
        date,
        section_{1,2,3}_avg_score / pos_ratio / neu_ratio / neg_ratio / news_count   (5×3 = 15)
        combined_score, combined_pos_ratio, combined_neu_ratio, combined_neg_ratio   (4)
        total_news_count, has_news, days_since_news                                  (3)

    section_* kolonları artık gerçek per-section agregat (NaN değil) — ancak
    o gün ilgili section'da haber yoksa o section'ın kolonları NaN olur,
    combined_score IC-weighted olarak mevcut section'lar üzerinden hesaplanır
    (V2 t4_features.py'deki aynı iki-seviyeli fallback).
    """
    # Boş kaynak → sadece date index'i ile boş çerçeve
    if scored.empty or "combined_score" not in scored.columns:
        scored = pd.DataFrame(columns=["date", "combined_score", "section"])

    df = scored.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["_dir"] = df["combined_score"].apply(_direction)
    # Section yoksa (eski caller veya hatalı upstream) section_2 default
    if "section" not in df.columns:
        df["section"] = 2
    df["section"] = df["section"].fillna(2).astype(int)

    # ── 1) Per-section günlük agregat ──────────────────────────────────
    if df.empty:
        per_sec = pd.DataFrame(columns=[
            "date", "section", "avg_score",
            "pos_ratio", "neu_ratio", "neg_ratio", "news_count"])
    else:
        g_sec = df.groupby(["date", "section"])
        per_sec = pd.DataFrame({
            "avg_score":  g_sec["combined_score"].mean(),
            "pos_ratio":  g_sec["_dir"].apply(lambda s: (s == "pos").mean()),
            "neu_ratio":  g_sec["_dir"].apply(lambda s: (s == "neu").mean()),
            "neg_ratio":  g_sec["_dir"].apply(lambda s: (s == "neg").mean()),
            "news_count": g_sec.size(),
        }).reset_index()

    # ── 2) Wide pivot: section_{1,2,3}_{metric} ─────────────────────────
    if per_sec.empty:
        # Boş çerçeve — date axis'ı fill_dates'ten kuracağız
        wide = pd.DataFrame()
    else:
        pivots = []
        for metric in ("avg_score", "pos_ratio", "neu_ratio", "neg_ratio", "news_count"):
            piv = per_sec.pivot(index="date", columns="section", values=metric)
            piv.columns = [f"section_{int(s)}_{metric}" for s in piv.columns]
            pivots.append(piv)
        wide = pd.concat(pivots, axis=1)

    # 3 section kolonunun da var olmasını garantile (eksik section → NaN/0)
    for sec in (1, 2, 3):
        for metric in ("avg_score", "pos_ratio", "neu_ratio", "neg_ratio"):
            col = f"section_{sec}_{metric}"
            if col not in wide.columns:
                wide[col] = np.nan
        nc_col = f"section_{sec}_news_count"
        if nc_col not in wide.columns:
            wide[nc_col] = 0

    # ── 3) Date axis'ı fill_dates'e reindex ─────────────────────────────
    if fill_dates is None and not wide.empty:
        fill_dates = pd.date_range(wide.index.min(), wide.index.max(), freq="D")
    if fill_dates is not None:
        wide = wide.reindex(fill_dates)
    wide.index.name = "date"

    # news_count NaN → 0 (haber yok)
    for sec in (1, 2, 3):
        wide[f"section_{sec}_news_count"] = wide[f"section_{sec}_news_count"].fillna(0).astype(int)

    # ── 4) combined_score = IC-weighted section avg_score ──────────────
    weights = _load_section_weights_for_coin(coin, scenario=weights_scenario)
    score_arr = np.vstack([wide[f"section_{s}_avg_score"].values for s in (1, 2, 3)])  # (3, N)
    w_arr = np.array([weights[s] for s in (1, 2, 3)]).reshape(-1, 1)                   # (3, 1)
    mask = ~np.isnan(score_arr)

    # Level 1: IC-weighted (renormalize over available sections)
    w_matrix = np.broadcast_to(w_arr, score_arr.shape).copy()
    w_matrix[~mask] = 0.0
    w_sum = w_matrix.sum(axis=0)
    score_zeroed = np.where(mask, score_arr, 0.0)
    weighted_sum = (w_matrix * score_zeroed).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        combined_ic = np.where(w_sum > 0, weighted_sum / w_sum, np.nan)

    # Level 2 (fallback): equal-weighted over available sections
    n_avail = mask.sum(axis=0)
    equal_sum = score_zeroed.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        combined_eq = np.where(n_avail > 0, equal_sum / n_avail, np.nan)

    wide["combined_score"] = np.where(np.isfinite(combined_ic), combined_ic, combined_eq)

    # ── 5) combined_*_ratio = news-count-weighted (V2 t4 ile aynı) ──────
    news_arr = np.vstack([wide[f"section_{s}_news_count"].values for s in (1, 2, 3)]).astype(float)
    total_news = news_arr.sum(axis=0)
    for ratio in ("pos_ratio", "neu_ratio", "neg_ratio"):
        r_arr = np.vstack([wide[f"section_{s}_{ratio}"].values for s in (1, 2, 3)])
        r_arr = np.nan_to_num(r_arr, nan=0.0)
        num = (r_arr * news_arr).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            wide[f"combined_{ratio}"] = np.where(total_news > 0, num / total_news, 0.0)

    # ── 6) total_news_count, has_news, days_since_news ─────────────────
    wide["total_news_count"] = total_news.astype(int)
    wide["has_news"] = (total_news > 0).astype(int)

    last_idx = None
    days_since = []
    for idx, hn in enumerate(wide["has_news"].values):
        if hn == 1:
            last_idx = idx
            days_since.append(0)
        else:
            days_since.append(idx - last_idx if last_idx is not None else np.nan)
    wide["days_since_news"] = days_since

    # ── 7) Kolon sırasını historical V2 ile hizala ─────────────────────
    daily = wide.reset_index()
    cols = [
        "date",
        "section_1_avg_score","section_1_pos_ratio","section_1_neu_ratio","section_1_neg_ratio","section_1_news_count",
        "section_2_avg_score","section_2_pos_ratio","section_2_neu_ratio","section_2_neg_ratio","section_2_news_count",
        "section_3_avg_score","section_3_pos_ratio","section_3_neu_ratio","section_3_neg_ratio","section_3_news_count",
        "combined_score","combined_pos_ratio","combined_neu_ratio","combined_neg_ratio",
        "total_news_count","has_news","days_since_news",
    ]
    return daily[cols]


# ============================================================
# 4. FEATURE ENGINEERING (lag + rolling)
# ============================================================

def add_lag_rolling(daily: pd.DataFrame) -> pd.DataFrame:
    """lag 1/2/3 + rolling mean/std 3/7 + news_count rolling 3/7 ekler."""
    d = daily.sort_values("date").copy()
    for k in [1, 2, 3]:
        d[f"combined_lag_{k}"] = d["combined_score"].shift(k)
    d["combined_roll3_mean"] = d["combined_score"].rolling(3, min_periods=1).mean()
    d["combined_roll7_mean"] = d["combined_score"].rolling(7, min_periods=1).mean()
    d["combined_roll7_std"]  = d["combined_score"].rolling(7, min_periods=2).std()
    d["news_roll7_sum"] = d["total_news_count"].rolling(7, min_periods=1).sum()
    d["news_roll3_sum"] = d["total_news_count"].rolling(3, min_periods=1).sum()
    return d


# ============================================================
# 5. ORCHESTRATION — tek coin güncelleme
# ============================================================

def _historical_path(coin: str) -> Path:
    return HISTORICAL_DATA_ROOT / "models" / "v2_sentiment_strategy" / "features" / f"features_{coin}_S1.parquet"


def update_coin(coin: str, lookback_days: int = 30,
                ensemble: Optional[NLPEnsemble] = None) -> Path:
    """Son lookback_days'ı çek → skorla → aggregate → mevcut parquet ile concat → kaydet.

    Mevcut pipeline son tarih kadar veriyi ffill mantığı ile uzatmaz;
    yalnızca son_tarih+1 itibariyle gelen yeni haberleri hesaba katar.
    """
    out = SENT_DIR / f"{coin}.parquet"

    # 1) mevcut veri
    if out.exists():
        existing = pd.read_parquet(out)
    else:
        existing = pd.read_parquet(_historical_path(coin))
    existing["date"] = pd.to_datetime(existing["date"])
    last_known = existing["date"].max()
    since = (last_known - pd.Timedelta(days=5)).date().isoformat()   # 5 gün güvenlik
    until = pd.Timestamp.utcnow().date().isoformat()

    if pd.Timestamp(until) <= last_known:
        print(f"[{coin}] zaten güncel (last={last_known.date()})")
        return out

    # 2) haber çek
    news = fetch_news(coin, since, until)
    if news.empty:
        warnings.warn(f"[{coin}] {since}..{until} aralığında haber bulunamadı "
                       "(API anahtarı yok veya sonuç boş). Mevcut veri dokunmadan yazıldı.")
        existing.to_parquet(out, index=False)
        return out

    # 3) skorla
    if ensemble is None:
        ensemble = NLPEnsemble()
    texts = (news["title"].fillna("") + ". " + news["body"].fillna("")).tolist()
    scores = ensemble.score_batch(texts)
    # Section etiketini scored'a taşı — aggregate_daily section-aware
    scored = pd.concat([
        news[["date", "section"]].reset_index(drop=True),
        scores,
    ], axis=1)

    # 4) günlük aggregate (yalnız yeni kısım)
    fill_range = pd.date_range(since, until, freq="D")
    new_daily = aggregate_daily(scored, coin, fill_dates=fill_range)

    # 5) lag/rolling için: existing'in son 10 gününü (seed) new_daily'nin başına koy
    #    — rolling 7 pencereli, 10 gün bol
    seed_cols = [c for c in new_daily.columns if c in existing.columns]
    seed = existing.loc[existing["date"] >= (last_known - pd.Timedelta(days=10)),
                        seed_cols].copy()
    merged = (pd.concat([seed, new_daily], ignore_index=True)
                .drop_duplicates(subset=["date"], keep="last")
                .sort_values("date")
                .reset_index(drop=True))

    merged = add_lag_rolling(merged)

    # 6) existing'in ÖNCE kısmı (last_known-10 günden eski) + merged (son kısım)
    keep_old = existing[existing["date"] < (last_known - pd.Timedelta(days=10))].copy()
    final = pd.concat([keep_old, merged], ignore_index=True)
    final = final.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)

    # 7) historical'de var olan ekstra kolonları (coin, scenario, w_s1…) koru
    for c in ("coin", "scenario", "w_s1", "w_s2", "w_s3", "close"):
        if c in existing.columns and c not in final.columns:
            # last-value ffill mantığı
            src = existing.set_index("date")[c]
            final[c] = final["date"].map(src).ffill()

    final.to_parquet(out, index=False)
    print(f"[{coin}] yazıldı: {out.name} shape={final.shape} "
          f"son={pd.to_datetime(final['date']).max().date()} "
          f"yeni_haber={len(news)}")
    return out


def update_all(lookback_days: int = 30) -> None:
    """Tüm coinler — tek NLPEnsemble örneği (model tekrar yüklenmesin)."""
    ensemble = NLPEnsemble()
    for c in COINS:
        try:
            update_coin(c, lookback_days, ensemble=ensemble)
        except Exception as e:
            print(f"[{c}] HATA: {e}")


def backfill_from_historical() -> None:
    """İlk kurulum için — V2 historical parquet'lerini data_live'e kopyala."""
    for c in COINS:
        out = SENT_DIR / f"{c}.parquet"
        if out.exists():
            print(f"[{c}] zaten var, atla")
            continue
        hist = _historical_path(c)
        df = pd.read_parquet(hist)
        df.to_parquet(out, index=False)
        print(f"[{c}] backfill → {out.name} ({df.shape})")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else "update"
    if arg == "backfill":
        backfill_from_historical()
    elif arg == "update":
        update_all()
    elif arg == "test":
        # Küçük NLP denemesi — API anahtarı gerektirmez, sadece model yükler
        ens = NLPEnsemble()
        out = ens.score_batch([
            "Bitcoin smashes all-time high as institutional inflows surge.",
            "Ethereum network faces critical vulnerability, prices tumble.",
            "SEC delays decision on crypto ETF applications.",
        ])
        print(out.round(3))
    else:
        print(f"Bilinmeyen komut: {arg}. Geçerli: backfill | update | test")