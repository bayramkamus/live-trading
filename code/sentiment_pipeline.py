"""Canlı sentiment pipeline — haber → NLP ensemble → günlük feature parquet.

Akış:
  1) fetch_news(coin, since, until)  → CryptoPanic / NewsAPI / RSS
  2) NLPEnsemble.score_batch(texts)  → FinBERT + CryptoBERT + J-Hartmann → combined_score
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


# ---------- Merkezi dispatcher ----------

def fetch_news(coin: str, since: str, until: str) -> pd.DataFrame:
    """Aktif kaynakları sırayla çağırır, sonuçları birleştirir.

    Sıra: CryptoCompare → RSS → NewsAPI → CryptoPanic (USE_* bayrakları kontrol).
    Dönüş kolonları: date (YYYY-MM-DD str), title, body, source, url.
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

    if not frames:
        return pd.DataFrame(columns=["date", "title", "body", "source", "url"])

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    df["_text"] = (df["title"].fillna("") + " " + df["body"].fillna("")).str.strip()
    df = df[df["_text"].str.len() > 10].drop(columns=["_text"]).reset_index(drop=True)
    return df


# ============================================================
# 2. NLP ENSEMBLE
# ============================================================

class NLPEnsemble:
    """FinBERT + CryptoBERT + J-Hartmann → combined_score.

    Model yüklemesi lazy: ilk score_batch çağrısında yapılır.
    Her model için score = P(positive) - P(negative) ∈ [-1, +1].
    combined_score = (score_fb + score_cb + score_jh) / 3.

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
        combined = (fb_arr + cb_arr + jh_arr) / 3.0

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


def aggregate_daily(scored: pd.DataFrame, coin: str,
                    fill_dates: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
    """(date × haber) → (date) seviyesinde sentiment features.

    Girdi kolonları beklenir: date, combined_score (ve isteğe bağlı score_fb/cb/jh).
    Çıktı kolonları (V2 historical şemasıyla uyumlu):
        date, combined_score (mean), combined_pos_ratio, combined_neu_ratio, combined_neg_ratio,
        total_news_count, has_news, days_since_news,
        section_{1,2,3}_avg_score / pos_ratio / neu_ratio / neg_ratio / news_count  (NaN — legacy)
    """
    # Boş kaynak → sadece date index'i ile boş çerçeve
    if scored.empty or "combined_score" not in scored.columns:
        scored = pd.DataFrame(columns=["date", "combined_score"])

    df = scored.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["_dir"] = df["combined_score"].apply(_direction)

    # Groupby agregasyon
    g = df.groupby("date")
    daily = pd.DataFrame({
        "combined_score":      g["combined_score"].mean(),
        "total_news_count":    g.size(),
        "combined_pos_ratio":  g["_dir"].apply(lambda s: (s == "pos").mean()),
        "combined_neu_ratio":  g["_dir"].apply(lambda s: (s == "neu").mean()),
        "combined_neg_ratio":  g["_dir"].apply(lambda s: (s == "neg").mean()),
    }).reset_index()

    # Tam tarih aralığına reindex (haber olmayan günler için has_news=0)
    if fill_dates is None and not daily.empty:
        fill_dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    if fill_dates is not None:
        daily = daily.set_index("date").reindex(fill_dates).rename_axis("date").reset_index()

    daily["has_news"] = (daily["total_news_count"].fillna(0) > 0).astype(int)
    daily["total_news_count"] = daily["total_news_count"].fillna(0).astype(int)
    for c in ["combined_pos_ratio", "combined_neu_ratio", "combined_neg_ratio"]:
        daily[c] = daily[c].fillna(0.0)
    # Haber yoksa combined_score NaN kalır (ffill'i add_lag_rolling öncesi yapmıyoruz;
    # böylece lag/rolling gerçek dağılımı yansıtır)

    # days_since_news — son haberli günden bu yana geçen gün sayısı
    last_idx = None
    days_since = []
    for idx, row in daily.iterrows():
        if row["has_news"] == 1:
            last_idx = idx
            days_since.append(0)
        else:
            days_since.append(idx - last_idx if last_idx is not None else np.nan)
    daily["days_since_news"] = days_since

    # V2 section_1/2/3 legacy kolonları — yeni pipeline'da üretilmiyor (NaN)
    for sec in (1, 2, 3):
        for suf in ("avg_score", "pos_ratio", "neu_ratio", "neg_ratio", "news_count"):
            daily[f"section_{sec}_{suf}"] = np.nan

    # Kolon sırasını historical ile hizala
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
    scored = pd.concat([news[["date"]].reset_index(drop=True), scores], axis=1)

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