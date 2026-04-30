# Live Trading — V4 Production

Kripto sentiment sinyal üretim sistemi. V4 leakage-fix pipeline'ının canlı taraftaki dağıtımı.

**Mimari:** Yerelde eğit → GitHub Actions her gece sinyal üretir → HuggingFace Space read-only dashboard sunar.

## Klasör yapısı

```
live_trading/
├── artifacts/v4_production/         # Eğitilmiş modeller (10 coin × 5 dosya) — yerelde üretilir, git'e commit
│   ├── production_metadata.json
│   ├── production_training_log.csv
│   └── {BTC,ETH,BNB,SOL,XRP,ADA,DOT,AVAX,LINK,LTC}/
│       ├── model.lgb
│       ├── feature_columns.json
│       ├── horizons.json
│       ├── label_quantiles.json
│       └── thresholds.json
│
├── code/
│   ├── paths.py                     # Path sabitleri
│   ├── feature_builders.py          # data_live/ okuyucuları
│   ├── inference.py                 # Sinyal üretici (public API)
│   ├── smoke_test.py                # Kurulum doğrulaması
│   ├── ohlcv_fetcher.py             # Binance günlük mum + 41 teknik kolon ✅
│   ├── macro_fetcher.py             # FRED VIX/DXY/IGREA + 27 transform ✅
│   ├── sentiment_pipeline.py        # NewsAPI + FinBERT/CryptoBERT/J-Hartmann ensemble ✅
│   ├── paper_broker.py              # Paper trading broker ✅
│   ├── db.py                        # SQLite katmanı (cloud için) ✅
│   └── orchestrate.py               # Günlük ana akış ✅
│
├── scripts/
│   └── daily_run.py                 # GitHub Actions cloud orkestratörü
│
├── app/                             # HuggingFace Space (Streamlit dashboard)
│   ├── streamlit_app.py             # Read-only UI
│   ├── requirements.txt             # Lightweight (no NLP)
│   └── README.md                    # HF Space metadata
│
├── broker/                          # paper broker state (git'e commit)
│   ├── state.json
│   ├── trades.csv
│   └── equity.csv
│
├── data_live/                       # (gitignore'da) — parquet cache'leri
│   ├── ohlcv/{COIN}.parquet
│   ├── sentiment/{COIN}.parquet
│   ├── tech/{COIN}.parquet
│   ├── macro/macro.parquet
│   └── app.db                       # SQLite (git'e commit)
│
├── .github/workflows/
│   ├── daily.yml                    # Her gece 02:10 UTC cron
│   └── sync-hf-space.yml            # app.db değiştiğinde HF Space'e push
│
├── logs/                            # signals_YYYY-MM-DD.csv
├── requirements.txt                 # Full (local + CI)
├── .gitignore
└── README.md
```

## Cloud dağıtım (önerilen)

Sistem **3 katmanlı** çalışır:

1. **Yerelde (ayda bir)**: modeli retrain et → `artifacts/v4_production/` altına yeni modelleri koy → commit & push.
2. **GitHub Actions (her gece 02:10 UTC)**: Binance/FRED/NewsAPI'den son veriyi çeker, 10 coin için sinyal üretir, PaperBroker'a uygular, `app.db`'yi güncelleyip commit eder.
3. **HuggingFace Space (her zaman)**: `app.db`'yi okur, Streamlit dashboard sunar.

### Gerekli secret'lar (GitHub → Settings → Secrets and variables → Actions)

| Secret | Açıklama |
|---|---|
| `FRED_API_KEY` | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) — ücretsiz |
| `NEWSAPI_KEY` | [newsapi.org](https://newsapi.org) — Developer plan ücretsiz |
| `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — Write yetkisi |
| `HF_USER` | HuggingFace kullanıcı adı (ör: `bayramk`) |
| `HF_SPACE` | Space adı (ör: `v4-crypto-signals`) |

Opsiyonel (tuning için — tüm sabitler `live_trading/code/config.py` içinde, env yoksa default'lar geçerli):

**Paper broker**

| Variable | Default | Açıklama |
|---|---|---|
| `PAPER_BASE_EQUITY` | `10000` | Başlangıç sermayesi ($) |
| `PAPER_RISK_PCT` | `0.10` | İşlem başına equity yüzdesi (10%) |
| `MAX_POSITIONS` | `5` | Aynı anda max açık pozisyon |
| `FEE_BPS` | `10` | Komisyon (bps, 0.10%) |
| `ALLOW_SHORTS` | `true` | SELL → SHORT_OPEN, BUY → cover |

**Sinyal kapıları (A2)**

| Variable | Default | Açıklama |
|---|---|---|
| `MIN_CONFIDENCE` | `0.0` | A2 üstüne ek p_buy/p_sell filtre (0=kapalı) |
| `DIR_MARGIN_DEFAULT` | `0.03` | Strong tier: \|p_dir − p_other_dir\| min |
| `HOLD_VETO_DEFAULT` | `0.05` | Strong tier: max(p_hold − p_dir) |
| `DIR_MARGIN_WEAK` | `0.06` | Weak tier sıkı margin |
| `HOLD_VETO_WEAK` | `0.0` | Weak tier sıkı HOLD veto |
| `WEAK_COINS` | `ADA,AVAX,DOT,ETH,LINK,LTC` | f1_val<0.25 veya best_iter≤1 |

**Haber kapsam (A2)**

| Variable | Default | Açıklama |
|---|---|---|
| `NEWS_MIN_LAST7D` | `3` | 7-gün toplam haber sayısı min |
| `NEWS_STALE_DAYS_MAX` | `2` | Son haberden bu yana max gün |

**Sinyal/fill timing (A1)**

| Variable | Default | Açıklama |
|---|---|---|
| `FILL_OFFSET` | `1` | T+1 fill (cron T'de: signal=T-2, fill=T-1) |

### Kurulum (ilk defa)

```bash
# 1) Repo'yu forkla / clone et, yerel kuruluma geç
git clone https://github.com/<kullanıcı>/<repo>.git
cd <repo>/live_trading
pip install -r requirements.txt

# 2) Modelleri yerelde üret (veya mevcut artifacts'ı commit et)
# ... eğitim akışı ...
python3 code/smoke_test.py                 # 10 coin sinyal üretmeli

# 3) Artifact'ları commit et
git add live_trading/artifacts/v4_production
git commit -m "add v4 production models"

# 4) GitHub'da secret'ları ekle (yukarıdaki tablo)

# 5) HuggingFace Space oluştur:
#    - huggingface.co/new-space
#    - SDK: Streamlit, Hardware: CPU Basic
#    - İsim: v4-crypto-signals (HF_SPACE ile aynı olmalı)
#    - Boş bırak, workflow ilk push'ta doldurur

# 6) İlk çalıştırmayı manuel tetikle:
#    GitHub → Actions → "daily-signals" → Run workflow
#    Bittiğinde app.db commit'lenir, sync-hf-space otomatik tetiklenir
```

### Günlük akış (tam otomatik)

```
02:10 UTC  → GitHub Actions "daily-signals" başlar
           → ohlcv_fetcher + macro_fetcher + sentiment_pipeline
           → orchestrate.run (10 coin × signal + paper trade)
           → db.py ile app.db güncellenir
           → commit: live_trading/data_live/app.db + broker/*.csv
           → push origin main
02:15 UTC  → "sync-to-hf-space" tetiklenir (app.db değişti)
           → HF Space repo'suna clone + copy + push
02:16 UTC  → Space yeniden build olur, dashboard güncel
```

## Yerel kurulum ve test

```bash
cd live_trading
pip install -r requirements.txt

# Klasörleri oluştur + backfill
cd code
python3 paths.py
python3 ohlcv_fetcher.py backfill
python3 macro_fetcher.py backfill
python3 sentiment_pipeline.py backfill

# Smoke test
python3 smoke_test.py
```

Beklenen çıktı:
```
=== V4 canlı inference smoke ===
coin as_of_date signal ...
Sinyal özet: {'HOLD': 7, 'SELL': 2, 'BUY': 1}
```

## Günlük çalıştırma

### Yerel (manuel test)

```bash
cd live_trading/code
python3 orchestrate.py                      # bugün (UTC)
python3 orchestrate.py --date 2026-04-20    # belirli tarih
python3 orchestrate.py --skip-update        # veri fetcher'ları atla
python3 orchestrate.py --no-execute         # paper trade'i atla
```

### Yerel (cloud orkestratörü — GH Actions'taki ile aynı)

```bash
cd live_trading
python3 scripts/daily_run.py                # tam akış + DB sync
python3 scripts/daily_run.py --dry-run      # DB/commit yazmadan dene
```

### Streamlit dashboard (yerel önizleme)

```bash
cd live_trading
streamlit run app/streamlit_app.py
# http://localhost:8501
```

## Public API

```python
import sys; sys.path.insert(0, "live_trading/code")
from inference import predict_signal_for_date

out = predict_signal_for_date("BTC", "2026-04-20")
# {'coin': 'BTC', 'signal': 'HOLD', 'p_sell': 0.30, 'p_hold': 0.40, 'p_buy': 0.30,
#  'buy_th': 0.35, 'sell_th': 0.30, 'horizon': 5, 'n_features': 71,
#  'as_of_date': '2026-04-20', 'has_sent': True, 'has_macro': True, 'has_tech': True}
```

DB'den sinyal oku:

```python
from db import DB
with DB() as db:
    signals = db.read_signals(days=30)      # son 30 gün
    trades  = db.read_trades()
    equity  = db.read_equity()
```

## Veri saklama politikası

| Veri | Retention | Neden |
|---|---|---|
| signals | 30 gün | Analiz/debug için yeterli; DB'yi küçük tutuyor |
| ohlcv/tech/sent/macro cache (DB'de) | 60 gün | Geriye dönük feature build + debug |
| parquet cache (data_live/) | ∞ (gitignore) | Yerelde kalıcı; git'e gitmiyor |
| trades | ∞ | Tam ticaret geçmişi |
| equity | ∞ | Tam eğri (çizim için) |

Her `daily_run.py` sonunda `db.prune_old(days=30, cache_days=60)` çağrılır.

## Modül durumu

| Modül | Durum |
|---|---|
| `inference.py` | ✅ |
| `feature_builders.py` | ✅ |
| `ohlcv_fetcher.py` | ✅ Binance REST + 41 teknik kolon |
| `macro_fetcher.py` | ✅ FRED API + 27 transform + retry/backoff |
| `sentiment_pipeline.py` | ✅ NewsAPI + FinBERT ensemble |
| `paper_broker.py` | ✅ Mark-to-market, position sizing |
| `orchestrate.py` | ✅ |
| `db.py` | ✅ SQLite (WAL→TRUNCATE fallback) |
| `scripts/daily_run.py` | ✅ Cloud orkestratörü |
| `app/streamlit_app.py` | ✅ Read-only dashboard |
| `.github/workflows/daily.yml` | ✅ Her gece 02:10 UTC |
| `.github/workflows/sync-hf-space.yml` | ✅ DB push → Space sync |

## Son eğitilen model tarihi

- Train: 2022-01-01 → 2025-12-31 (1461 gün)
- Val:   2026-01-01 → 2026-04-15 (100–105 gün)
- Son görülen gün: **2026-04-20**

## Dikkat

- **Rolling retrain (aylık önerilen)**: `features_code/v4/production.py` script'ini train_end/val_end tarihlerini ileri alarak yeniden çalıştır → yeni `v4_production/` → commit & push → sonraki cron yeni modelle çalışır.
- **BTC/BNB/DOT** için ilk eğitimde `best_iter=1` çıktı (val penceresi kısa). Retrain öncesi val ≥180 gün olmalı.
- **Order execution YOK**: Paper broker simülasyon. Gerçek emir yerleştirme yok.
- **Finansal uyarı**: Sinyaller araştırma amaçlıdır. Yatırım tavsiyesi değildir.
