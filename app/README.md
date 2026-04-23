---
title: V4 Crypto Signals
emoji: 📈
colorFrom: indigo
colorTo: green
sdk: streamlit
sdk_version: "1.31.0"
app_file: streamlit_app.py
pinned: false
short_description: "V4 LightGBM crypto signals dashboard (10 coin, read-only)"
---

# V4 Crypto Signals

Günlük `daily-signals` GitHub Actions job'ının ürettiği kararları gösteren
read-only dashboard.

## Nasıl çalışıyor?

1. Yerelde eğitilmiş LightGBM modelleri (`artifacts/v4_production/{COIN}/`) git repo'da.
2. GitHub Actions her gece 02:10 UTC'de:
   - Binance/FRED/NewsAPI'den son veriyi çeker
   - 10 coin için sinyal üretir
   - PaperBroker'a uygular
   - SQLite'a yazar, commit eder
3. Bu Space repo'nun son halini okur, dashboard sunar.

## Manuel test (yerel)

```bash
cd live_trading
streamlit run app/streamlit_app.py
```

## Gizlilik / Disclaimer

Bu bir araştırma/paper-trading aracıdır. Gerçek para ile trade yapmaz.
Sinyaller yatırım tavsiyesi değildir.
