"""V4 — Canlı inference modülü.

Artifact'lar live_trading/artifacts/v4_production/{coin}/ altından yüklenir.
Özellikler feature_builders.load_*() ile çekilir — önce data_live/, sonra backfill.

Public API:
    load_coin_artifacts(coin) -> dict
    predict_signal_for_date(coin, as_of_date) -> dict
    predict_signal_from_row(coin, feature_row, artifacts=None) -> dict
    build_feature_row(coin, as_of_date) -> (pd.Series, meta)

CLI:
    python3 inference.py              # 10 coin için smoke test
    python3 inference.py BTC 2026-04-20
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from paths import ARTIFACTS_ROOT, COINS
from feature_builders import (
    load_sentiment_features, load_macro_features, load_technical_features,
)


# ---------- artifact loader ----------

def load_coin_artifacts(coin: str) -> dict:
    coin_dir = ARTIFACTS_ROOT / coin
    if not coin_dir.exists():
        raise FileNotFoundError(f"{coin} artifact klasörü yok: {coin_dir}")

    model      = lgb.Booster(model_file=str(coin_dir / "model.lgb"))
    thresholds = json.loads((coin_dir / "thresholds.json").read_text())
    horizons   = json.loads((coin_dir / "horizons.json").read_text())
    quantiles  = json.loads((coin_dir / "label_quantiles.json").read_text())
    feat       = json.loads((coin_dir / "feature_columns.json").read_text())

    return {
        "coin": coin,
        "model": model,
        "feature_columns": feat["columns"],
        "buy_th":   float(thresholds["buy_th"]),
        "sell_th":  float(thresholds["sell_th"]),
        "horizon":  int(horizons["horizon"]),
        "q30":      float(quantiles["q30"]),
        "q70":      float(quantiles["q70"]),
    }


# ---------- feature row ----------

def build_feature_row(coin: str, as_of_date) -> tuple[pd.Series, dict]:
    as_of_date = pd.Timestamp(as_of_date)

    sent = load_sentiment_features(coin)
    sent_row = sent.loc[sent["date"] == as_of_date]

    macro = load_macro_features()
    macro_row = macro.loc[macro["date"] == as_of_date]

    tech = load_technical_features(coin)
    tech_row = tech.loc[tech["date"] == as_of_date]

    meta = {
        "coin": coin,
        "as_of_date": str(as_of_date.date()),
        "has_sent":  bool(not sent_row.empty),
        "has_macro": bool(not macro_row.empty),
        "has_tech":  bool(not tech_row.empty),
    }
    if sent_row.empty:
        raise ValueError(f"Sentiment satırı yok: {coin} @ {as_of_date.date()}")

    sent_s = sent_row.iloc[0].drop(labels=["date", "close", "fwd_return", "label"], errors="ignore")
    full = dict(sent_s)

    if not macro_row.empty:
        m = macro_row.iloc[0].drop(labels=["date"], errors="ignore")
        for k, v in m.items():
            full[f"macro_{k}"] = v
    if not tech_row.empty:
        t = tech_row.iloc[0].drop(labels=["date", "coin"], errors="ignore")
        for k, v in t.items():
            full[f"tech_{k}"] = v

    return pd.Series(full), meta


# ---------- predict ----------

# A3: tek kaynak config.py
from config import (
    DIR_MARGIN_DEFAULT,
    HOLD_VETO_DEFAULT,
    DIR_MARGIN_WEAK,
    HOLD_VETO_WEAK,
    WEAK_COINS,
)


def _signal_from_probs(p_sell: float, p_hold: float, p_buy: float,
                       buy_th: float, sell_th: float,
                       dir_margin: float = DIR_MARGIN_DEFAULT,
                       hold_veto: float = HOLD_VETO_DEFAULT) -> tuple[int, str]:
    """3-kapi sinyal kurali. Donus: (signal_int, reason).
    BUY  iff: p_buy  >= buy_th AND  (p_buy  - p_sell) >= dir_margin
                                AND (p_hold - p_buy)  <= hold_veto
    SELL iff: p_sell >= sell_th AND (p_sell - p_buy)  >= dir_margin
                                AND (p_hold - p_sell) <= hold_veto
    Diger: HOLD."""
    # BUY denemesi
    if p_buy >= buy_th and p_buy > p_sell:
        if (p_buy - p_sell) < dir_margin:
            return 0, "blocked_margin_buy"
        if (p_hold - p_buy) > hold_veto:
            return 0, "blocked_hold_veto_buy"
        return 1, "buy"
    # SELL denemesi
    if p_sell >= sell_th and p_sell > p_buy:
        if (p_sell - p_buy) < dir_margin:
            return 0, "blocked_margin_sell"
        if (p_hold - p_sell) > hold_veto:
            return 0, "blocked_hold_veto_sell"
        return -1, "sell"
    return 0, "below_threshold"


def predict_signal_from_row(coin: str, feature_row: pd.Series,
                            artifacts: dict | None = None) -> dict:
    if artifacts is None:
        artifacts = load_coin_artifacts(coin)

    feat_cols = artifacts["feature_columns"]
    x = np.array([feature_row.get(c, np.nan) for c in feat_cols],
                 dtype=np.float32).reshape(1, -1)

    proba = artifacts["model"].predict(x)[0]
    p_sell, p_hold, p_buy = float(proba[0]), float(proba[1]), float(proba[2])

    # Per-coin gate parametreleri: weak ise siki, degilse default
    is_weak = coin in WEAK_COINS
    dir_margin = DIR_MARGIN_WEAK if is_weak else DIR_MARGIN_DEFAULT
    hold_veto  = HOLD_VETO_WEAK  if is_weak else HOLD_VETO_DEFAULT

    sig_int, gate_reason = _signal_from_probs(
        p_sell, p_hold, p_buy,
        buy_th=artifacts["buy_th"], sell_th=artifacts["sell_th"],
        dir_margin=dir_margin, hold_veto=hold_veto,
    )
    sig_txt = {1: "BUY", 0: "HOLD", -1: "SELL"}[sig_int]

    return {
        "coin": coin,
        "signal": sig_txt,
        "signal_int": sig_int,
        "p_sell": round(p_sell, 4),
        "p_hold": round(p_hold, 4),
        "p_buy":  round(p_buy,  4),
        "buy_th": artifacts["buy_th"],
        "sell_th": artifacts["sell_th"],
        "horizon": artifacts["horizon"],
        "n_features": len(feat_cols),
        "gate_reason": gate_reason,
        "gate_dir_margin": dir_margin,
        "gate_hold_veto": hold_veto,
        "coin_tier": "weak" if is_weak else "strong",
    }


def predict_signal_for_date(coin: str, as_of_date) -> dict:
    art = load_coin_artifacts(coin)
    row, meta = build_feature_row(coin, as_of_date)
    out = predict_signal_from_row(coin, row, artifacts=art)
    out.update({
        "as_of_date": meta["as_of_date"],
        "has_sent":  meta["has_sent"],
        "has_macro": meta["has_macro"],
        "has_tech":  meta["has_tech"],
    })
    return out


# ---------- CLI ----------

def smoke_test() -> None:
    print(f"=== V4 canlı inference smoke ({datetime.utcnow().date()}) ===\n")
    results = []
    for coin in COINS:
        sent = load_sentiment_features(coin)
        last = sent["date"].max()
        for off in range(0, 10):
            d = last - pd.Timedelta(days=off)
            try:
                out = predict_signal_for_date(coin, d)
                if out["has_sent"] and out["has_macro"] and out["has_tech"]:
                    results.append(out)
                    break
            except Exception:
                continue
        else:
            print(f"[{coin}] ERR: uygun son satır bulunamadı")

    if results:
        df = pd.DataFrame(results)
        cols = ["coin", "as_of_date", "signal", "p_sell", "p_hold", "p_buy",
                "buy_th", "sell_th", "horizon"]
        print(df[cols].to_string(index=False))
        print()
        print(f"Sinyal özet: {df['signal'].value_counts().to_dict()}")
    else:
        print("Hiç sinyal üretilemedi.")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        coin, date = sys.argv[1], sys.argv[2]
        out = predict_signal_for_date(coin, date)
        print(json.dumps(out, indent=2))
    else:
        smoke_test()
