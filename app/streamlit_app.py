"""Streamlit dashboard — HuggingFace Spaces veya yerel kullanım.

Read-only: günlük GitHub Actions job'ının yazdığı SQLite'tan okur.
3 tab:
    1) Bugünkü sinyaller — 10 coin tablosu, renk kodlu
    2) Portföy — equity eğrisi, açık pozisyonlar
    3) İşlem geçmişi — filtrelenebilir

Çalıştırma:
    streamlit run app/streamlit_app.py

HF Spaces'te:
    - Space type: Streamlit
    - Requirements: app/requirements.txt
    - README.md başına YAML meta: sdk: streamlit
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

# code/ ve scripts/ path — hem yerel (live_trading/app/) hem HF Space (flat) layout
_here = Path(__file__).resolve().parent
_candidates = [
    _here,                    # HF Space: code/, data_live/ streamlit_app.py ile aynı seviyede
    _here / "code",           # HF Space: code/ streamlit_app.py içinde
    _here.parent,             # yerel: live_trading/
    _here.parent / "code",    # yerel: live_trading/code/
    _here.parent / "scripts",
]
for p in _candidates:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from db import DB, DB_PATH  # noqa: E402


# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="V4 Crypto Signals",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# Data loaders (cached)
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def load_signals(days: int = 30) -> pd.DataFrame:
    with DB() as db:
        db.init()
        return db.read_signals(days=days)


@st.cache_data(ttl=300, show_spinner=False)
def load_trades(days: int | None = None) -> pd.DataFrame:
    with DB() as db:
        db.init()
        return db.read_trades(days=days)


@st.cache_data(ttl=300, show_spinner=False)
def load_equity(days: int | None = None) -> pd.DataFrame:
    with DB() as db:
        db.init()
        return db.read_equity(days=days)


@st.cache_data(ttl=300, show_spinner=False)
def load_positions() -> pd.DataFrame:
    with DB() as db:
        db.init()
        return db.read_positions()


@st.cache_data(ttl=300, show_spinner=False)
def load_meta(key: str) -> str | None:
    with DB() as db:
        db.init()
        return db.get_meta(key)


@st.cache_data(ttl=300, show_spinner=False)
def load_last_closes() -> dict:
    """Açık pozisyonları mark etmek için son close'ları çek."""
    out = {}
    with DB() as db:
        db.init()
        rows = db.conn.execute(
            "SELECT coin, MAX(date) d FROM ohlcv_cache GROUP BY coin"
        ).fetchall()
        for r in rows:
            c = db.read_last_close(r["coin"], r["d"])
            if c is not None:
                out[r["coin"]] = c
    return out


# ============================================================
# Header
# ============================================================

def header() -> None:
    col1, col2, col3 = st.columns([3, 2, 2])
    col1.title("📈 V4 Crypto Signals")
    last_run = load_meta("last_run") or "—"
    last_date = load_meta("last_date") or "—"
    col2.metric("Son güncelleme", last_run.split("T")[0] if "T" in last_run else last_run)
    col3.metric("Son sinyal tarihi", last_date)

    if not DB_PATH.exists():
        st.error(f"DB yok: {DB_PATH}. Önce `python scripts/daily_run.py` çalıştır.")
        st.stop()


# ============================================================
# Tab 1 — Today's signals
# ============================================================

SIGNAL_COLOR = {"BUY": "#16a34a", "SELL": "#dc2626", "HOLD": "#6b7280"}


def tab_signals() -> None:
    df = load_signals(days=30)
    if df.empty:
        st.warning("Sinyal yok.")
        return

    latest_date = df["date"].max()
    today = df[df["date"] == latest_date].copy().sort_values("coin")

    st.subheader(f"Bugün ({latest_date})")
    counts = today["signal"].value_counts().to_dict()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BUY",  int(counts.get("BUY", 0)))
    c2.metric("SELL", int(counts.get("SELL", 0)))
    c3.metric("HOLD", int(counts.get("HOLD", 0)))
    c4.metric("Toplam", len(today))

    # Tablo
    show = today[["coin", "signal", "p_buy", "p_hold", "p_sell",
                  "buy_th", "sell_th", "horizon"]].copy()
    show.columns = ["Coin", "Sinyal", "P(BUY)", "P(HOLD)", "P(SELL)",
                    "buy_th", "sell_th", "horizon"]
    for c in ("P(BUY)", "P(HOLD)", "P(SELL)", "buy_th", "sell_th"):
        show[c] = show[c].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "—")

    def _color(row):
        c = SIGNAL_COLOR.get(row["Sinyal"], "#6b7280")
        return [f"background-color: {c}1a"] * len(row)

    st.dataframe(show.style.apply(_color, axis=1), use_container_width=True, hide_index=True)

    # Son 30 gün özet
    st.subheader("Son 30 gün — günlük sinyal dağılımı")
    piv = (df.groupby(["date", "signal"]).size()
             .unstack(fill_value=0)
             .reindex(columns=["BUY", "HOLD", "SELL"], fill_value=0)
             .reset_index()
             .sort_values("date"))
    st.bar_chart(piv.set_index("date"), use_container_width=True)


# ============================================================
# Tab 2 — Portföy
# ============================================================

def tab_portfolio() -> None:
    eq = load_equity(days=30)
    pos = load_positions()
    marks = load_last_closes()

    if eq.empty:
        st.info("Henüz equity kaydı yok.")
    else:
        latest = eq.iloc[-1]
        first = eq.iloc[0]
        pnl = latest["equity"] - first["equity"]
        pnl_pct = 100 * pnl / first["equity"] if first["equity"] else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Equity",  f"${latest['equity']:,.2f}",
                  delta=f"{pnl_pct:+.2f}% ({pnl:+.2f})")
        c2.metric("Cash",    f"${latest['cash']:,.2f}")
        c3.metric("Pozisyonlar",
                  f"${latest['positions_value']:,.2f}")
        c4.metric("Açık pozisyon", int(latest["n_open"]))

        st.line_chart(eq.set_index("date")[["equity", "cash"]],
                      use_container_width=True)

    st.subheader("Açık pozisyonlar")
    if pos.empty:
        st.caption("Açık pozisyon yok.")
    else:
        p = pos.copy()
        p["mark"] = p["coin"].map(marks).fillna(p["avg_price"])
        p["mkt_value"] = p["qty"] * p["mark"]
        p["unreal_pnl"] = (p["mark"] - p["avg_price"]) * p["qty"]
        p["unreal_pnl_pct"] = 100 * (p["mark"] / p["avg_price"] - 1)
        show = p[["coin", "qty", "avg_price", "mark",
                  "mkt_value", "unreal_pnl", "unreal_pnl_pct", "opened_at"]].copy()
        show.columns = ["Coin", "Adet", "Giriş", "Mark", "MV ($)",
                        "Unreal PnL ($)", "Unreal PnL (%)", "Açılış"]
        for c in ("Giriş", "Mark", "MV ($)", "Unreal PnL ($)", "Unreal PnL (%)"):
            show[c] = show[c].apply(lambda v: f"{v:,.2f}" if pd.notna(v) else "—")
        st.dataframe(show, use_container_width=True, hide_index=True)


# ============================================================
# Tab 3 — Geçmiş işlemler
# ============================================================

def tab_history() -> None:
    st.subheader("İşlem geçmişi")

    c1, c2 = st.columns([1, 3])
    days = c1.selectbox("Pencere", [7, 30, 90, 365, None],
                        index=1,
                        format_func=lambda x: f"{x} gün" if x else "Tümü")

    tr = load_trades(days=days)
    if tr.empty:
        st.caption("İşlem yok.")
        return

    coin_filter = c2.multiselect("Coin filtresi",
                                 sorted(tr["coin"].unique()),
                                 default=[])
    if coin_filter:
        tr = tr[tr["coin"].isin(coin_filter)]

    show = tr[["date", "coin", "side", "qty", "price",
               "fee", "gross", "realized_pnl", "note"]].copy()
    show.columns = ["Tarih", "Coin", "Yön", "Adet", "Fiyat",
                    "Komisyon", "Gross", "Realized PnL", "Not"]
    st.dataframe(show, use_container_width=True, hide_index=True)

    # realize pnl özet
    realized = tr["realized_pnl"].fillna(0).sum()
    fees = tr["fee"].fillna(0).sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Toplam realized PnL", f"${realized:,.2f}")
    c2.metric("Toplam komisyon", f"${fees:,.2f}")
    c3.metric("İşlem adedi", len(tr))


# ============================================================
# Main
# ============================================================

def main() -> None:
    header()
    tab1, tab2, tab3 = st.tabs(["📊 Bugünkü sinyaller",
                                "💼 Portföy",
                                "📜 Geçmiş"])
    with tab1:
        tab_signals()
    with tab2:
        tab_portfolio()
    with tab3:
        tab_history()

    with st.sidebar:
        st.markdown("### Ayarlar")
        if st.button("🔄 Cache'i temizle"):
            st.cache_data.clear()
            st.rerun()
        st.caption(f"DB: `{DB_PATH}`")
        st.caption("**Meta**")
        for k in ("last_run", "last_date", "last_model_version"):
            v = load_meta(k)
            if v:
                st.caption(f"{k}: `{v}`")


if __name__ == "__main__":
    main()
