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


@st.cache_data(ttl=300, show_spinner=False)
def load_decisions(days: int = 30) -> pd.DataFrame:
    """A5.2: broker_decisions tablosundan kararlari oku."""
    with DB() as db:
        db.init()
        return db.read_decisions(days=days)


# ============================================================
# Header
# ============================================================

def header() -> None:
    col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
    col1.title("📈 V4 Crypto Signals")

    last_run = load_meta("last_run") or "—"
    last_date = load_meta("last_date") or "—"
    last_mv = load_meta("last_model_version") or "—"
    last_mca = load_meta("last_model_created_at") or ""

    col2.metric("Son güncelleme",
                last_run.split("T")[0] if "T" in last_run else last_run)
    col3.metric("Son sinyal tarihi", last_date)
    # A5.3: aktif model versiyonu
    mv_label = last_mv if last_mv == "—" else last_mv[:12]
    mv_delta = last_mca.split("T")[0] if "T" in last_mca else (last_mca or None)
    col4.metric("Aktif model", mv_label, delta=mv_delta,
                delta_color="off",
                help="last_model_version (artifact_hash) + production training tarihi")

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
        # A5.1: marks dict'te olmayan coin icin entry'ye sessiz fallback YAPMA.
        # NaN birak; tabloda "—" olarak gozukur, unreal PnL hesaplanmaz.
        p["mark"] = p["coin"].map(marks)
        missing_marks = sorted(p.loc[p["mark"].isna(), "coin"].tolist())
        if missing_marks:
            st.warning(
                f"⚠️ {len(missing_marks)} coin için son fiyat (mark) alınamadı; "
                f"unrealized PnL hesaplanamadı: **{', '.join(missing_marks)}**. "
                f"Daha güncel veri için bir sonraki daily-signals run'ını bekleyin."
            )

        p["mkt_value"]      = p["qty"] * p["mark"]
        p["unreal_pnl"]     = (p["mark"] - p["avg_price"]) * p["qty"]
        p["unreal_pnl_pct"] = 100 * (p["mark"] / p["avg_price"] - 1)
        # Duration: bugünden opened_at'e kaç gün (her ikisini tz-naive yap)
        opened = pd.to_datetime(p["opened_at"], errors="coerce", utc=True)
        opened = opened.dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
        today = pd.Timestamp.utcnow()
        if today.tzinfo is not None:
            today = today.tz_convert("UTC").tz_localize(None)
        today = today.normalize()
        p["duration"] = (today - opened).dt.days

        show = p[["coin", "qty", "avg_price", "mark",
                  "mkt_value", "unreal_pnl", "unreal_pnl_pct",
                  "opened_at", "duration"]].reset_index(drop=True)
        show.columns = ["Coin", "Adet", "Giriş", "Mark", "MV ($)",
                        "Unreal PnL ($)", "Unreal PnL (%)",
                        "Açılış", "Gün"]

        # Ham sayısal değerleri renk için saklayalım
        raw_unreal = show["Unreal PnL ($)"].astype(float).values
        for c in ("Giriş", "Mark", "MV ($)", "Unreal PnL ($)", "Unreal PnL (%)"):
            show[c] = show[c].apply(lambda v: f"{v:,.2f}" if pd.notna(v) else "—")

        def _color_unreal(row):
            try:
                v = float(raw_unreal[row.name])
                if pd.isna(v):
                    return ["background-color: transparent"] * len(row)
            except Exception:
                v = 0.0
            color = "#16a34a22" if v > 0 else ("#dc262622" if v < 0 else "transparent")
            return [f"background-color: {color}"] * len(row)

        st.dataframe(show.style.apply(_color_unreal, axis=1),
                     use_container_width=True, hide_index=True)


# ============================================================
# Tab 3 — Geçmiş işlemler
# ============================================================

def _match_roundtrips(tr: pd.DataFrame) -> pd.DataFrame:
    """FIFO: her coin için LONG ve SHORT roundtrip'lerini ayrı eşleştir.

    Trade side dilini paper_broker tanımlar:
        BUY_OPEN     — LONG açılış
        SELL_CLOSE   — LONG kapanış      (LONG roundtrip kapatır)
        SELL_OPEN    — SHORT açılış
        BUY_COVER    — SHORT kapanış     (SHORT roundtrip kapatır)
    Eski "BUY"/"SELL" formatı da fallback olarak desteklenir (eski trades.csv).

    LONG PnL  = (close_px − open_px) × qty                  → fiyat artarsa kâr
    SHORT PnL = (open_px  − close_px) × qty                 → fiyat düşerse kâr
    pnl_pct dönüş yüzdesi olarak yöne göre hesaplanır.

    Dönüş: her satır bir tam kapatılmış pozisyon.
        coin, side (LONG/SHORT), open_date, close_date, qty,
        entry, exit, days, pnl, pnl_pct
    """
    cols = ["coin", "side", "open_date", "close_date", "qty",
            "entry", "exit", "days", "pnl", "pnl_pct"]
    if tr.empty:
        return pd.DataFrame(columns=cols)

    tr2 = tr.copy()
    tr2["date"] = pd.to_datetime(tr2["date"], errors="coerce")
    tr2 = tr2.sort_values(["coin", "date"]).reset_index(drop=True)

    from collections import deque, defaultdict
    long_q:  dict[str, deque] = defaultdict(deque)   # coin → BUY_OPEN kuyruğu
    short_q: dict[str, deque] = defaultdict(deque)   # coin → SELL_OPEN kuyruğu

    rows: list[dict] = []

    def _close_against(queue: deque, direction: str, coin: str,
                       close_qty: float, close_px: float, close_dt) -> None:
        """Verilen kuyruktan FIFO sırada eşleştir, roundtrip satırları üret."""
        remaining = close_qty
        while remaining > 1e-12 and queue:
            o_dt, o_q, o_px = queue[0]
            take = min(o_q, remaining)
            if direction == "LONG":
                pnl = (close_px - o_px) * take
                pct = 100 * (close_px / o_px - 1) if o_px else 0.0
            else:  # SHORT
                pnl = (o_px - close_px) * take
                pct = 100 * (o_px / close_px - 1) if close_px else 0.0
            try:
                days = max((close_dt.normalize() - o_dt.normalize()).days, 0) \
                    if pd.notna(close_dt) and pd.notna(o_dt) else 0
            except Exception:
                days = 0
            rows.append({
                "coin": coin,
                "side": direction,
                "open_date":  o_dt.date().isoformat() if pd.notna(o_dt) else "",
                "close_date": close_dt.date().isoformat() if pd.notna(close_dt) else "",
                "qty":     round(take, 6),
                "entry":   o_px,
                "exit":    close_px,
                "days":    int(days),
                "pnl":     round(pnl, 2),
                "pnl_pct": round(pct, 3),
            })
            queue[0][1] = o_q - take
            if queue[0][1] <= 1e-12:
                queue.popleft()
            remaining -= take

    for _, r in tr2.iterrows():
        side = str(r.get("side", "")).upper().strip()
        coin = str(r.get("coin", ""))
        q    = float(r.get("qty", 0) or 0)
        px   = float(r.get("price", 0) or 0)
        dt   = r.get("date")
        if q <= 0 or coin == "":
            continue

        if side in ("BUY_OPEN",):
            long_q[coin].append([dt, q, px])
        elif side in ("SELL_OPEN",):
            short_q[coin].append([dt, q, px])
        elif side in ("SELL_CLOSE",):
            _close_against(long_q[coin], "LONG", coin, q, px, dt)
        elif side in ("BUY_COVER",):
            _close_against(short_q[coin], "SHORT", coin, q, px, dt)
        # ---- legacy fallback (eski "BUY"/"SELL" yalnız LONG akışı) ----
        elif side == "BUY":
            long_q[coin].append([dt, q, px])
        elif side == "SELL":
            _close_against(long_q[coin], "LONG", coin, q, px, dt)
        # bilinmeyen side'lar atlanır

    return pd.DataFrame(rows, columns=cols)


def tab_history() -> None:
    st.subheader("İşlem geçmişi")

    c1, c2 = st.columns([1, 3])
    days = c1.selectbox("Pencere", [7, 30, 90, 365, None],
                        index=1,
                        format_func=lambda x: f"{x} gün" if x else "Tümü")

    # roundtrip eşleşmesi için: kapama'ya ait BUY geriye dönük gerekebilir
    # → filtreyi eşleşme sonrası uygulayalım
    tr_full = load_trades(days=None)
    if tr_full.empty:
        st.caption("İşlem yok.")
        return

    tr = load_trades(days=days)
    if tr.empty:
        st.caption("Seçilen pencerede işlem yok.")
        return

    coin_filter = c2.multiselect("Coin filtresi",
                                 sorted(tr["coin"].unique()),
                                 default=[])
    if coin_filter:
        tr = tr[tr["coin"].isin(coin_filter)]

    # ---- KAPANMIŞ POZİSYONLAR (roundtrip) ----
    st.markdown("#### ✅ Kapanmış pozisyonlar")
    rt_all = _match_roundtrips(tr_full)
    if coin_filter and not rt_all.empty:
        rt_all = rt_all[rt_all["coin"].isin(coin_filter)]
    # Seçilen pencereye göre close_date filtresi
    if days is not None and not rt_all.empty:
        cutoff = (pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=days)).date().isoformat()
        rt = rt_all[rt_all["close_date"] >= cutoff].copy()
    else:
        rt = rt_all.copy()

    if rt.empty:
        st.caption("Henüz kapanmış pozisyon yok.")
    else:
        # Win/loss istatistikleri
        wins = rt[rt["pnl"] > 0]
        losses = rt[rt["pnl"] < 0]
        total_pnl = rt["pnl"].sum()
        win_rate = 100 * len(wins) / len(rt) if len(rt) else 0
        avg_win = wins["pnl"].mean() if len(wins) else 0
        avg_loss = losses["pnl"].mean() if len(losses) else 0
        profit_factor = (wins["pnl"].sum() / abs(losses["pnl"].sum())) \
            if len(losses) and losses["pnl"].sum() != 0 else float("inf") if len(wins) else 0
        avg_days = rt["days"].mean() if len(rt) else 0

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Kapanan", len(rt))
        k2.metric("Win rate", f"{win_rate:.1f}%",
                  delta=f"{len(wins)}W / {len(losses)}L")
        k3.metric("Toplam PnL", f"${total_pnl:,.2f}")
        k4.metric("Ort. kazanç / kayıp",
                  f"${avg_win:,.2f} / ${avg_loss:,.2f}")
        k5.metric("Ort. tutma", f"{avg_days:.1f} gün")

        k6, k7, k8 = st.columns(3)
        pf_txt = "∞" if profit_factor == float("inf") else f"{profit_factor:.2f}"
        k6.metric("Profit factor", pf_txt,
                  help="Toplam kazanç / toplam kayıp. >1 karlı.")
        best = rt.loc[rt["pnl"].idxmax()] if len(rt) else None
        if best is not None:
            k7.metric("En iyi işlem",
                      f"{best['coin']} +${best['pnl']:,.2f}",
                      delta=f"{best['pnl_pct']:+.2f}%")

        # LONG vs SHORT kırılımı
        if "side" in rt.columns:
            n_long  = int((rt["side"] == "LONG").sum())
            n_short = int((rt["side"] == "SHORT").sum())
            pnl_long  = float(rt.loc[rt["side"] == "LONG",  "pnl"].sum())
            pnl_short = float(rt.loc[rt["side"] == "SHORT", "pnl"].sum())
            k8.metric("LONG / SHORT (adet · PnL)",
                      f"{n_long}L / {n_short}S",
                      delta=f"${pnl_long:+,.0f} / ${pnl_short:+,.0f}",
                      help="Yöne göre kapanmış işlem sayısı ve toplam realized PnL.")

        # Cumulative PnL zaman içinde
        rt_sorted = rt.sort_values("close_date").copy()
        rt_sorted["cum_pnl"] = rt_sorted["pnl"].cumsum()
        st.line_chart(
            rt_sorted.set_index("close_date")[["cum_pnl"]]
                     .rename(columns={"cum_pnl": "Kümülatif realize PnL ($)"}),
            use_container_width=True,
        )

        # Detaylı tablo
        show_rt = rt.sort_values("close_date", ascending=False).copy()
        show_rt["result"] = show_rt["pnl"].apply(
            lambda v: "🟢 WIN" if v > 0 else ("🔴 LOSS" if v < 0 else "⚪ FLAT"))
        # "side" kolonu eski roundtrip dataframe'inde olmayabilir → güvenli al
        if "side" not in show_rt.columns:
            show_rt["side"] = "LONG"
        show_rt["side_label"] = show_rt["side"].map(
            {"LONG": "🟢 LONG", "SHORT": "🔴 SHORT"}).fillna(show_rt["side"])
        disp = show_rt[["coin", "side_label", "open_date", "close_date", "days",
                        "qty", "entry", "exit", "pnl", "pnl_pct", "result"]].copy()
        disp.columns = ["Coin", "Yön", "Açılış", "Kapanış", "Gün",
                        "Adet", "Giriş", "Çıkış", "PnL ($)", "PnL (%)", "Sonuç"]
        raw_pnl = show_rt["pnl"].values
        for c in ("Adet",):
            disp[c] = disp[c].apply(lambda v: f"{v:,.6f}".rstrip("0").rstrip("."))
        for c in ("Giriş", "Çıkış", "PnL ($)"):
            disp[c] = disp[c].apply(lambda v: f"{v:,.2f}")
        disp["PnL (%)"] = disp["PnL (%)"].apply(lambda v: f"{v:+.2f}%")

        def _row_color(row):
            i = row.name
            v = raw_pnl[i] if i < len(raw_pnl) else 0
            c = "#16a34a22" if v > 0 else ("#dc262622" if v < 0 else "transparent")
            return [f"background-color: {c}"] * len(row)

        st.dataframe(disp.style.apply(_row_color, axis=1),
                     use_container_width=True, hide_index=True)

    st.markdown("---")

    # ---- TÜM TRADE'LER (raw) ----
    st.markdown("#### 📜 Tüm işlemler (ham)")
    show = tr[["date", "coin", "side", "qty", "price",
               "fee", "gross", "realized_pnl", "note"]].copy()
    show.columns = ["Tarih", "Coin", "Yön", "Adet", "Fiyat",
                    "Komisyon", "Gross", "Realized PnL", "Not"]

    raw_side = show["Yön"].values

    def _side_color(row):
        i = row.name
        s = str(raw_side[i]).upper() if i < len(raw_side) else ""
        # LONG açılış → koyu yeşil; LONG kapanış → açık yeşil/kırmızı (PnL'e göre)
        # SHORT açılış → kırmızı; SHORT kapanış (cover) → mavi/yeşil
        if s == "BUY_OPEN":
            c = "#16a34a22"     # LONG aç (yeşil)
        elif s == "SELL_CLOSE":
            c = "#16a34a14"     # LONG kapa (açık yeşil)
        elif s == "SELL_OPEN":
            c = "#dc262622"     # SHORT aç (kırmızı)
        elif s == "BUY_COVER":
            c = "#dc262614"     # SHORT kapa (açık kırmızı)
        elif s == "BUY":         # legacy
            c = "#16a34a14"
        elif s == "SELL":        # legacy
            c = "#dc262614"
        else:
            c = "transparent"
        return [f"background-color: {c}"] * len(row)

    st.dataframe(show.style.apply(_side_color, axis=1),
                 use_container_width=True, hide_index=True)

    # Özet (pencere içinde)
    realized = tr["realized_pnl"].fillna(0).sum()
    fees = tr["fee"].fillna(0).sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Pencere realized PnL", f"${realized:,.2f}")
    c2.metric("Pencere komisyonu", f"${fees:,.2f}")
    c3.metric("Pencere işlem adedi", len(tr))


# ============================================================
# Tab 4 — Kararlar (A5.2)
# ============================================================

def _categorize_reason(r) -> str:
    """broker_decisions.reason -> kategori (gate dagilimi icin)."""
    if r is None or r == "" or r == "ok":
        return "✅ ok"
    s = str(r)
    if s.startswith("missing_features"):  return "⚠️ missing_features"
    if s.startswith("low_news_count"):    return "📰 low_news_count"
    if s.startswith("stale_news"):        return "📅 stale_news"
    if s == "no_sentiment_row":           return "📭 no_sentiment_row"
    if s.startswith("blocked_margin"):    return "🚧 blocked_margin"
    if s.startswith("blocked_hold_veto"): return "🚧 blocked_hold_veto"
    if s == "below_threshold":            return "🔻 below_threshold"
    return s


def tab_decisions() -> None:
    """A5.2: Per-coin per-day kararlar tabi — A4'te broker_decisions tablosu."""
    df = load_decisions(days=30)
    if df.empty:
        st.info("Henüz karar kaydı yok. `daily_run.py` calistirildiginda dolar.")
        return

    df = df.copy()
    df["category"] = df["reason"].apply(_categorize_reason)
    latest_date = df["date"].max()
    today_df = df[df["date"] == latest_date].copy().sort_values("coin")

    # === Bugün metrik kartları ===
    st.subheader(f"Bugün ({latest_date}) — gate kırılımı")
    cat_counts = today_df["category"].value_counts()
    if len(cat_counts) > 0:
        cols = st.columns(min(5, len(cat_counts)))
        for i, (cat, n) in enumerate(cat_counts.head(5).items()):
            cols[i].metric(cat, int(n))

    # === Bugünkü kararlar tablosu ===
    st.subheader("Bugünkü kararlar")
    show = today_df[["coin", "raw_signal", "final_action", "reason",
                     "model_version"]].copy()
    show.columns = ["Coin", "Ham Sinyal", "Final Aksiyon", "Reason", "Model"]

    def _color_action(row):
        a = row["Final Aksiyon"]
        c = SIGNAL_COLOR.get(a, "#6b7280")
        return [f"background-color: {c}1a"] * len(row)

    st.dataframe(show.style.apply(_color_action, axis=1),
                 use_container_width=True, hide_index=True)

    # === 30 gün gate dağılımı bar chart ===
    st.subheader("Son 30 gün — günlük gate dağılımı")
    piv = (df.groupby(["date", "category"]).size()
             .unstack(fill_value=0)
             .reset_index()
             .sort_values("date"))
    st.bar_chart(piv.set_index("date"), use_container_width=True)

    # === Filtreli detaylı geçmiş ===
    st.subheader("Detaylı geçmiş")
    c1, c2 = st.columns([1, 3])
    days_sel = c1.selectbox("Pencere", [7, 14, 30, 90], index=2,
                            format_func=lambda x: f"{x} gün")
    coin_filter = c2.multiselect("Coin filtresi",
                                  sorted(df["coin"].unique()), default=[])

    cutoff = (pd.Timestamp.utcnow().tz_localize(None).normalize()
              - pd.Timedelta(days=days_sel)).date().isoformat()
    df_filt = df[df["date"] >= cutoff].copy()
    if coin_filter:
        df_filt = df_filt[df_filt["coin"].isin(coin_filter)]
    df_filt = df_filt.sort_values(["date", "coin"], ascending=[False, True])

    show2 = df_filt[["date", "coin", "raw_signal", "final_action",
                     "reason", "model_version"]].copy()
    show2.columns = ["Tarih", "Coin", "Ham Sinyal", "Final", "Reason", "Model"]
    st.dataframe(show2, use_container_width=True, hide_index=True)

    # Özet
    n_blocked = (df_filt["final_action"] != df_filt["raw_signal"]).sum()
    st.caption(
        f"Toplam {len(df_filt)} karar · {df_filt['coin'].nunique()} coin · "
        f"**{n_blocked}** sinyal gate'lerce engellendi."
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    header()
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Bugünkü sinyaller",
                                       "💼 Portföy",
                                       "📜 Geçmiş",
                                       "📋 Kararlar"])
    with tab1:
        tab_signals()
    with tab2:
        tab_portfolio()
    with tab3:
        tab_history()
    with tab4:
        tab_decisions()

    with st.sidebar:
        st.markdown("### Ayarlar")
        if st.button("🔄 Cache'i temizle"):
            st.cache_data.clear()
            st.rerun()
        st.caption(f"DB: `{DB_PATH}`")
        st.caption("**Meta**")
        for k in ("last_run", "last_date",
                  "last_model_version", "last_model_created_at"):
            v = load_meta(k)
            if v:
                st.caption(f"{k}: `{v}`")


if __name__ == "__main__":
    main()
