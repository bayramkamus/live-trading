"""SQLite DB katmanı — bulut deploy için kalıcı depolama.

Yerelde `data_live/app.db` dosyası. GitHub Actions günlük job sonrası
DB'yi commit'ler; HF Space repo'dan okur. Rolling 30-gün retention.

Tablolar:
    signals       — günlük model çıktıları (30 gün tut)
    trades        — tüm işlem logu (limitsiz — portföy geçmişi)
    equity        — günlük equity snapshot (limitsiz)
    positions     — açık pozisyon canlı snapshot (replace)
    ohlcv_cache   — coin günlük OHLCV (60 gün tut)
    tech_cache    — coin günlük 42-tech feature (60 gün tut)
    sent_cache    — coin günlük 29-sent feature (60 gün tut)
    macro_cache   — günlük 27 makro feature (60 gün tut)
    meta          — key-value: last_run, last_model_version, vs.

Tipik kullanım:
    from db import DB
    with DB() as db:
        db.init()
        db.write_signals(df)
        db.prune_old(days=30)
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from paths import DATA_LIVE


DB_PATH = DATA_LIVE / "app.db"


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS signals (
    date        TEXT NOT NULL,
    coin        TEXT NOT NULL,
    signal      TEXT NOT NULL,
    signal_int  INTEGER NOT NULL,
    p_sell      REAL,
    p_hold      REAL,
    p_buy       REAL,
    buy_th      REAL,
    sell_th     REAL,
    horizon     INTEGER,
    n_features  INTEGER,
    has_sent    INTEGER,
    has_macro   INTEGER,
    has_tech    INTEGER,
    PRIMARY KEY (date, coin)
);
CREATE INDEX IF NOT EXISTS idx_signals_date ON signals(date);

CREATE TABLE IF NOT EXISTS trades (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    date          TEXT NOT NULL,
    coin          TEXT NOT NULL,
    side          TEXT NOT NULL,
    qty           REAL NOT NULL,
    price         REAL NOT NULL,
    fee           REAL NOT NULL,
    gross         REAL NOT NULL,
    realized_pnl  REAL NOT NULL DEFAULT 0,
    note          TEXT
);
CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(date);
CREATE INDEX IF NOT EXISTS idx_trades_coin ON trades(coin);

CREATE TABLE IF NOT EXISTS equity (
    date             TEXT PRIMARY KEY,
    cash             REAL NOT NULL,
    positions_value  REAL NOT NULL,
    equity           REAL NOT NULL,
    n_open           INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS positions (
    coin       TEXT PRIMARY KEY,
    qty        REAL NOT NULL,
    avg_price  REAL NOT NULL,
    opened_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    k TEXT PRIMARY KEY,
    v TEXT
);

-- feature cache'leri generic: date + coin(null for macro) + payload (JSON)
CREATE TABLE IF NOT EXISTS ohlcv_cache (
    coin    TEXT NOT NULL,
    date    TEXT NOT NULL,
    open    REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (coin, date)
);
CREATE INDEX IF NOT EXISTS idx_ohlcv_date ON ohlcv_cache(date);

CREATE TABLE IF NOT EXISTS tech_cache (
    coin     TEXT NOT NULL,
    date     TEXT NOT NULL,
    payload  TEXT NOT NULL,   -- JSON: 42 kolon
    PRIMARY KEY (coin, date)
);
CREATE INDEX IF NOT EXISTS idx_tech_date ON tech_cache(date);

CREATE TABLE IF NOT EXISTS sent_cache (
    coin     TEXT NOT NULL,
    date     TEXT NOT NULL,
    payload  TEXT NOT NULL,   -- JSON: 29 kolon
    PRIMARY KEY (coin, date)
);
CREATE INDEX IF NOT EXISTS idx_sent_date ON sent_cache(date);

CREATE TABLE IF NOT EXISTS broker_decisions (
    date          TEXT NOT NULL,
    coin          TEXT NOT NULL,
    raw_signal    TEXT,        -- model'in ham sinyali (BUY/HOLD/SELL)
    final_action  TEXT,        -- gate'ler sonrasi (BUY/HOLD/SELL)
    reason        TEXT,        -- 'ok'|'blocked_*'|'missing_features:*'|'low_news_count*'|...
    raw_signal_int INTEGER,    -- ham sinyalin -1/0/+1 hali
    model_version TEXT,        -- artifact_hash (12 hex)
    PRIMARY KEY (date, coin)
);
CREATE INDEX IF NOT EXISTS idx_decisions_date ON broker_decisions(date);

CREATE TABLE IF NOT EXISTS macro_cache (
    date     TEXT PRIMARY KEY,
    payload  TEXT NOT NULL    -- JSON: 27 kolon
);
"""


class DB:
    def __init__(self, path: Path = DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn: Optional[sqlite3.Connection] = None

    # ------------- context --------------
    def __enter__(self) -> "DB":
        self.conn = sqlite3.connect(
            self.path,
            detect_types=sqlite3.PARSE_DECLTYPES,
            isolation_level=None,           # autocommit
            check_same_thread=False,
        )
        self.conn.row_factory = sqlite3.Row
        # WAL mode bazı FS'lerde (virtio-fs, NFS) desteklenmiyor; TRUNCATE fallback
        try:
            self.conn.execute("PRAGMA journal_mode=WAL;")
        except sqlite3.OperationalError:
            self.conn.execute("PRAGMA journal_mode=TRUNCATE;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA foreign_keys=ON;")
        return self

    def __exit__(self, *_a) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def init(self) -> None:
        assert self.conn is not None
        for stmt in SCHEMA_SQL.strip().split(";\n\n"):
            s = stmt.strip()
            if s:
                self.conn.executescript(s + ";")
        # A4: idempotent migration - signals tablosuna yeni kolonlar
        self._migrate_signals_columns()

    def _migrate_signals_columns(self) -> None:
        """Mevcut signals tablosuna eksik kolonlari ekle (A4)."""
        cur = self.conn.execute("PRAGMA table_info(signals)")
        existing = {row["name"] for row in cur.fetchall()}
        new_cols = [
            ("reason",            "TEXT"),
            ("model_version",     "TEXT"),
            ("model_created_at",  "TEXT"),
            ("feature_set",       "TEXT"),
            ("gate_dir_margin",   "REAL"),
            ("gate_hold_veto",    "REAL"),
            ("coin_tier",         "TEXT"),
            ("signal_date",       "TEXT"),
            ("fill_date",         "TEXT"),
        ]
        for name, typ in new_cols:
            if name not in existing:
                self.conn.execute(f"ALTER TABLE signals ADD COLUMN {name} {typ}")

    # ------------- meta --------------
    def get_meta(self, k: str, default: Optional[str] = None) -> Optional[str]:
        row = self.conn.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return row["v"] if row else default

    def set_meta(self, k: str, v: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(k,v) VALUES(?,?) "
            "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
            (k, v),
        )

    # ------------- signals --------------
    def write_signals(self, df: pd.DataFrame, as_of_date: str) -> int:
        """signals upsert. A4: reason, model_version, gate_*, signal/fill_date eklendi."""
        if df.empty:
            return 0
        cols = ["coin","signal","signal_int","p_sell","p_hold","p_buy",
                "buy_th","sell_th","horizon","n_features",
                "has_sent","has_macro","has_tech",
                "reason","model_version","model_created_at","feature_set",
                "gate_dir_margin","gate_hold_veto","coin_tier",
                "signal_date","fill_date"]
        for c in cols:
            if c not in df.columns:
                df[c] = None
        rows = []
        for _, r in df.iterrows():
            rows.append((
                as_of_date, str(r["coin"]),
                str(r.get("signal", "HOLD")),
                int(r["signal_int"]) if pd.notna(r.get("signal_int")) else 0,
                _f(r.get("p_sell")), _f(r.get("p_hold")), _f(r.get("p_buy")),
                _f(r.get("buy_th")), _f(r.get("sell_th")),
                _i(r.get("horizon")), _i(r.get("n_features")),
                _i(r.get("has_sent")), _i(r.get("has_macro")), _i(r.get("has_tech")),
                str(r["reason"])           if pd.notna(r.get("reason"))           else None,
                str(r["model_version"])    if pd.notna(r.get("model_version"))    else None,
                str(r["model_created_at"]) if pd.notna(r.get("model_created_at")) else None,
                str(r["feature_set"])      if pd.notna(r.get("feature_set"))      else None,
                _f(r.get("gate_dir_margin")), _f(r.get("gate_hold_veto")),
                str(r["coin_tier"])        if pd.notna(r.get("coin_tier"))        else None,
                str(r["signal_date"])      if pd.notna(r.get("signal_date"))      else None,
                str(r["fill_date"])        if pd.notna(r.get("fill_date"))        else None,
            ))
        self.conn.executemany(
            "INSERT INTO signals("
            "date,coin,signal,signal_int,p_sell,p_hold,p_buy,buy_th,sell_th,"
            "horizon,n_features,has_sent,has_macro,has_tech,"
            "reason,model_version,model_created_at,feature_set,"
            "gate_dir_margin,gate_hold_veto,coin_tier,signal_date,fill_date"
            ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(date,coin) DO UPDATE SET "
            "signal=excluded.signal,signal_int=excluded.signal_int,"
            "p_sell=excluded.p_sell,p_hold=excluded.p_hold,p_buy=excluded.p_buy,"
            "buy_th=excluded.buy_th,sell_th=excluded.sell_th,"
            "horizon=excluded.horizon,n_features=excluded.n_features,"
            "has_sent=excluded.has_sent,has_macro=excluded.has_macro,has_tech=excluded.has_tech,"
            "reason=excluded.reason,model_version=excluded.model_version,"
            "model_created_at=excluded.model_created_at,feature_set=excluded.feature_set,"
            "gate_dir_margin=excluded.gate_dir_margin,gate_hold_veto=excluded.gate_hold_veto,"
            "coin_tier=excluded.coin_tier,"
            "signal_date=excluded.signal_date,fill_date=excluded.fill_date",
            rows,
        )
        return len(rows)

    def write_decision(self, date: str, coin: str,
                       raw_signal: str, final_action: str, reason: str,
                       raw_signal_int: int = 0,
                       model_version: Optional[str] = None) -> None:
        """A4: orchestrate per-coin decision'i broker_decisions tablosuna upsert."""
        self.conn.execute(
            "INSERT INTO broker_decisions("
            "date,coin,raw_signal,final_action,reason,raw_signal_int,model_version"
            ") VALUES(?,?,?,?,?,?,?) "
            "ON CONFLICT(date,coin) DO UPDATE SET "
            "raw_signal=excluded.raw_signal,final_action=excluded.final_action,"
            "reason=excluded.reason,raw_signal_int=excluded.raw_signal_int,"
            "model_version=excluded.model_version",
            (date, coin, raw_signal, final_action, reason, int(raw_signal_int),
             model_version),
        )

    def read_decisions(self, days: int = 30) -> pd.DataFrame:
        cutoff = (datetime.utcnow().date() - timedelta(days=days)).isoformat()
        return pd.read_sql_query(
            "SELECT * FROM broker_decisions WHERE date>=? "
            "ORDER BY date DESC, coin ASC",
            self.conn, params=(cutoff,),
        )

    def read_signals(self, days: int = 30) -> pd.DataFrame:
        cutoff = (datetime.utcnow().date() - timedelta(days=days)).isoformat()
        return pd.read_sql_query(
            "SELECT * FROM signals WHERE date>=? ORDER BY date DESC, coin ASC",
            self.conn, params=(cutoff,),
        )

    def read_signals_for_date(self, date: str) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM signals WHERE date=? ORDER BY coin", self.conn, params=(date,)
        )

    # ------------- trades / equity / positions --------------
    def append_trade(self, t: dict) -> None:
        self.conn.execute(
            "INSERT INTO trades(date,coin,side,qty,price,fee,gross,realized_pnl,note) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (t["date"], t["coin"], t["side"], _f(t["qty"]), _f(t["price"]),
             _f(t["fee"]), _f(t["gross"]), _f(t.get("realized_pnl", 0)),
             t.get("note", "")),
        )

    def append_equity(self, date: str, cash: float, positions_value: float,
                      equity: float, n_open: int) -> None:
        self.conn.execute(
            "INSERT INTO equity(date,cash,positions_value,equity,n_open) VALUES(?,?,?,?,?) "
            "ON CONFLICT(date) DO UPDATE SET cash=excluded.cash,"
            "positions_value=excluded.positions_value,equity=excluded.equity,n_open=excluded.n_open",
            (date, _f(cash), _f(positions_value), _f(equity), int(n_open)),
        )

    def replace_positions(self, positions: Iterable[dict]) -> None:
        self.conn.execute("DELETE FROM positions")
        rows = [(p["coin"], _f(p["qty"]), _f(p["avg_price"]), p["opened_at"])
                for p in positions]
        if rows:
            self.conn.executemany(
                "INSERT INTO positions(coin,qty,avg_price,opened_at) VALUES(?,?,?,?)",
                rows,
            )

    def read_trades(self, days: Optional[int] = None,
                    coin: Optional[str] = None) -> pd.DataFrame:
        q = "SELECT * FROM trades"
        where, params = [], []
        if days is not None:
            cutoff = (datetime.utcnow().date() - timedelta(days=days)).isoformat()
            where.append("date>=?"); params.append(cutoff)
        if coin:
            where.append("coin=?"); params.append(coin)
        if where:
            q += " WHERE " + " AND ".join(where)
        q += " ORDER BY id DESC"
        return pd.read_sql_query(q, self.conn, params=tuple(params))

    def read_equity(self, days: Optional[int] = None) -> pd.DataFrame:
        q = "SELECT * FROM equity"
        params = ()
        if days is not None:
            cutoff = (datetime.utcnow().date() - timedelta(days=days)).isoformat()
            q += " WHERE date>=?"
            params = (cutoff,)
        q += " ORDER BY date ASC"
        return pd.read_sql_query(q, self.conn, params=params)

    def read_positions(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM positions ORDER BY coin", self.conn)

    # ------------- cache'ler --------------
    def write_ohlcv(self, coin: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        need = {"date", "open", "high", "low", "close", "volume"}
        if not need.issubset(df.columns):
            return
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
        rows = [(coin, r["date"], _f(r["open"]), _f(r["high"]),
                 _f(r["low"]), _f(r["close"]), _f(r["volume"]))
                for _, r in d.iterrows()]
        self.conn.executemany(
            "INSERT INTO ohlcv_cache(coin,date,open,high,low,close,volume) "
            "VALUES(?,?,?,?,?,?,?) "
            "ON CONFLICT(coin,date) DO UPDATE SET "
            "open=excluded.open,high=excluded.high,low=excluded.low,"
            "close=excluded.close,volume=excluded.volume",
            rows,
        )

    def write_features(self, table: str, coin: Optional[str], df: pd.DataFrame) -> None:
        """table ∈ {tech_cache, sent_cache, macro_cache}. Macro'da coin=None."""
        if df.empty or table not in ("tech_cache", "sent_cache", "macro_cache"):
            return
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
        feat_cols = [c for c in d.columns if c not in ("date", "coin")]
        rows = []
        for _, r in d.iterrows():
            payload = json.dumps({c: _jsonable(r[c]) for c in feat_cols})
            if table == "macro_cache":
                rows.append((r["date"], payload))
            else:
                rows.append((coin or r.get("coin", ""), r["date"], payload))
        if table == "macro_cache":
            self.conn.executemany(
                "INSERT INTO macro_cache(date,payload) VALUES(?,?) "
                "ON CONFLICT(date) DO UPDATE SET payload=excluded.payload",
                rows,
            )
        else:
            self.conn.executemany(
                f"INSERT INTO {table}(coin,date,payload) VALUES(?,?,?) "
                f"ON CONFLICT(coin,date) DO UPDATE SET payload=excluded.payload",
                rows,
            )

    def read_last_close(self, coin: str, date: str) -> Optional[float]:
        row = self.conn.execute(
            "SELECT close FROM ohlcv_cache WHERE coin=? AND date<=? "
            "ORDER BY date DESC LIMIT 1",
            (coin, date),
        ).fetchone()
        return float(row["close"]) if row else None

    # ------------- retention --------------
    def prune_old(self, days: int = 30, cache_days: int = 60) -> dict:
        """Rolling retention:
            signals           → son `days` gün
            ohlcv/tech/sent/macro cache → son `cache_days` gün
            trades, equity    → dokunulmaz (geçmiş önemli)
        """
        today = datetime.utcnow().date()
        sig_cut   = (today - timedelta(days=days)).isoformat()
        cache_cut = (today - timedelta(days=cache_days)).isoformat()
        counts = {}
        counts["signals"]     = self.conn.execute(
            "DELETE FROM signals WHERE date<?", (sig_cut,)).rowcount
        for tbl in ("ohlcv_cache", "tech_cache", "sent_cache", "macro_cache"):
            counts[tbl] = self.conn.execute(
                f"DELETE FROM {tbl} WHERE date<?", (cache_cut,)).rowcount
        self.conn.execute("VACUUM")
        return counts


# -------- helpers --------

def _f(x) -> Optional[float]:
    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def _i(x) -> Optional[int]:
    v = _f(x)
    return int(v) if v is not None else None


def _jsonable(x):
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            return float(x)
    return x


# -------- CLI --------

if __name__ == "__main__":
    import sys
    with DB() as db:
        if len(sys.argv) > 1 and sys.argv[1] == "init":
            db.init()
            print(f"init → {DB_PATH}")
        elif len(sys.argv) > 1 and sys.argv[1] == "prune":
            db.init()
            print(db.prune_old())
        else:
            db.init()
            tables = [r["name"] for r in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
            print(f"DB: {DB_PATH}")
            print("tablolar:", tables)
            for t in tables:
                n = db.conn.execute(f"SELECT COUNT(*) c FROM {t}").fetchone()["c"]
                print(f"  {t}: {n} satır")
