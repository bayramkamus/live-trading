"""PaperBroker — JSON-destekli kağıt broker.

State dosyaları (varsayılan: live_trading/data_live/broker/):
    state.json    — {cash, positions, last_updated}
    trades.csv    — tüm tarihi işlem logu (tarih, coin, side, qty, price, fee, pnl)
    equity.csv    — günlük toplam equity snapshot

Sabitler:
    BASE_EQUITY          = 10_000 USD başlangıç kasa
    FEE_BPS              = 10 bps (0.10% per fill)
    MAX_POSITIONS        = 5 (aynı anda açık tutulacak max pozisyon sayısı)
    RISK_PCT_PER_TRADE   = 0.10 (her emir için equity %10'u)
    ALLOW_SHORTS         = False (long-only, V4 default)

Fiyat kaynağı: son close (ohlcv_fetcher zaten günlük kline yazıyor);
bu modül sadece dışarıdan verilen fiyatları kullanır.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from paths import DATA_LIVE
from broker import BrokerAdapter, Position, Trade


# ============================================================
# Parametreler
# ============================================================

BROKER_DIR = DATA_LIVE / "broker"
STATE_FILE  = BROKER_DIR / "state.json"
TRADES_FILE = BROKER_DIR / "trades.csv"
EQUITY_FILE = BROKER_DIR / "equity.csv"

BASE_EQUITY        = 10_000.0
FEE_BPS            = 10.0          # 0.10%
RISK_PCT_PER_TRADE = 0.10          # equity'nin %10'u
MAX_POSITIONS      = 5
ALLOW_SHORTS       = False

TRADE_HEADER  = ["date", "coin", "side", "qty", "price", "fee",
                 "gross", "realized_pnl", "note"]
EQUITY_HEADER = ["date", "cash", "positions_value", "equity", "n_open"]


# ============================================================
# PaperBroker
# ============================================================

class PaperBroker(BrokerAdapter):

    def __init__(self, cash: float, positions: Dict[str, Position],
                 state_file: Path = STATE_FILE,
                 trades_file: Path = TRADES_FILE,
                 equity_file: Path = EQUITY_FILE,
                 fee_bps: float = FEE_BPS,
                 risk_pct: float = RISK_PCT_PER_TRADE,
                 max_positions: int = MAX_POSITIONS,
                 allow_shorts: bool = ALLOW_SHORTS):
        self.cash = float(cash)
        self.positions: Dict[str, Position] = dict(positions)
        self.state_file = Path(state_file)
        self.trades_file = Path(trades_file)
        self.equity_file = Path(equity_file)
        self.fee_bps = fee_bps
        self.risk_pct = risk_pct
        self.max_positions = max_positions
        self.allow_shorts = allow_shorts
        self._ensure_files()

    # ---------- state / files ----------

    def _ensure_files(self) -> None:
        BROKER_DIR.mkdir(parents=True, exist_ok=True)
        if not self.trades_file.exists():
            with self.trades_file.open("w", newline="") as f:
                csv.writer(f).writerow(TRADE_HEADER)
        if not self.equity_file.exists():
            with self.equity_file.open("w", newline="") as f:
                csv.writer(f).writerow(EQUITY_HEADER)

    @classmethod
    def load_or_init(cls, state_file: Path = STATE_FILE,
                     base_equity: float = BASE_EQUITY) -> "PaperBroker":
        if state_file.exists():
            js = json.loads(state_file.read_text())
            cash = float(js.get("cash", base_equity))
            positions = {c: Position.from_dict(d)
                         for c, d in js.get("positions", {}).items()}
            pb = cls(cash=cash, positions=positions, state_file=state_file)
            return pb
        return cls(cash=base_equity, positions={}, state_file=state_file)

    def save(self) -> None:
        BROKER_DIR.mkdir(parents=True, exist_ok=True)
        state = {
            "cash": self.cash,
            "positions": {c: p.to_dict() for c, p in self.positions.items()},
            "last_updated": datetime.utcnow().isoformat(),
        }
        self.state_file.write_text(json.dumps(state, indent=2, default=str))

    def _append_trade(self, t: Trade) -> None:
        with self.trades_file.open("a", newline="") as f:
            csv.writer(f).writerow([t.date, t.coin, t.side, t.qty, t.price,
                                    t.fee, t.gross, t.realized_pnl, t.note])

    def _append_equity(self, date: str, mark_prices: Dict[str, float]) -> None:
        pos_val = sum(p.qty * mark_prices.get(p.coin, p.avg_price)
                      for p in self.positions.values())
        eq = self.cash + pos_val
        with self.equity_file.open("a", newline="") as f:
            csv.writer(f).writerow([date, round(self.cash, 2),
                                    round(pos_val, 2), round(eq, 2),
                                    len(self.positions)])

    # ---------- getters ----------

    def get_cash(self) -> float:
        return self.cash

    def get_positions(self) -> Dict[str, Position]:
        return dict(self.positions)

    def get_equity(self, mark_prices: Dict[str, float]) -> float:
        pos_val = sum(p.qty * mark_prices.get(p.coin, p.avg_price)
                      for p in self.positions.values())
        return self.cash + pos_val

    # ---------- sizing ----------

    def _notional_per_trade(self, mark_prices: Dict[str, float]) -> float:
        equity = self.get_equity(mark_prices)
        return max(0.0, equity * self.risk_pct)

    def _apply_fee(self, gross: float) -> float:
        return abs(gross) * (self.fee_bps / 10_000.0)

    # ---------- emir ----------

    def open_position(self, coin: str, side: str, notional_usd: float,
                      price: float, date: str) -> Trade:
        """side: 'LONG' veya 'SHORT'. Var olan pozisyon varsa ÜZERİNE EKLEME YAPMAZ — önce close()."""
        assert side in ("LONG", "SHORT"), f"geçersiz side: {side}"
        if coin in self.positions:
            raise RuntimeError(f"{coin} için zaten açık pozisyon var, önce close_position()")

        if side == "SHORT" and not self.allow_shorts:
            raise RuntimeError(f"Short kapalı — {coin} SHORT emir reddedildi")

        if len(self.positions) >= self.max_positions:
            raise RuntimeError(f"max_positions={self.max_positions} dolu")

        if notional_usd <= 0 or price <= 0:
            raise ValueError(f"geçersiz notional/price: {notional_usd} @ {price}")

        qty_sign = 1.0 if side == "LONG" else -1.0
        qty = qty_sign * (notional_usd / price)
        gross = qty * price
        fee = self._apply_fee(gross)

        # LONG: cash -= notional + fee
        # SHORT: cash += notional - fee  (satış karşılığı kasaya girer)
        if side == "LONG":
            cost = notional_usd + fee
            if cost > self.cash:
                raise RuntimeError(
                    f"nakit yetersiz: {self.cash:.2f} < {cost:.2f} ({coin} {side})"
                )
            self.cash -= cost
        else:  # SHORT
            self.cash += notional_usd - fee

        self.positions[coin] = Position(
            coin=coin, qty=qty, avg_price=price, opened_at=date,
        )
        t = Trade(
            date=date, coin=coin,
            side="BUY_OPEN" if side == "LONG" else "SELL_OPEN",
            qty=abs(qty), price=price, fee=round(fee, 4),
            gross=round(gross, 2), realized_pnl=0.0,
            note=f"open {side}",
        )
        self._append_trade(t)
        return t

    def close_position(self, coin: str, price: float, date: str) -> Optional[Trade]:
        if coin not in self.positions:
            return None
        pos = self.positions.pop(coin)
        gross = pos.qty * price
        fee = self._apply_fee(gross)

        realized = (price - pos.avg_price) * pos.qty
        if pos.qty > 0:
            # LONG close: satış, cash += gross - fee
            self.cash += gross - fee
            side = "SELL_CLOSE"
        else:
            # SHORT close: alış, cash -= |gross| + fee
            self.cash -= abs(gross) + fee
            side = "BUY_COVER"

        t = Trade(
            date=date, coin=coin, side=side,
            qty=abs(pos.qty), price=price,
            fee=round(fee, 4), gross=round(gross, 2),
            realized_pnl=round(realized - fee, 2),
            note=f"close {pos.side()}",
        )
        self._append_trade(t)
        return t

    # ---------- policy runner ----------

    def step(self, signals: Dict[str, int], prices: Dict[str, float],
             date: str) -> list[Trade]:
        """Signals → emirler. Basit long-only politika:

            signal == +1  → FLAT ise LONG aç
            signal == -1  → LONG ise kapat (short kapalıysa yeni SHORT açma)
            signal == 0   → değişiklik yok
        """
        trades: list[Trade] = []

        # 1) kapatılacaklar önce işle → sermaye yeniden kullanılabilsin
        for coin, sig in signals.items():
            if coin not in self.positions:
                continue
            pos = self.positions[coin]
            price = prices.get(coin)
            if price is None:
                continue
            close_it = False
            if pos.qty > 0 and sig == -1:
                close_it = True
            elif pos.qty > 0 and sig == 0:
                # HOLD: long'u koru
                close_it = False
            elif pos.qty < 0 and sig == +1:
                close_it = True
            if close_it:
                t = self.close_position(coin, price, date)
                if t is not None:
                    trades.append(t)

        # 2) açılacaklar
        for coin, sig in signals.items():
            if coin in self.positions:
                continue
            price = prices.get(coin)
            if price is None or price <= 0:
                continue
            if len(self.positions) >= self.max_positions:
                break
            notional = self._notional_per_trade(prices)
            if notional <= 0 or notional > self.cash:
                continue
            if sig == +1:
                t = self.open_position(coin, "LONG", notional, price, date)
                trades.append(t)
            elif sig == -1 and self.allow_shorts:
                t = self.open_position(coin, "SHORT", notional, price, date)
                trades.append(t)

        # equity snapshot
        self._append_equity(date, prices)
        return trades

    # ---------- raporlama ----------

    def summary(self, mark_prices: Dict[str, float]) -> dict:
        eq = self.get_equity(mark_prices)
        rows = []
        for p in self.positions.values():
            mp = mark_prices.get(p.coin, p.avg_price)
            rows.append({
                "coin": p.coin,
                "side": p.side(),
                "qty": round(p.qty, 6),
                "avg_price": round(p.avg_price, 4),
                "mark": round(mp, 4),
                "unreal_pnl": round(p.unrealized_pnl(mp), 2),
            })
        return {
            "cash": round(self.cash, 2),
            "equity": round(eq, 2),
            "n_open": len(self.positions),
            "open": rows,
        }


# ============================================================
# CLI — manuel test
# ============================================================

def _cli_demo() -> None:
    pb = PaperBroker.load_or_init()
    print("başlangıç:", pb.summary({}))

    # sahte bir gün
    prices = {"BTC": 67000.0, "ETH": 3300.0, "SOL": 170.0}
    signals = {"BTC": +1, "ETH": +1, "SOL": 0}
    trades = pb.step(signals, prices, date="2026-04-23")
    for t in trades:
        print(" ", t.date, t.coin, t.side, t.qty, "@", t.price)
    print("gün sonu:", pb.summary(prices))

    pb.save()


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if args and args[0] == "reset":
        if STATE_FILE.exists():
            STATE_FILE.unlink()
        if TRADES_FILE.exists():
            TRADES_FILE.unlink()
        if EQUITY_FILE.exists():
            EQUITY_FILE.unlink()
        print("broker state silindi")
    elif args and args[0] == "demo":
        _cli_demo()
    else:
        pb = PaperBroker.load_or_init()
        print(json.dumps(pb.summary({}), indent=2))
