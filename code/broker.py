"""Broker abstraction — gerçek ve kağıt (paper) broker implementasyonları
arkada aynı arayüzü paylaşır.

Gerçek broker entegrasyonu (Binance/Coinbase/Alpaca) burada oluşturulacak
başka sınıflarla eklenir (ör. BinanceBroker(BrokerAdapter)).  Şu an
yalnızca abstract + veri tipleri var.

Pozitif iş akışı:
    adapter = PaperBroker.load_or_init()
    signals = {"BTC": +1, "ETH": -1, ...}
    prices  = {"BTC": 67000.0, ...}
    execute(adapter, signals, prices, as_of_date)
    adapter.save()
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional


# ============================================================
# Veri tipleri
# ============================================================

@dataclass
class Position:
    coin: str
    qty:  float            # + long, - short (USD cinsinden değil coin cinsinden)
    avg_price: float       # son ortalama giriş fiyatı
    opened_at: str         # ISO date

    def side(self) -> str:
        if self.qty > 0: return "LONG"
        if self.qty < 0: return "SHORT"
        return "FLAT"

    def market_value(self, price: float) -> float:
        return self.qty * price

    def unrealized_pnl(self, price: float) -> float:
        return (price - self.avg_price) * self.qty

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        return cls(**d)


@dataclass
class Trade:
    date: str            # ISO date
    coin: str
    side: str            # BUY_OPEN / SELL_CLOSE / SELL_OPEN / BUY_COVER
    qty: float           # coin adedi (pozitif)
    price: float
    fee: float
    gross: float         # qty * price (fee hariç)
    realized_pnl: float  # kapanışta dolayı değiştiyse; açılışta 0
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# Abstract adapter
# ============================================================

class BrokerAdapter(abc.ABC):
    """Minimal arayüz: pozisyonlar, nakit, emir yerleştirme, serialize."""

    # ---- state erişimi ----
    @abc.abstractmethod
    def get_cash(self) -> float: ...

    @abc.abstractmethod
    def get_positions(self) -> Dict[str, Position]: ...

    @abc.abstractmethod
    def get_equity(self, mark_prices: Dict[str, float]) -> float:
        """cash + sum(pos.qty * mark_price)"""

    # ---- emir ----
    @abc.abstractmethod
    def open_position(self, coin: str, side: str, notional_usd: float,
                      price: float, date: str) -> Trade: ...

    @abc.abstractmethod
    def close_position(self, coin: str, price: float, date: str) -> Optional[Trade]: ...

    # ---- persistence ----
    @abc.abstractmethod
    def save(self) -> None: ...

    @classmethod
    @abc.abstractmethod
    def load_or_init(cls, *args, **kwargs) -> "BrokerAdapter": ...
