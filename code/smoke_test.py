"""Kurulum doğrulaması — tüm 10 coin için artifact yüklenir, son tarih sinyali üretilir."""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import inference

if __name__ == "__main__":
    inference.smoke_test()
