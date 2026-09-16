from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Output directories
FIGURES_DIR = PROJECT_ROOT / "output" / "figures"
TABLES_DIR = PROJECT_ROOT / "output" / "tables"
MODELS_DIR = PROJECT_ROOT / "output" / "models"

# Portfolio
TICKERS = [
    "AAPL",
    "JPM",
    "XOM",
    "JNJ",
    "WMT",
]

WEIGHTS = {
    "AAPL": 0.20,
    "JPM": 0.20,
    "XOM": 0.20,
    "JNJ": 0.20,
    "WMT": 0.20,
}

# Sample period
START_DATE = "2015-01-01"
END_DATE = "2026-01-01"

