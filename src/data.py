# %%
from pathlib import Path
import pandas as pd
import yfinance as yf
import sys


PROJECT_ROOT = Path.cwd().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TICKERS,
    START_DATE,
    END_DATE,
)


def download_prices() -> pd.DataFrame:
    """
    Download adjusted daily prices for the portfolio assets.
    """

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    data = yf.download(
        TICKERS,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
    )

    prices = data["Close"]

    prices = prices.dropna(how="all")
    prices = prices.ffill()

    output_path = RAW_DATA_DIR / "prices.csv"
    prices.to_csv(output_path)

    print(f"Downloaded data saved to: {output_path}")
    print(f"Shape: {prices.shape}")

    return prices


if __name__ == "__main__":
    download_prices()
