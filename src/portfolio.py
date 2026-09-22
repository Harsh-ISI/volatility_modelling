# %%
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, WEIGHTS, TRAIN_START, TRAIN_END, VALIDATION_END,  VALIDATION_START, TEST_END, TEST_START

prices = pd.read_csv(
    RAW_DATA_DIR / "prices.csv",
    index_col=0,
    parse_dates=True,
)
asset_returns = np.log(prices / prices.shift(1)).dropna()
asset_returns.name = "Asset Returns"
asset_returns.to_csv(PROCESSED_DATA_DIR / "asset_returns.csv")

# Buy and Hold Portfolio Returns

normalized_prices = prices / prices.iloc[0]

portfolio_value = normalized_prices.mul(
    pd.Series(WEIGHTS),
    axis=1
).sum(axis=1)

portfolio_value.name = "Portfolio"
portfolio_returns = np.log(portfolio_value / portfolio_value.shift(1)).dropna()

portfolio_returns.name = "Portfolio Return"

portfolio_returns.to_csv(PROCESSED_DATA_DIR / "portfolio_returns.csv")

# for continuously rebalancing portfolio, needs more effort to maintain the weight balance


train = portfolio_returns[
    (portfolio_returns.index >= TRAIN_START) &
    (portfolio_returns.index <= TRAIN_END)
]
train.name = "TRAIN"
train.to_csv(PROCESSED_DATA_DIR / "train.csv")

validation = portfolio_returns[
    (portfolio_returns.index >= VALIDATION_START) &
    (portfolio_returns.index <= VALIDATION_END)
]
validation.name = "TRAIN"
validation.to_csv(PROCESSED_DATA_DIR / "VALIDATION.csv")

test = portfolio_returns[
    (portfolio_returns.index >= TEST_START) &
    (portfolio_returns.index <= TEST_END)
]
test.name = "TEST"
test.to_csv(PROCESSED_DATA_DIR / "test.csv")

# %%
