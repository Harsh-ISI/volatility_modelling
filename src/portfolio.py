# %%
from pathlib import Path
import numpy as np
import pandas as pd
import sys


PROJECT_ROOT = Path.cwd().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.config import WEIGHTS

# %%
def calculate_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily logarithmic returns.
    """

    returns = np.log(prices / prices.shift(1))

    return returns.dropna()


def calculate_portfolio_returns(
    returns: pd.DataFrame,
    weights: dict,
) -> pd.Series:
    """
    Calculate fixed-weight portfolio log returns.
    """

    weight_series = pd.Series(weights)

    portfolio_returns = returns.mul(
        weight_series,
        axis=1,
    ).sum(axis=1)

    portfolio_returns.name = "portfolio"

    return portfolio_returns


if __name__ == "__main__":
    from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

    prices = pd.read_csv(
        RAW_DATA_DIR / "prices.csv",
        index_col=0,
        parse_dates=True,
    )

    returns = calculate_log_returns(prices)


    returns.to_csv(
        PROCESSED_DATA_DIR / "asset_returns.csv"
    )



    # Buy and Hold Portfolio Returns
    weights = {
        "AAPL": 0.20,
        "JPM": 0.20,
        "XOM": 0.20,
        "JNJ": 0.20,
        "WMT": 0.20,
    }

    normalized_prices = prices / prices.iloc[0]

    portfolio_value = normalized_prices.mul(
        pd.Series(weights),
        axis=1
    ).sum(axis=1)

    portfolio_value.name = "Portfolio"
    portfolio_returns = np.log(
        portfolio_value / portfolio_value.shift(1)
    ).dropna()

    portfolio_returns.name = "Portfolio Return"

    portfolio_returns.to_csv(
        PROCESSED_DATA_DIR / "portfolio_returns.csv"
    )
   
    print("\nPortfolio returns:")
    print(portfolio_returns.head())

    # for continuously rebalancing portfolio, needs more effort to maintain the weight balance
    # portfolio_returns = calculate_portfolio_returns(
    #     returns,
    #     WEIGHTS,
    # )
    # portfolio_returns.to_csv(
    #     PROCESSED_DATA_DIR / "xx_portfolio_returns.csv"
    # )