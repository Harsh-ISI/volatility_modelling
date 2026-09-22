import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm, t as student_t

from src.statistical_tests import (
    kupiec_test,
    christoffersen_independence_test,
    christoffersen_conditional_coverage_test
)


def walk_forward_garch_1d(
    portfolio_returns: pd.DataFrame,
    forecast_start: str,
    forecast_end: str,
    p: int = 1,
    q: int = 1,
    dist: str = "t",
    mean_model: str = "Constant",
) -> tuple[dict, pd.DataFrame]:
    """
    Expanding-window 1-day ahead GARCH forecasting and risk backtesting.

    Parameters:
        portfolio_returns (pd.DataFrame): Date-indexed DataFrame with one log_returns column.
        forecast_start (str): Start date of forecast window ('YYYY-MM-DD').
        forecast_end (str): End date of forecast window ('YYYY-MM-DD').
        p (int): GARCH volatility lag order.
        q (int): ARCH lag order.
        dist (str): Distribution ('t' or 'normal').
        mean_model (str): Conditional mean ('Constant', 'Zero', 'AR').

    Returns:
        tuple[dict, pd.DataFrame]: (summary_metrics, time_series_df)
    """
    # Extract series regardless of column name
    if isinstance(portfolio_returns, pd.DataFrame):
        returns_series = portfolio_returns.iloc[:, 0]
    else:
        returns_series = portfolio_returns
    val_dates = returns_series.loc[forecast_start:forecast_end].index

    results = []

    for date in val_dates:
        # 1. Historical returns prior to evaluation date 'date'
        train_slice = returns_series.loc[returns_series.index < date]

        # Multiply by 100 for numerical stability during MLE optimization
        am = arch_model(
            train_slice * 100,
            p=p,
            q=q,
            mean=mean_model,
            vol="GARCH",
            dist=dist,
        )
        res = am.fit(disp="off", show_warning=False)

        # 2. Extract 1-step forecast for day t
        fc = res.forecast(horizon=1, reindex=False)
        mu_hat = fc.mean.iloc[-1, 0] / 100.0
        var_hat = fc.variance.iloc[-1, 0] / 10000.0
        vol_hat = np.sqrt(var_hat)

        # 3. Calculate VaR and ES multipliers
        multipliers = {}
        for alpha in [0.01, 0.05]:
            if dist.lower() in ["t", "studentst", "students_t"]:
                nu = res.params.get("nu", 6.0)
                nu = max(nu, 2.01)  # Bound to avoid division by zero if nu <= 2
                x_a = student_t.ppf(1 - alpha, df=nu)
                adj = np.sqrt((nu - 2) / nu)

                var_mult = x_a * adj
                es_mult = (
                    (student_t.pdf(x_a, df=nu) / alpha)
                    * ((nu + x_a**2) / (nu - 1))
                    * adj
                )
            else:  # Normal distribution
                z_a = norm.ppf(1 - alpha)
                var_mult = z_a
                es_mult = norm.pdf(z_a) / alpha

            multipliers[alpha] = (var_mult, es_mult)

        # 4. Compute VaR and ES thresholds
        var_1pct = -mu_hat + (vol_hat * multipliers[0.01][0])
        es_1pct = -mu_hat + (vol_hat * multipliers[0.01][1])

        var_5pct = -mu_hat + (vol_hat * multipliers[0.05][0])
        es_5pct = -mu_hat + (vol_hat * multipliers[0.05][1])

        realized_return = returns_series.loc[date]

        results.append({
            "Date": date,
            "Realized_Return": realized_return,
            "Forecast_Vol": vol_hat,
            "VaR_1pct": var_1pct,
            "VaR_5pct": var_5pct,
            "ES_1pct": es_1pct,
            "ES_5pct": es_5pct,
            "Breach_1pct": realized_return < -var_1pct,
            "Breach_5pct": realized_return < -var_5pct,
        })

    # Assemble time-series DataFrame
    time_series_df = pd.DataFrame(results).set_index("Date")

    # 5. Compute Forecast Accuracy Metrics
    realized_var = time_series_df["Realized_Return"] ** 2
    forecast_var = time_series_df["Forecast_Vol"] ** 2

    mse = np.mean((forecast_var - realized_var) ** 2)
    mae = np.mean(np.abs(time_series_df["Forecast_Vol"] - np.abs(time_series_df["Realized_Return"])))
    qlike = np.mean((realized_var / forecast_var) - np.log(realized_var / forecast_var) - 1)

    summary_metrics = {
        "MSE": float(mse),
        "MAE": float(mae),
        "QLIKE": float(qlike),
    }

    # 6. Compute Risk & Backtesting Metrics for 1% and 5%
    for alpha_label, alpha_val in [("1pct", 0.01), ("5pct", 0.05)]:
        breaches = time_series_df[f"Breach_{alpha_label}"]
        breach_count = int(breaches.sum())
        breach_rate = float(breaches.mean())

        kupiec_res = kupiec_test(breaches, alpha_val)
        ind_res = christoffersen_independence_test(breaches)
        cc_res = christoffersen_conditional_coverage_test(breaches, alpha_val)

        exceedance_returns = time_series_df.loc[breaches, "Realized_Return"]
        avg_loss = float(exceedance_returns.mean()) if breach_count > 0 else np.nan

        summary_metrics.update({
            f"VaR_{alpha_label}_mean": float(time_series_df[f"VaR_{alpha_label}"].mean()),
            f"ES_{alpha_label}_mean": float(time_series_df[f"ES_{alpha_label}"].mean()),
            f"Breach_Count_{alpha_label}": breach_count,
            f"Breach_Rate_{alpha_label}": breach_rate,
            f"Kupiec_pvalue_{alpha_label}": kupiec_res["p_value"],
            f"Christoffersen_Indep_pvalue_{alpha_label}": ind_res["p_value"],
            f"Christoffersen_CondCov_pvalue_{alpha_label}": cc_res["p_value"],
            f"Avg_Exceedance_Loss_{alpha_label}": avg_loss,
        })

    return summary_metrics, time_series_df


def walk_forward_garch_10d(
    portfolio_returns: pd.DataFrame | pd.Series,
    forecast_start: str,
    forecast_end: str,
    p: int = 1,
    q: int = 1,
    dist: str = "t",
    mean_model: str = "Constant",
    overlap: bool = True,
    scaling_method: str = "sqrt_time",
) -> tuple[dict, pd.DataFrame]:
    """
    Expanding-window 10-day ahead GARCH forecasting and risk backtesting.

    Parameters:
        portfolio_returns (pd.DataFrame | pd.Series): Daily log returns series or single-column DataFrame.
        forecast_start (str): Start date of forecast window ('YYYY-MM-DD').
        forecast_end (str): End date of forecast window ('YYYY-MM-DD').
        p (int): GARCH volatility lag order.
        q (int): ARCH lag order.
        dist (str): Innovation distribution ('t' or 'normal').
        mean_model (str): Conditional mean specification ('Constant', 'Zero', 'AR').
        overlap (bool):
            - If True: Step by 1 day at each iteration (rolling 10-day cumulative window).
            - If False: Step by 10 days at each iteration (non-overlapping periods).
        scaling_method (str):
            - 'sqrt_time': Scale 1-day vol forecast by sqrt(10).
            - 'multistep': Analytical GARCH multi-step ahead 10-day variance sum.

    Returns:
        tuple[dict, pd.DataFrame]: (summary_metrics, time_series_df)
    """
    # 1. Normalize input to pandas Series
    if isinstance(portfolio_returns, pd.DataFrame):
        returns_series = portfolio_returns.iloc[:, 0]
    else:
        returns_series = portfolio_returns

    # 2. Determine evaluation dates
    val_dates = returns_series.loc[forecast_start:forecast_end].index
    if not overlap:
        val_dates = val_dates[::10]

    results = []

    for date in val_dates:
        idx_pos = returns_series.index.get_loc(date)

        # Ensure we have a full 10-day horizon forward to evaluate
        future_returns = returns_series.iloc[idx_pos : idx_pos + 10]
        if len(future_returns) < 10:
            break

        # Cumulative 10-day realized log return
        realized_10d_return = future_returns.sum()

        # Historical training slice prior to 'date'
        train_slice = returns_series.iloc[:idx_pos]

        # Rescale returns (* 100) for MLE numerical stability
        am = arch_model(
            train_slice * 100,
            p=p,
            q=q,
            mean=mean_model,
            vol="GARCH",
            dist=dist,
        )
        res = am.fit(disp="off", show_warning=False)

        # 3. Compute 10-Day Volatility and Mean Forecasts
        if scaling_method == "sqrt_time":
            fc = res.forecast(horizon=1, reindex=False)
            mu_1d = fc.mean.iloc[-1, 0] / 100.0
            var_1d = fc.variance.iloc[-1, 0] / 10000.0

            mu_10d = mu_1d * 10.0
            vol_10d = np.sqrt(var_1d * 10.0)
        elif scaling_method == "multistep":
            fc = res.forecast(horizon=10, reindex=False)
            mu_10d = fc.mean.iloc[-1].sum() / 100.0
            var_10d = fc.variance.iloc[-1].sum() / 10000.0
            vol_10d = np.sqrt(var_10d)
        else:
            raise ValueError(f"Unknown scaling_method: {scaling_method}")

        # 4. Tail Risk Multipliers
        multipliers = {}
        for alpha in [0.01, 0.05]:
            if dist.lower() in ["t", "studentst", "students_t"]:
                nu = res.params.get("nu", 6.0)
                nu = max(nu, 2.01)
                x_a = student_t.ppf(1 - alpha, df=nu)
                adj = np.sqrt((nu - 2) / nu)

                var_mult = x_a * adj
                es_mult = (
                    (student_t.pdf(x_a, df=nu) / alpha)
                    * ((nu + x_a**2) / (nu - 1))
                    * adj
                )
            else:  # Normal distribution
                z_a = norm.ppf(1 - alpha)
                var_mult = z_a
                es_mult = norm.pdf(z_a) / alpha

            multipliers[alpha] = (var_mult, es_mult)

        # 5. Compute 10-Day VaR and ES Thresholds
        var_1pct = -mu_10d + (vol_10d * multipliers[0.01][0])
        es_1pct = -mu_10d + (vol_10d * multipliers[0.01][1])

        var_5pct = -mu_10d + (vol_10d * multipliers[0.05][0])
        es_5pct = -mu_10d + (vol_10d * multipliers[0.05][1])

        results.append({
            "Date": date,
            "Realized_Return": realized_10d_return,
            "Forecast_Vol": vol_10d,
            "VaR_1pct": var_1pct,
            "VaR_5pct": var_5pct,
            "ES_1pct": es_1pct,
            "ES_5pct": es_5pct,
            "Breach_1pct": realized_10d_return < -var_1pct,
            "Breach_5pct": realized_10d_return < -var_5pct,
        })

    # Assemble time-series DataFrame
    time_series_df = pd.DataFrame(results).set_index("Date")

    # 6. Accuracy Metrics
    realized_var_10d = time_series_df["Realized_Return"] ** 2
    forecast_var_10d = time_series_df["Forecast_Vol"] ** 2

    mse = np.mean((forecast_var_10d - realized_var_10d) ** 2)
    mae = np.mean(np.abs(time_series_df["Forecast_Vol"] - np.abs(time_series_df["Realized_Return"])))
    qlike = np.mean((realized_var_10d / forecast_var_10d) - np.log(realized_var_10d / forecast_var_10d) - 1)

    summary_metrics = {
        "MSE": float(mse),
        "MAE": float(mae),
        "QLIKE": float(qlike),
    }

    # 7. Risk & Backtesting Metrics for 1% and 5%
    for alpha_label, alpha_val in [("1pct", 0.01), ("5pct", 0.05)]:
        breaches = time_series_df[f"Breach_{alpha_label}"]
        breach_count = int(breaches.sum())
        breach_rate = float(breaches.mean())

        kupiec_res = kupiec_test(breaches, alpha_val)
        ind_res = christoffersen_independence_test(breaches)
        cc_res = christoffersen_conditional_coverage_test(breaches, alpha_val)

        exceedance_returns = time_series_df.loc[breaches, "Realized_Return"]
        avg_loss = float(exceedance_returns.mean()) if breach_count > 0 else np.nan

        summary_metrics.update({
            f"VaR_{alpha_label}_mean": float(time_series_df[f"VaR_{alpha_label}"].mean()),
            f"ES_{alpha_label}_mean": float(time_series_df[f"ES_{alpha_label}"].mean()),
            f"Breach_Count_{alpha_label}": breach_count,
            f"Breach_Rate_{alpha_label}": breach_rate,
            f"Kupiec_pvalue_{alpha_label}": kupiec_res["p_value"],
            f"Christoffersen_Indep_pvalue_{alpha_label}": ind_res["p_value"],
            f"Christoffersen_CondCov_pvalue_{alpha_label}": cc_res["p_value"],
            f"Avg_Exceedance_Loss_{alpha_label}": avg_loss,
        })

    return summary_metrics, time_series_df