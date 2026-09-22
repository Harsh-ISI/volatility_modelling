import sys
import numpy as np
import pandas as pd

from src.statistical_tests import (
    kupiec_test,
    christoffersen_independence_test,
    christoffersen_conditional_coverage_test,
)


def walk_forward_heston(
    portfolio_returns: pd.DataFrame | pd.Series,
    forecast_start: str,
    forecast_end: str,
    horizon: int = 1,
    overlap: bool = True,
    spot_vol_window: int = 21,
    n_sims: int = 20000,
    seed: int | None = 42,
) -> tuple[dict, pd.DataFrame]:
    """
    Expanding-window walk-forward forecasting using a simplified Heston Stochastic
    Volatility model with Euler-Maruyama discretization (Full Truncation).

    Parameters:
        portfolio_returns (pd.DataFrame | pd.Series): Daily log returns series or single-column DataFrame.
        forecast_start (str): Start date of forecast window ('YYYY-MM-DD').
        forecast_end (str): End date of forecast window ('YYYY-MM-DD').
        horizon (int): Forecast horizon in days (1 or 10).
        overlap (bool): Active only if horizon > 1.
                        - If True: Step by 1 day (rolling cumulative).
                        - If False: Step by `horizon` days (non-overlapping).
        spot_vol_window (int): Rolling window lookback to proxy spot variance V_t.
        n_sims (int): Number of Monte Carlo simulation paths per forecast date.
        seed (int | None): Random seed for reproducible Monte Carlo simulation.

    Returns:
        tuple[dict, pd.DataFrame]: (summary_metrics, time_series_df)
    """
    if seed is not None:
        np.random.seed(seed)

    # 1. Normalize input to pandas Series
    if isinstance(portfolio_returns, pd.DataFrame):
        returns_series = portfolio_returns.iloc[:, 0]
    else:
        returns_series = portfolio_returns

    # 2. Determine evaluation dates
    val_dates = returns_series.loc[forecast_start:forecast_end].index
    if (horizon > 1) and (not overlap):
        val_dates = val_dates[::horizon]

    total_steps = len(val_dates)
    results = []

    for i, date in enumerate(val_dates):
        # Progress percentage tracker
        pct = ((i + 1) / total_steps) * 100
        sys.stdout.write(f"\rHeston Walk-Forward Progress: {pct:.1f}% ({i + 1}/{total_steps})")
        sys.stdout.flush()

        idx_pos = returns_series.index.get_loc(date)

        # Ensure full horizon exists in forward data
        if idx_pos + horizon > len(returns_series):
            break

        # Extract realized return over horizon
        future_returns = returns_series.iloc[idx_pos : idx_pos + horizon]
        realized_return = future_returns.sum()

        # Historical training slice
        train_slice = returns_series.iloc[:idx_pos]
        if len(train_slice) < spot_vol_window + 10:
            continue

        # 3. Estimate Physical Heston Parameters & Spot Volatility
        mu = train_slice.mean()

        # Compute rolling variance series to fit variance dynamics
        roll_var = train_slice.rolling(window=spot_vol_window).var().dropna()
        v_t = roll_var.iloc[-1]  # Spot variance proxy V_t at day t-1
        theta = roll_var.mean()  # Long-run average variance

        # Estimate Volatility-of-Volatility (xi)
        d_var = roll_var.diff().dropna()
        xi = d_var.std()

        # Estimate Mean Reversion Speed (kappa) via AR(1) autocorrelation
        if len(roll_var) > 2:
            phi = np.corrcoef(roll_var.values[1:], roll_var.values[:-1])[0, 1]
            kappa = max(0.01, 1.0 - phi)
        else:
            kappa = 0.1

        # Estimate Leverage Effect (rho = corr(return, delta_variance))
        aligned_rets = train_slice.loc[d_var.index]
        if len(aligned_rets) > 2:
            rho = np.corrcoef(aligned_rets.values, d_var.values)[0, 1]
            if np.isnan(rho):
                rho = -0.5
        else:
            rho = -0.5

        # Sanitize parameter bounds for numerical stability
        v_t = max(v_t, 1e-6)
        theta = max(theta, 1e-6)
        kappa = np.clip(kappa, 0.005, 0.5)
        xi = np.clip(xi, 1e-5, 0.1)
        rho = np.clip(rho, -0.99, 0.99)

        # 4. Monte Carlo Simulation via Euler-Maruyama (Full Truncation)
        v = np.full(n_sims, v_t)
        cum_returns = np.zeros(n_sims)

        for _ in range(horizon):
            z1 = np.random.normal(size=n_sims)
            z2 = np.random.normal(size=n_sims)

            # Correlated shocks
            z_v = z1
            z_s = rho * z1 + np.sqrt(1.0 - rho**2) * z2

            # Full Truncation: max(V, 0)
            v_tilde = np.maximum(v, 0.0)

            # Euler updates
            v_next = v + kappa * (theta - v_tilde) + xi * np.sqrt(v_tilde) * z_v
            r_step = (mu - 0.5 * v_tilde) + np.sqrt(v_tilde) * z_s

            v = v_next
            cum_returns += r_step

        # 5. Extract Volatility, VaR, and ES from Simulated Path Distribution
        vol_hat = np.std(cum_returns)

        var_1pct = -np.percentile(cum_returns, 1.0)
        var_5pct = -np.percentile(cum_returns, 5.0)

        tail_1pct = cum_returns[cum_returns < -var_1pct]
        es_1pct = -tail_1pct.mean() if len(tail_1pct) > 0 else var_1pct

        tail_5pct = cum_returns[cum_returns < -var_5pct]
        es_5pct = -tail_5pct.mean() if len(tail_5pct) > 0 else var_5pct

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

    print("\nCompleted!")  # Newline after progress bar finishes

    # Assemble time-series DataFrame
    time_series_df = pd.DataFrame(results).set_index("Date")

    # 6. Accuracy Metrics
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