import numpy as np
import pandas as pd
from scipy.stats import linregress


def estimate_realized_variance(
    returns,
    window=21
):
    """
    Estimate annualized variance using
    rolling daily returns.
    """

    variance = (
        returns
        .rolling(window)
        .var()
        * 252
    )

    return variance.dropna()


def estimate_variance_dynamics(
    variance,
    dt=1 / 252
):
    """
    Estimate kappa and theta from the
    discretized Heston variance process.
    """

    v = variance.values

    delta_v = np.diff(v)
    v_lag = v[:-1]

    slope, intercept, _, _, _ = linregress(
        v_lag,
        delta_v
    )

    kappa = -slope / dt

    if kappa <= 0:
        raise ValueError(
            "Estimated kappa is non-positive."
        )

    theta = (
        intercept
        / (kappa * dt)
    )

    if theta <= 0:
        raise ValueError(
            "Estimated theta is non-positive."
        )

    return kappa, theta


def estimate_vol_of_vol(
    variance,
    kappa,
    theta,
    dt=1 / 252
):
    """
    Estimate Heston volatility-of-volatility xi.
    """

    v = variance.values

    delta_v = np.diff(v)
    v_lag = v[:-1]

    # Avoid numerical problems from zero variance
    valid = v_lag > 1e-12

    residuals = (
        delta_v
        - kappa
        * (theta - v_lag)
        * dt
    )

    xi_squared = np.mean(
        residuals[valid] ** 2
        / (
            v_lag[valid] * dt
        )
    )

    xi = np.sqrt(
        max(xi_squared, 0)
    )

    return xi


def estimate_correlation(
    returns,
    variance,
    kappa,
    theta,
    xi,
    mu,
    dt=1 / 252
):
    """
    Estimate correlation between return and
    variance shocks.

    The return r_{t+1} and variance transition
    v_{t+1} - v_t correspond to the same
    time interval.
    """

    # Variance at time t
    v_t = variance.iloc[:-1].values

    # Variance at time t+1
    v_next = variance.iloc[1:].values

    # Return during t -> t+1
    r_next = returns.loc[
        variance.index[1:]
    ].values

    valid = (
        np.isfinite(v_t)
        & np.isfinite(v_next)
        & np.isfinite(r_next)
        & (v_t > 1e-12)
    )

    v_t = v_t[valid]
    v_next = v_next[valid]
    r_next = r_next[valid]

    # Return shock
    z = (
        r_next - mu * dt
    ) / np.sqrt(
        v_t * dt
    )

    # Variance shock
    u = (
        v_next
        - v_t
        - kappa * (theta - v_t) * dt
    ) / (
        xi * np.sqrt(v_t * dt)
    )

    valid_shocks = (
        np.isfinite(z)
        & np.isfinite(u)
    )

    if valid_shocks.sum() < 2:
        raise ValueError(
            "Insufficient valid observations "
            "to estimate rho."
        )

    rho = np.corrcoef(
        z[valid_shocks],
        u[valid_shocks]
    )[0, 1]

    return float(
        np.clip(rho, -0.999, 0.999)
    )


def estimate_heston_parameters(
    returns,
    window=21,
    dt=1 / 252
):
    """
    Estimate Heston parameters using a
    two-step historical estimation procedure.
    """

    returns = returns.dropna()

    # Step 1: estimate variance proxy
    variance = estimate_realized_variance(
        returns,
        window=window
    )

    # Annualized drift
    mu = returns.mean() * 252

    # Step 2: estimate variance dynamics
    kappa, theta = estimate_variance_dynamics(
        variance,
        dt=dt
    )

    # Step 3: estimate volatility of volatility
    xi = estimate_vol_of_vol(
        variance,
        kappa,
        theta,
        dt=dt
    )

    # Step 4: estimate return/variance correlation
    rho = estimate_correlation(
        returns,
        variance,
        kappa,
        theta,
        xi,
        mu,
        dt=dt
    )

    # Initial variance
    v0 = variance.iloc[-1]

    return {
        "mu": mu,
        "kappa": kappa,
        "theta": theta,
        "xi": xi,
        "rho": rho,
        "v0": v0,
    }


def simulate_heston(
    mu,
    kappa,
    theta,
    xi,
    rho,
    v0,
    horizon=1,
    n_steps=None,
    n_simulations=50000,
    dt=1 / 252,
    random_state=42
):
    """
    Simulate portfolio returns under the Heston model.

    Uses full-truncation Euler discretization to
    handle variance paths when the Feller condition
    is not satisfied.

    Returns
    -------
    np.ndarray
        Cumulative simulated returns.
    """

    if n_steps is None:
        n_steps = horizon

    rng = np.random.default_rng(random_state)

    sqrt_dt = np.sqrt(dt)

    # Independent standard normal shocks
    z1 = rng.standard_normal(
        (n_simulations, n_steps)
    )

    z2 = rng.standard_normal(
        (n_simulations, n_steps)
    )

    # Correlated Brownian shocks
    dW_s = z1 * sqrt_dt

    dW_v = (
        rho * z1
        + np.sqrt(1 - rho**2) * z2
    ) * sqrt_dt

    # Initial variance
    variance = np.full(
        n_simulations,
        v0
    )

    cumulative_returns = np.zeros(
        n_simulations
    )

    for t in range(n_steps):

        # Full truncation
        variance_positive = np.maximum(
            variance,
            0
        )

        # Asset return
        returns = (
            mu * dt
            + np.sqrt(variance_positive)
            * dW_s[:, t]
        )

        cumulative_returns += returns

        # Variance process
        variance = (
            variance
            + kappa
            * (
                theta
                - variance_positive
            )
            * dt
            + xi
            * np.sqrt(variance_positive)
            * dW_v[:, t]
        )

        # Keep variance non-negative
        variance = np.maximum(
            variance,
            0
        )

    return cumulative_returns


def calculate_var_es(
    simulated_returns,
    confidence_level
):
    """
    Calculate VaR and Expected Shortfall
    from simulated returns.
    """

    alpha = 1 - confidence_level

    var = -np.quantile(
        simulated_returns,
        alpha
    )

    tail_losses = simulated_returns[
        simulated_returns <= -var
    ]

    es = -tail_losses.mean()

    return var, es


def walk_forward_heston(
    returns,
    forecast_start,
    forecast_end,
    horizon=1,
    n_simulations=10000,
    window=21,
    random_state=42
):
    """
    Expanding-window Heston VaR/ES forecasts.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns.
    forecast_start : str
        Start of validation/test period.
    forecast_end : str
        End of validation/test period.
    horizon : int
        Forecast horizon in trading days.
    n_simulations : int
        Number of Monte Carlo paths.
    """

    forecast_dates = returns.loc[
        forecast_start:forecast_end
    ].index

    results = []

    for i, date in enumerate(forecast_dates):

        # Information available strictly before
        # the forecast date
        estimation_returns = returns[
            returns.index < date
        ]

        # Estimate Heston parameters
        params = estimate_heston_parameters(
            estimation_returns,
            window=window
        )

        # Simulate future cumulative returns
        simulated_returns = simulate_heston(
            **params,
            horizon=horizon,
            n_steps=horizon,
            n_simulations=n_simulations,
            random_state=random_state + i
        )

        var_95, es_95 = calculate_var_es(
            simulated_returns,
            confidence_level=0.95
        )

        var_99, es_99 = calculate_var_es(
            simulated_returns,
            confidence_level=0.99
        )

        results.append({
            "Date": date,
            "VaR_95": var_95,
            "ES_95": es_95,
            "VaR_99": var_99,
            "ES_99": es_99
        })

    return pd.DataFrame(
        results
    ).set_index("Date")