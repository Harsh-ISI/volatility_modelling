import numpy as np
from scipy.stats import chi2
from arch import arch_model
import pandas as pd


def kupiec_test(violations, confidence_level):
    """
    Kupiec unconditional coverage test.

    Parameters
    ----------
    violations : array-like
        Boolean or 0/1 VaR violation indicators.
    confidence_level : float
        VaR confidence level, e.g. 0.95 or 0.99.

    Returns
    -------
    dict
        Test statistic, p-value, observed and expected
        violation rates.
    """

    violations = np.asarray(violations, dtype=int)

    T = len(violations)
    N = violations.sum()

    expected_probability = 1 - confidence_level
    observed_probability = N / T

    # Handle boundary cases
    if N == 0 or N == T:
        return {
            "n_observations": T,
            "n_violations": N,
            "expected_rate": expected_probability,
            "observed_rate": observed_probability,
            "lr_statistic": np.nan,
            "p_value": np.nan,
        }

    log_likelihood_null = (
        (T - N) * np.log(1 - expected_probability)
        + N * np.log(expected_probability)
    )

    log_likelihood_unrestricted = (
        (T - N) * np.log(1 - observed_probability)
        + N * np.log(observed_probability)
    )

    lr_statistic = -2 * (
        log_likelihood_null
        - log_likelihood_unrestricted
    )

    p_value = 1 - chi2.cdf(
        lr_statistic,
        df=1
    )

    return {
        "n_observations": T,
        "n_violations": N,
        "expected_rate": expected_probability,
        "observed_rate": observed_probability,
        "lr_statistic": lr_statistic,
        "p_value": p_value,
    }

# def christoffersen_independence_test(violations):
#     """
#     Christoffersen independence test for VaR violations.
#     """

#     violations = np.asarray(
#         violations,
#         dtype=int
#     )

#     previous = violations[:-1]
#     current = violations[1:]

#     n00 = np.sum(
#         (previous == 0) & (current == 0)
#     )

#     n01 = np.sum(
#         (previous == 0) & (current == 1)
#     )

#     n10 = np.sum(
#         (previous == 1) & (current == 0)
#     )

#     n11 = np.sum(
#         (previous == 1) & (current == 1)
#     )

#     # Transition probabilities
#     pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
#     pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0

#     total_transitions = (
#         n00 + n01 + n10 + n11
#     )

#     total_violations = n01 + n11

#     pi = (
#         total_violations / total_transitions
#         if total_transitions > 0
#         else 0
#     )

#     # Handle boundary cases
#     if (
#         pi01 in [0, 1]
#         or pi11 in [0, 1]
#         or pi in [0, 1]
#     ):
#         return {
#             "n00": n00,
#             "n01": n01,
#             "n10": n10,
#             "n11": n11,
#             "lr_statistic": np.nan,
#             "p_value": np.nan,
#         }

#     log_likelihood_independent = (
#         (n00 + n10) * np.log(1 - pi)
#         + (n01 + n11) * np.log(pi)
#     )

#     log_likelihood_markov = (
#         n00 * np.log(1 - pi01)
#         + n01 * np.log(pi01)
#         + n10 * np.log(1 - pi11)
#         + n11 * np.log(pi11)
#     )

#     lr_statistic = -2 * (
#         log_likelihood_independent
#         - log_likelihood_markov
#     )

#     p_value = 1 - chi2.cdf(
#         lr_statistic,
#         df=1
#     )

#     return {
#         "n00": n00,
#         "n01": n01,
#         "n10": n10,
#         "n11": n11,
#         "lr_statistic": lr_statistic,
#         "p_value": p_value,
#     }

# def christoffersen_conditional_coverage_test(
#     violations,
#     confidence_level
# ):
#     """
#     Christoffersen conditional coverage test.
#     """

#     uc = kupiec_test(
#         violations,
#         confidence_level
#     )

#     ind = christoffersen_independence_test(
#         violations
#     )

#     if (
#         np.isnan(uc["lr_statistic"])
#         or np.isnan(ind["lr_statistic"])
#     ):
#         return {
#             "lr_uc": uc["lr_statistic"],
#             "lr_ind": ind["lr_statistic"],
#             "lr_cc": np.nan,
#             "p_value": np.nan,
#         }

#     lr_cc = (
#         uc["lr_statistic"]
#         + ind["lr_statistic"]
#     )

#     p_value = 1 - chi2.cdf(
#         lr_cc,
#         df=2
#     )

#     return {
#         "lr_uc": uc["lr_statistic"],
#         "lr_ind": ind["lr_statistic"],
#         "lr_cc": lr_cc,
#         "p_value": p_value,
#     }

def christoffersen_independence_test(violations):
    """
    Christoffersen independence test for VaR violations.
    """

    violations = np.asarray(
        violations,
        dtype=int
    )

    previous = violations[:-1]
    current = violations[1:]

    n00 = np.sum(
        (previous == 0) & (current == 0)
    )

    n01 = np.sum(
        (previous == 0) & (current == 1)
    )

    n10 = np.sum(
        (previous == 1) & (current == 0)
    )

    n11 = np.sum(
        (previous == 1) & (current == 1)
    )

    # Transition probabilities
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0

    total_transitions = (
        n00 + n01 + n10 + n11
    )

    total_violations = n01 + n11

    pi = (
        total_violations / total_transitions
        if total_transitions > 0
        else 0
    )

    # Helper function to compute n * log(p) safely (0 * log(0) = 0)
    def safe_n_log_p(n, p):
        return n * np.log(p) if n > 0 and p > 0 else 0.0

    # If no violations occurred at all
    if total_violations == 0:
        return {
            "n00": n00,
            "n01": n01,
            "n10": n10,
            "n11": n11,
            "lr_statistic": 0.0,
            "p_value": 1.0,
        }

    log_likelihood_independent = (
        safe_n_log_p(n00 + n10, 1 - pi)
        + safe_n_log_p(n01 + n11, pi)
    )

    log_likelihood_markov = (
        safe_n_log_p(n00, 1 - pi01)
        + safe_n_log_p(n01, pi01)
        + safe_n_log_p(n10, 1 - pi11)
        + safe_n_log_p(n11, pi11)
    )

    lr_statistic = max(0.0, -2 * (
        log_likelihood_independent
        - log_likelihood_markov
    ))

    p_value = 1 - chi2.cdf(
        lr_statistic,
        df=1
    )

    return {
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "lr_statistic": lr_statistic,
        "p_value": p_value,
    }

def christoffersen_conditional_coverage_test(
    violations,
    confidence_level
):
    """
    Christoffersen conditional coverage test.
    """

    uc = kupiec_test(
        violations,
        confidence_level
    )

    ind = christoffersen_independence_test(
        violations
    )

    if (
        np.isnan(uc["lr_statistic"])
        or np.isnan(ind["lr_statistic"])
    ):
        return {
            "lr_uc": uc["lr_statistic"],
            "lr_ind": ind["lr_statistic"],
            "lr_cc": np.nan,
            "p_value": np.nan,
        }

    lr_cc = (
        uc["lr_statistic"]
        + ind["lr_statistic"]
    )

    p_value = 1 - chi2.cdf(
        lr_cc,
        df=2
    )

    return {
        "lr_uc": uc["lr_statistic"],
        "lr_ind": ind["lr_statistic"],
        "lr_cc": lr_cc,
        "p_value": p_value,
    }

def simulate_garch_horizon(
    omega,
    alpha,
    beta,
    nu,
    mu,
    last_return,
    last_variance,
    horizon=10,
    n_simulations=10000,
    random_state=None
):
    """
    Simulate future cumulative returns from a
    Student-t GARCH(1,1) model.

    Returns
    -------
    np.ndarray
        Simulated cumulative returns with shape
        (n_simulations,).
    """

    rng = np.random.default_rng(random_state)

    # Standardized Student-t innovations
    z = (
        rng.standard_t(
            df=nu,
            size=(n_simulations, horizon)
        )
        * np.sqrt((nu - 2) / nu)
    )

    returns = np.zeros(
        (n_simulations, horizon)
    )

    variances = np.zeros(
        (n_simulations, horizon)
    )

    # First step
    previous_variance = last_variance
    previous_return = last_return

    for h in range(horizon):

        if h == 0:
            previous_residual_squared = (
                previous_return - mu
            ) ** 2

        else:
            previous_residual_squared = (
                returns[:, h - 1] - mu
            ) ** 2

        if h == 0:
            variances[:, h] = (
                omega
                + alpha * previous_residual_squared
                + beta * previous_variance
            )
        else:
            variances[:, h] = (
                omega
                + alpha * previous_residual_squared
                + beta * variances[:, h - 1]
            )

        returns[:, h] = (
            mu
            + np.sqrt(variances[:, h]) * z[:, h]
        )

    cumulative_returns = returns.sum(axis=1)

    return cumulative_returns

def walk_forward_garch_10day(
    returns,
    forecast_start,
    forecast_end,
    n_simulations=10000,
    random_state=42
):
    """
    Expanding-window 10-day VaR and ES forecasts
    using Student-t GARCH(1,1).
    """

    forecast_dates = returns.loc[
        forecast_start:forecast_end
    ].index

    results = []

    for i, date in enumerate(forecast_dates):

        estimation_data = returns[
            returns.index < date
        ] * 100

        model = arch_model(
            estimation_data,
            mean="Constant",
            vol="GARCH",
            p=1,
            q=1,
            dist="t"
        )

        fitted = model.fit(
            update_freq=0,
            disp="off"
        )

        params = fitted.params

        omega = params["omega"]
        alpha = params["alpha[1]"]
        beta = params["beta[1]"]
        nu = params["nu"]
        mu = params["mu"]

        last_variance = (
            fitted.conditional_volatility.iloc[-1]
            ** 2
        )

        last_return = estimation_data.iloc[-1]

        simulated_returns = simulate_garch_horizon(
            omega=omega,
            alpha=alpha,
            beta=beta,
            nu=nu,
            mu=mu,
            last_return=last_return,
            last_variance=last_variance,
            horizon=10,
            n_simulations=n_simulations,
            random_state=random_state + i
        )

        # Convert from percentage units back to decimals
        simulated_returns = simulated_returns / 100

        var_95 = -np.quantile(
            simulated_returns,
            0.05
        )

        var_99 = -np.quantile(
            simulated_returns,
            0.01
        )

        es_95 = -simulated_returns[
            simulated_returns
            <= -var_95
        ].mean()

        es_99 = -simulated_returns[
            simulated_returns
            <= -var_99
        ].mean()

        results.append({
            "Date": date,
            "VaR_10_95": var_95,
            "ES_10_95": es_95,
            "VaR_10_99": var_99,
            "ES_10_99": es_99
        })

    return pd.DataFrame(
        results
    ).set_index("Date")