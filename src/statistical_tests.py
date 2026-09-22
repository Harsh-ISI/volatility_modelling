import numpy as np
import pandas as pd
from scipy.stats import chi2


def _safe_log(val: float) -> float:
    """Helper function to calculate natural log, treating 0 * log(0) as 0."""
    return np.log(val) if val > 0 else 0.0


def kupiec_test(breaches: pd.Series, alpha: float) -> dict:
    """
    Kupiec Proportion of Failures (POF) Likelihood Ratio Test.
    
    Inputs:
        breaches (pd.Series): Boolean series (True if breach occurred, False otherwise).
        alpha (float): Expected failure rate (e.g., 0.01 for 1% VaR, 0.05 for 5% VaR).
        
    Outputs:
        dict: {
            "stat": float,       # Likelihood Ratio statistic (chi-square df=1)
            "p_value": float,    # p-value of test
            "decision": str      # "Reject" (if p <= 0.05) or "Fail to Reject"
        }
    """
    b = breaches.astype(int).values
    n = len(b)
    x = int(b.sum())
    p_hat = x / n if n > 0 else 0.0

    # Log-likelihood under Null Hypothesis H0: p = alpha
    log_l0 = (n - x) * _safe_log(1 - alpha) + x * _safe_log(alpha)

    # Log-likelihood under Alternative Hypothesis H1: p = p_hat
    log_l1 = (n - x) * _safe_log(1 - p_hat) + x * _safe_log(p_hat)

    # Likelihood Ratio statistic
    stat = max(0.0, -2 * (log_l0 - log_l1))
    p_value = float(chi2.sf(stat, df=1))

    return {
        "stat": float(stat),
        "p_value": p_value,
        "decision": "Reject" if p_value <= 0.05 else "Fail to Reject"
    }


def christoffersen_independence_test(breaches: pd.Series) -> dict:
    """
    Christoffersen Independence Test (verifies breaches are not clustered).
    
    Inputs:
        breaches (pd.Series): Boolean series.
        
    Outputs:
        dict: {
            "stat": float,       # Likelihood Ratio statistic (chi-square df=1)
            "p_value": float,    # p-value of test
            "decision": str      # "Reject" or "Fail to Reject"
        }
    """
    b = breaches.astype(int).values
    if len(b) < 2:
        return {"stat": 0.0, "p_value": 1.0, "decision": "Fail to Reject"}

    i_prev = b[:-1]
    i_curr = b[1:]

    # Calculate transition matrix frequencies
    n00 = np.sum((i_prev == 0) & (i_curr == 0))
    n01 = np.sum((i_prev == 0) & (i_curr == 1))
    n10 = np.sum((i_prev == 1) & (i_curr == 0))
    n11 = np.sum((i_prev == 1) & (i_curr == 1))

    # Calculate transition probabilities
    pi_0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi_1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) > 0 else 0.0

    # Log-likelihood under Null Hypothesis H0 (Independence: pi_0 = pi_1 = pi)
    log_l0 = (n00 + n10) * _safe_log(1 - pi) + (n01 + n11) * _safe_log(pi)

    # Log-likelihood under Alternative Hypothesis H1 (1st Order Markov Chain)
    log_l1 = (
        n00 * _safe_log(1 - pi_0) +
        n01 * _safe_log(pi_0) +
        n10 * _safe_log(1 - pi_1) +
        n11 * _safe_log(pi_1)
    )

    stat = max(0.0, -2 * (log_l0 - log_l1))
    p_value = float(chi2.sf(stat, df=1))

    return {
        "stat": float(stat),
        "p_value": p_value,
        "decision": "Reject" if p_value <= 0.05 else "Fail to Reject"
    }


def christoffersen_conditional_coverage_test(breaches: pd.Series, alpha: float) -> dict:
    """
    Christoffersen Joint Conditional Coverage Test (POF + Independence).
    LR_cc = LR_pof + LR_ind ~ Chi-Square(df=2)
    
    Inputs:
        breaches (pd.Series): Boolean series.
        alpha (float): Expected failure rate.
        
    Outputs:
        dict: {
            "stat": float,       # LR_cc statistic (chi-square df=2)
            "p_value": float,    # p-value of joint test
            "decision": str      # "Reject" or "Fail to Reject"
        }
    """
    pof_res = kupiec_test(breaches, alpha)
    ind_res = christoffersen_independence_test(breaches)

    stat_cc = pof_res["stat"] + ind_res["stat"]
    p_value_cc = float(chi2.sf(stat_cc, df=2))

    return {
        "stat": float(stat_cc),
        "p_value": p_value_cc,
        "decision": "Reject" if p_value_cc <= 0.05 else "Fail to Reject"
    }