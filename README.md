# Stochastic Modelling of Portfolio Returns under Volatility Dynamics

A quantitative risk framework comparing **Student's t-GARCH(1,1)** and **Heston stochastic volatility** models for portfolio tail-risk forecasting. The project evaluates **1-day and 10-day Value-at-Risk (VaR)** and **Expected Shortfall (ES)** at 95% and 99% confidence levels using out-of-sample walk-forward forecasts and formal statistical backtesting.

---

## Project Overview

The project investigates how discrete-time conditional volatility and continuous-time stochastic volatility models perform when forecasting portfolio tail risk.

The core comparison is between:

- **Student's t-GARCH(1,1)** — discrete-time conditional heteroskedasticity with heavy-tailed innovations.
- **Heston stochastic volatility** — continuous-time stochastic variance with mean reversion, volatility-of-volatility, and return-volatility correlation.

The evaluation focuses on **out-of-sample tail-risk calibration**, rather than simply comparing in-sample model fit.

### Key Features

- Chronological train/validation/test framework
  - Training: **2015–2021**
  - Validation: **2022–2023**
  - Final test: **2024–2025**
- 1-day and 10-day VaR and Expected Shortfall
- 95% and 99% confidence levels
- Monte Carlo simulation
- Student's t innovations for heavy-tailed returns
- Heston Euler-Maruyama simulation with Full Truncation
- Kupiec Unconditional Coverage / Proportion of Failures testing
- Christoffersen Independence testing
- Christoffersen Conditional Coverage testing
- Realized exceedance-loss vs. forecast ES comparison

---

# Final Out-of-Sample Results

The following results are from the **untouched 2024–2025 test period**.

| Metric | 1D GARCH-t | 1D Heston | 10D GARCH-t | 10D Heston |
|---|---:|---:|---:|---:|
| **VaR Mean (1%)** | 2.37% | 2.08% | 7.12% | 6.39% |
| **ES Mean (1%)** | 3.07% | 2.39% | 9.41% | 7.40% |
| **Breach Count / Rate (1%)** | 7 / 1.39% | 9 / 1.79% | 9 / 1.83% | 11 / 2.23% |
| **Kupiec POF p-value (1%)** | **0.4019** | 0.1082 | 0.0986 | 0.0180 |
| **Christoffersen Independence p-value (1%)** | 0.0786 | 0.1441 | 0.0000* | 0.0000* |
| **Christoffersen Conditional Coverage p-value (1%)** | 0.1498 | 0.0947 | 0.0000* | 0.0000* |
| **Avg. Exceedance Loss (1%)** | -3.70% | -3.11% | -11.13% | -10.00% |
| **VaR Mean (5%)** | 1.43% | 1.46% | 4.03% | 4.37% |
| **ES Mean (5%)** | 2.03% | 1.84% | 6.01% | 5.61% |
| **Breach Count / Rate (5%)** | 27 / 5.38% | 28 / 5.58% | 28 / 5.68% | **22 / 4.46%** |
| **Kupiec POF p-value (5%)** | 0.7005 | 0.5595 | 0.4976 | 0.5773 |
| **Christoffersen Independence p-value (5%)** | 0.6743 | 0.6109 | 0.0000* | 0.0000* |
| **Christoffersen Conditional Coverage p-value (5%)** | 0.8502 | 0.7411 | 0.0000* | 0.0000* |
| **Avg. Exceedance Loss (5%)** | -2.02% | -1.98% | -6.99% | -7.70% |

\* The 10-day forecasts are overlapping. Successive 10-day realized windows share 9 out of 10 observations, creating serial dependence in the violation sequence. Therefore, the near-zero Christoffersen p-values should **not** be interpreted as direct evidence that the underlying volatility model fails.

---

# Key Findings

## 1. Student's t-GARCH performs strongly in the 1-day extreme tail

The **1-day Student's t-GARCH** provides the strongest 1% VaR coverage among the tested specifications:

- Observed 1% breach rate: **1.39%**
- Target breach rate: **1.00%**
- Kupiec POF p-value: **0.4019**
- Christoffersen conditional-coverage p-value: **0.1498**

The null hypothesis of correct unconditional coverage is therefore not rejected at conventional significance levels.

The model also produces a close match between predicted and realized 5% tail severity:

- Forecast ES: **2.03%**
- Average realized exceedance loss: **2.02%**

This is one of the strongest empirical results in the final test period.

---

## 2. Both models provide reasonable 1-day 5% coverage

At the 5% confidence level:

- GARCH-t breach rate: **5.38%**
- Heston breach rate: **5.58%**

Both models have non-rejectionary Kupiec tests:

- GARCH-t: **p = 0.7005**
- Heston: **p = 0.5595**

Thus, both specifications provide reasonably calibrated unconditional 5% VaR over the final test sample.

---

## 3. 10-day extreme-tail performance is more difficult

At the 10-day, 1% level:

- GARCH-t breach rate: **1.83%**
- Heston breach rate: **2.23%**

The corresponding Kupiec p-values are:

- GARCH-t: **0.0986**
- Heston: **0.0180**

Heston therefore rejects nominal 1% unconditional coverage at the 5% significance level, whereas GARCH-t does not.

However, both models still produce more violations than the nominal 1% target, indicating that **10-day extreme-tail risk remains difficult to calibrate accurately**.

---

## 4. The 10-day 5% forecasts are comparatively well calibrated

At the 10-day 5% level:

- GARCH-t: **5.68%** breach rate, Kupiec p = **0.4976**
- Heston: **4.46%** breach rate, Kupiec p = **0.5773**

Neither model is rejected by the Kupiec test.

This suggests that both models capture the broader 10-day downside distribution reasonably well at the 5% tail, despite the challenges observed at the more extreme 1% level.

---

# Methodology

## 1. Portfolio Construction

The portfolio is represented by a normalized weighted basket of component assets.

For asset prices $S_{i,t}$ and fixed weights $w_i$:

$$
V_t =
\sum_{i=1}^{N}
w_i
\frac{S_{i,t}}{S_{i,0}}
$$

with:

$$
\sum_i w_i = 1.
$$

Portfolio log returns are then calculated as:

$$
r_t =
\ln\left(
\frac{V_t}{V_{t-1}}
\right).
$$

The resulting portfolio return series is used as the common input for both volatility models.

---

# 2. Chronological Data Architecture

The project avoids random train/test splitting and instead follows a chronological structure:

```text
2015–2021
    │
    └── Training
        Model estimation

2022–2023
    │
    └── Validation
        Specification and diagnostic analysis

2024–2025
    │
    └── Final Test
        Untouched out-of-sample evaluation