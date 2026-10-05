# Stochastic Modelling of Portfolio Returns under Volatility Dynamics

## Project Overview

This project studies portfolio return dynamics and tail risk under two volatility frameworks:

- **Student-t GARCH**
- **Heston stochastic volatility**

The main objective is to compare their ability to model volatility dynamics and forecast portfolio tail risk using **Value-at-Risk (VaR)** and **Expected Shortfall (ES)** at **95% and 99% confidence levels**, over both **1-day and 10-day horizons**.

The project is designed around an out-of-sample workflow rather than fitting and evaluating models on the same data.

---

## Portfolio Construction

The portfolio was treated as a synthetic portfolio constructed from multiple assets.

The portfolio value was defined as:

$$
V_t = \sum_i w_i \frac{S_{i,t}}{S_{i,0}}
$$

where:

- $S_{i,t}$ is the price of asset $i$ at time $t$,
- $S_{i,0}$ is its initial price,
- $w_i$ is its portfolio weight.

Portfolio returns were then obtained from the portfolio value series.

---

## Data Split

The project uses a chronological train-validation-test structure:

```text
2015–2021  → Training
2022–2023  → Validation
2024–2025  → Final Test
```

The training period is used for model estimation.

The validation period is used for methodology/model diagnostics.

The final test period is kept separate for evaluating the selected methodology on unseen observations.

---

# 1. Student-t GARCH

## Motivation

GARCH was used to model conditional volatility and volatility clustering in portfolio returns.

Both Gaussian and Student-t GARCH specifications were considered.

The Student-t specification was retained for the main comparison because it provides a heavier-tailed conditional return distribution.

## VaR and ES

The GARCH framework was used to generate 1-day and 10-day ahead tail-risk forecasts.

Risk forecasts were produced at:

- 95% confidence
- 99% confidence

For each forecast:

$$
VaR_{\alpha}
$$

measures the loss threshold exceeded with probability approximately $1-\alpha$, while Expected Shortfall measures the average loss conditional on exceeding that threshold.

## GARCH Validation Findings

For the 1-day validation forecasts, the Student-t GARCH model produced approximately:

- **95% violation rate:** 6.79%
- **99% violation rate:** 1.60%

The violations were found to be independent under the independence diagnostic.

Student-t GARCH did not produce a dramatic improvement over Gaussian GARCH in all parts of the tail, although it produced more accurate 99% coverage in the earlier comparison.

For 10-day forecasts using overlapping observations:

- **95% violation rate:** 9.35%
- **99% violation rate:** 1.02%

The 95% result indicated underestimation of the 10-day 95% tail.

Because overlapping multi-day forecasts introduce dependence between observations, a non-overlapping diagnostic was also performed:

- **95% violation rate:** 6.0%
- **99% violation rate:** 0.0%

For the non-overlapping 95% forecasts, there were 3 violations out of 50 observations and the Kupiec p-value was approximately **0.753**, so the 6% observed rate was statistically consistent with the 5% target given the small sample.

The 99% non-overlapping sample had zero violations, so the Kupiec likelihood-ratio statistic was not informative.

The 10-day results were therefore treated as a diagnostic rather than a reason to continually tune the model.

---

# 2. Heston Stochastic Volatility

The second framework models the portfolio using a Heston-style stochastic variance process.

The model is:

$$
dS_t = \mu S_t dt + \sqrt{v_t}S_t dW_t^S
$$

and

$$
dv_t =
\kappa(\theta-v_t)dt
+
\xi\sqrt{v_t}dW_t^v
$$

with:

$$
dW_t^S dW_t^v = \rho dt.
$$

The parameters are:

- $\mu$: annualized portfolio drift
- $\kappa$: variance mean-reversion speed
- $\theta$: long-run variance
- $\xi$: volatility of volatility
- $\rho$: return-volatility shock correlation
- $v_0$: initial variance

## Historical Estimation

Because the synthetic portfolio does not have a directly observable option-implied volatility surface, the project uses a historical two-step estimation procedure.

A **21-day rolling realized variance** was used as an observable proxy for the latent Heston variance:

$$
\hat v_t =
252 \times Var(r_{t-20},...,r_t).
$$

The variance dynamics were then estimated from the discretized Heston variance equation.

The return/variance shock correlation was estimated from standardized return and variance shocks.

This should be described as an **observable rolling-variance proxy for the latent variance state**, rather than claiming that the latent state was recovered using a Kalman or particle filter.

---

## Estimated Heston Parameters

Using the training period, the estimated parameters were:

| Parameter | Estimate |
|---|---:|
| $\mu$ | 0.1580 |
| $\kappa$ | 1.6454 |
| $\theta$ | 0.03423 |
| $\xi$ | 0.4074 |
| $\rho$ | -0.1141 |
| $v_0$ | 0.02849 |

Interpretation:

- Long-run annualized volatility implied by $\theta$ is approximately **18.5%**.
- Initial annualized volatility implied by $v_0$ is approximately **16.9%**.
- $\rho<0$ gives a modest negative return-volatility relationship.

### Feller Condition

The fitted parameters give:

$$
2\kappa\theta = 0.11265
$$

and

$$
\xi^2 = 0.16601.
$$

Therefore:

$$
2\kappa\theta < \xi^2
$$

and the Feller condition is **not satisfied**.

The estimated parameters were **not artificially modified to force the Feller condition**.

Instead, the Monte Carlo implementation uses a **full-truncation Euler discretization**, which prevents numerical negative variances during simulation.

---

# 3. Heston Monte Carlo Simulation

The Heston model was simulated using correlated Brownian shocks and Euler discretization.

The simulation uses:

- **50,000 paths** for model sanity checks/final-scale simulation
- **10,000 paths** for computationally lighter validation forecasts

Both 1-day and 10-day cumulative returns were simulated.

For the 1-day training-period sanity check:

- Mean simulated return: **0.000621**
- Standard deviation: **0.010661**
- 5% quantile: **-0.01691**
- 1% quantile: **-0.02422**

The simulated annualized volatility was approximately:

- Historical: **18.73%**
- Heston simulated: **16.92%**

For 10-day simulations:

- Mean return: **0.006214**
- Volatility: **0.03384**
- 5% quantile: **-0.05011**
- 1% quantile: **-0.07544**

These checks indicated that the Heston simulation was numerically stable and generated plausible return distributions before proceeding to VaR/ES evaluation.

---

# 4. VaR and Expected Shortfall Evaluation

Both models were evaluated using:

- 1-day VaR
- 10-day VaR
- 95% confidence
- 99% confidence
- Expected Shortfall

The project uses Monte Carlo distributions to estimate the relevant lower-tail quantiles and conditional tail losses.

For a simulated return distribution, VaR is calculated from the relevant lower quantile. ES is calculated as the average simulated return beyond the VaR threshold, expressed as a positive loss.

---

## VaR Backtesting

VaR forecasts were evaluated using:

### Kupiec Unconditional Coverage Test

Tests whether the observed violation frequency is consistent with the nominal VaR level.

For example:

- 95% VaR → expected violation rate of 5%
- 99% VaR → expected violation rate of 1%

### Christoffersen Conditional Coverage Test

Combines unconditional coverage with an independence test to assess whether VaR violations occur independently rather than clustering.

These tests were used to distinguish:

- correct overall violation frequency,
- independent violations,
- and situations where violations cluster.

---

# 5. Key Project Findings So Far

The main findings from the completed work are:

1. **Both GARCH and Heston provide usable frameworks for modelling portfolio volatility and tail risk.**

2. **Student-t GARCH improves the representation of extreme tails compared with Gaussian GARCH in some settings**, particularly around the 99% confidence level, although the improvement is not uniformly large.

3. **1-day GARCH VaR showed reasonably close coverage**, with approximately 6.79% violations at 95% and 1.60% at 99% during validation.

4. **10-day VaR is more difficult to calibrate accurately**, particularly around the 95% tail. Overlapping 10-day forecasts produced elevated 95% violations, while the non-overlapping robustness check gave a 6% violation rate.

5. The Heston parameter estimates produced a **negative return-volatility correlation ($\rho\approx-0.114$)**, consistent with a modest leverage-type effect.

6. The estimated Heston parameters **violate the Feller condition**, illustrating a practical issue when estimating stochastic-volatility models from historical data rather than liquid option surfaces.

7. **Full-truncation Euler simulation** was used to handle the numerical consequences of the Feller violation without artificially altering the fitted parameters.

8. The project explicitly evaluates both **VaR threshold accuracy and Expected Shortfall tail severity**, rather than relying on VaR alone.

---

# Current Status

The project has completed:

- Portfolio construction
- Data preparation
- Chronological train/validation/test design
- Gaussian vs Student-t GARCH comparison
- GARCH VaR/ES forecasting
- GARCH VaR backtesting
- Historical Heston parameter estimation
- Feller-condition diagnostic
- Heston Monte Carlo simulation
- Initial Heston VaR/ES framework
- 1-day and 10-day validation analysis

The remaining work is to clean up the implementation and produce the final **2024–2025 untouched test-set comparison** between Student-t GARCH and Heston.

The final comparison should focus on:

- 95% VaR coverage
- 99% VaR coverage
- Expected Shortfall
- 1-day horizon
- 10-day horizon
- Kupiec coverage
- Christoffersen independence/conditional coverage
- realized tail losses versus forecast ES

---

## Important Methodological Notes

### Synthetic portfolio and Heston aggregation

The portfolio is treated as a single synthetic return series for the purpose of the Heston model. This is a modelling approximation: a weighted sum of assets does not generally inherit a standard Heston process exactly.

### Historical Heston estimation

The Heston variance is latent. This project does not claim to perform full latent-state filtering or option-surface calibration. Instead, it uses rolling realized variance as an observable proxy and estimates the Heston parameters from historical dynamics.

### Multi-day VaR

10-day forecasts can be constructed using overlapping observations, but overlapping observations create dependence between realized losses. Therefore, non-overlapping diagnostics are useful as a robustness check.

### Feller condition

The Feller condition was treated as a diagnostic rather than an estimation constraint. The fitted parameters were retained and a full-truncation Euler scheme was used for stable simulation.
