# Experimental Results & Framework Comparison — Team 6

**Team 6 Members:**
- **Ramitha MV** (`PES1UG24AM392`)
- **Gayathri Priya V** (`PES1UG24AM105`)
- **Gonu Guntla Hema Kowshik** (`PES1UG24AM106`)

## 1. Quantitative Regression & Option Regime Classification Metrics (Test Split, $N_{\text{test}}=450$)

| Framework | Test RMSE $\downarrow$ | Test MAE $\downarrow$ | Test $R^2$ $\uparrow$ | MAPE (%) $\downarrow$ | No-Arbitrage (%) $\uparrow$ | Moneyness Regime Acc (%) $\uparrow$ | Train Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **XGBoost** | `0.00682` | `0.00519` | `0.99640` | `4.62%` | `95.78%` | `99.11%` | `5.83s` |
| **Keras (TF)** | `0.03081` | `0.02133` | `0.92639` | `23.00%` | `93.11%` | `96.00%` | `23.33s` |
| **PyTorch + fastai** | `0.01405` | `0.01059` | `0.98470` | `12.98%` | `94.67%` | `97.33%` | `34.60s` |
| **PyTorch + TabNet** | `0.02486` | `0.01878` | `0.95208` | `22.89%` | `92.00%` | `94.44%` | `29.99s` |
| **PyTorch + Lightning Flash** | `0.02247` | `0.01675` | `0.96085` | `18.79%` | `92.44%` | `95.56%` | `34.63s` |
| **PI-BSNet (Innovation)** | `0.00624` | `0.00468` | `0.99698` | `4.76%` | `100.00%` | `99.11%` | `44.20s` |

## 2. Qualitative Comparison: Simplicity, Transparency, and Efficacy

| Framework | Simplicity (Score / 10) | Transparency (Score / 10) | Efficacy (Score / 10) | Boilerplate LOC |
|---|---|---|---|---:|
| **XGBoost** | **9.5/10** — Very High (5 LOC fit/predict) | **9.2/10** — Very High (Exact Split Gain & Tree SHAP) | **9.4/10** — Very High (Fastest CPU Training & Strong R2) | `12` |
| **Keras (TF)** | **8.5/10** — High (Sequential Layer API) | **5.8/10** — Moderate-Low (Black-box Dense Weights, post-hoc SHAP) | **8.6/10** — High (Smooth Continuous Surface Approximation) | `24` |
| **PyTorch + fastai** | **9.0/10** — Very High (TabularDataLoaders + fit_one_cycle) | **6.8/10** — Moderate (Permutation Importance & Gradient Attribution) | **9.3/10** — Very High (1-Cycle Super-Convergence & BatchNorm) | `16` |
| **PyTorch + TabNet** | **7.8/10** — Moderate-High (Scikit-style TabNetRegressor wrapper) | **9.6/10** — Highest (Built-in Instance-wise Sparsemax Attention Masks) | **8.9/10** — High (Attentive Feature Selection on Nonlinear Skew) | `22` |
| **PyTorch + Lightning Flash** | **8.8/10** — High (Task-centric TabularRegressor + Trainer) | **6.5/10** — Moderate (Lightning Callback Hooks & Feature Gradients) | **9.1/10** — High (Gated Residual Highway Tabular Backbone) | `18` |
| **PI-BSNet (Innovation)** | **7.5/10** — Moderate (Custom No-Arbitrage Loss & Greek Gradients) | **9.4/10** — Very High (Recovers Analytical Greeks Delta & Vega) | **9.8/10** — Highest (Best RMSE/R2 + 100% No-Arbitrage Compliance) | `32` |

## 3. Feature Attribution Transparency: XGBoost Split Gain vs. TabNet Sparsemax Mask

| Feature | XGBoost Split Gain (%) | TabNet Sparsemax Attention (%) | Economic Role in Black-Scholes |
|---|---:|---:|---|
| **Moneyness (M)** | `55.99%` | `21.36%` | Primary strike-to-spot ratio $M = K/S$ |
| **Maturity (τ)** | `11.33%` | `7.02%` | Time horizon $\tau = T - t$ governing time value decay |
| **Risk-Free (r)** | `0.09%` | `0.00%` | Risk-free rate $r$ discounting strike payoff $e^{-r\tau}M$ |
| **Volatility (σ)** | `3.52%` | `11.45%` | Implied volatility smile $\sigma(M, \tau)$ from Eq. 5-2 |
| **ln(M)** | `0.00%` | `16.78%` | Log-moneyness $\ln(M)$ inside $d_1$ and $d_2$ numerator |
| **σ√τ** | `0.18%` | `14.39%` | Total horizon volatility $\sigma\sqrt{\tau}$ in $d_1, d_2$ denominator |
| **exp(-rτ)** | `2.34%` | `0.00%` | Present-value discount factor $e^{-r\tau}$ |
| **Intrinsic max(1-M,0)** | `26.55%` | `29.00%` | Intrinsic option lower bound $\max(1 - M, 0)$ |
