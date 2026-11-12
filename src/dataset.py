"""
Parametric Black-Scholes Option Pricing Dataset Generator with Volatility Smile/Skew
and Economic Greeks (Delta, Gamma, Vega) for Team 6 Business Data Case Study.

Implements exact specifications from Case Study Document:
  - Normalized Moneyness: M = K / S  (Strike-to-Spot ratio)
  - Dividend Yield: q = 0
  - Volatility Smile/Skew Surface (Equation 5-2):
        sigma(M, tau) = sigma_0 + alpha * tau + beta * (M - 1)^2
  - Normalized Black-Scholes Call Option Price (C / S):
        C_norm = Phi(d_1) - e^(-r * tau) * M * Phi(d_2)
        d_1 = (-ln(M) + (r + sigma^2 / 2) * tau) / (sigma * sqrt(tau))
        d_2 = d_1 - sigma * sqrt(tau)

Authors (Team 6):
  1. Ramitha MV (PES1UG24AM392)
  2. Gayathri Priya V (PES1UG24AM105)
  3. Gonu Guntla Hema Kowshik (PES1UG24AM106)
"""

import os
import csv
import math
import json
import random
from typing import List, Dict, Tuple, Any


def norm_cdf(x: float) -> float:
    """Cumulative distribution function for the standard normal distribution Phi(x)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    """Probability density function for the standard normal distribution phi(x)."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def implied_volatility_surface(
    M: float,
    tau: float,
    sigma_0: float = 0.15,
    alpha: float = 0.08,
    beta: float = 0.65,
    micro_noise: float = 0.0,
) -> float:
    """
    Equation 5-2 from Case Study:
        sigma(M, tau) = sigma_0 + alpha * tau + beta * (M - 1)^2
    Captures both the term structure of volatility (alpha * tau) and the quadratic
    volatility smile/skew across moneyness (beta * (M - 1)^2).
    """
    sigma = sigma_0 + alpha * tau + beta * ((M - 1.0) ** 2) + micro_noise
    return max(0.05, sigma)


def black_scholes_normalized(
    M: float, r: float, tau: float, sigma: float, q: float = 0.0
) -> Dict[str, float]:
    """
    Computes exact Black-Scholes normalized call option price (C / S) and analytical Greeks
    using the simplified moneyness formula on Page 3 of the case study:
        e^(-q * tau) * Phi(d_1) - e^(-r * tau) * M * Phi(d_2)
    with d_1 = (-ln(M) + (r - q + sigma^2 / 2) * tau) / (sigma * sqrt(tau))
    and  d_2 = d_1 - sigma * sqrt(tau).
    """
    sqrt_tau = math.sqrt(max(tau, 1e-6))
    vol_sqrt_tau = sigma * sqrt_tau

    d1 = (-math.log(M) + (r - q + 0.5 * sigma * sigma) * tau) / vol_sqrt_tau
    d2 = d1 - vol_sqrt_tau

    phi_d1 = norm_cdf(d1)
    phi_d2 = norm_cdf(d2)
    pdf_d1 = norm_pdf(d1)

    disc_q = math.exp(-q * tau)
    disc_r = math.exp(-r * tau)

    # Normalized Call Price (c = C / S)
    call_price_norm = disc_q * phi_d1 - disc_r * M * phi_d2
    call_price_norm = max(0.0, call_price_norm)

    # Intrinsic value per unit of stock price: max(1 - M, 0)
    intrinsic_norm = max(0.0, 1.0 - M)
    time_value_norm = max(0.0, call_price_norm - intrinsic_norm)

    # Analytical Option Greeks (normalized with respect to spot S=1)
    delta = disc_q * phi_d1                                  # dC / dS
    gamma = (disc_q * pdf_d1) / vol_sqrt_tau                 # S * d^2C / dS^2
    vega = disc_q * pdf_d1 * sqrt_tau                        # (1/S) * dC / d(sigma)
    theta = (
        -(disc_q * pdf_d1 * sigma) / (2.0 * sqrt_tau)
        - r * M * disc_r * phi_d2
        + q * disc_q * phi_d1
    )

    return {
        "call_price_norm": call_price_norm,
        "intrinsic_norm": intrinsic_norm,
        "time_value_norm": time_value_norm,
        "d1": d1,
        "d2": d2,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
    }


def classify_moneyness_regime(M: float) -> Tuple[str, int]:
    """
    Classifies option moneyness M = K / S for Call Options:
      - In-The-Money (ITM):      M < 0.95  (Strike K < Spot S) -> Class 0
      - At-The-Money (ATM):      0.95 <= M <= 1.05             -> Class 1
      - Out-Of-The-Money (OTM):  M > 1.05  (Strike K > Spot S) -> Class 2
    """
    if M < 0.95:
        return "ITM", 0
    elif M <= 1.05:
        return "ATM", 1
    else:
        return "OTM", 2


def generate_option_dataset(
    n_samples: int = 3000,
    seed: int = 42,
    out_dir: str = "data",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Generates simulated Black-Scholes option pricing dataset with Equation 5-2
    volatility smile/skew surface and partitions into Train (70%), Val (15%), Test (15%).
    """
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)

    records: List[Dict[str, Any]] = []

    for i in range(n_samples):
        # Sample underlying stock price S in [$50, $250]
        spot_S = rng.uniform(50.0, 250.0)

        # Sample Moneyness M = K / S in [0.65, 1.35]
        M = rng.uniform(0.65, 1.35)
        strike_K = spot_S * M

        # Sample Time to Maturity tau = T - t in [0.05, 2.00] years
        tau = rng.uniform(0.05, 2.00)

        # Sample Risk-Free Interest Rate r in [0.01, 0.08] (1% to 8%)
        r = rng.uniform(0.01, 0.08)

        # Dividend yield q = 0 as specified on Page 3 of Case Study
        q = 0.0

        # Equation 5-2 Volatility Surface with tiny realistic market microstructure noise
        micro_noise = rng.gauss(0.0, 0.0025)
        sigma = implied_volatility_surface(
            M=M, tau=tau, sigma_0=0.15, alpha=0.08, beta=0.65, micro_noise=micro_noise
        )

        bs = black_scholes_normalized(M=M, r=r, tau=tau, sigma=sigma, q=q)
        regime_label, regime_id = classify_moneyness_regime(M)

        # Assign split: 70% Train (2100), 15% Val (450), 15% Test (450)
        if i < int(0.70 * n_samples):
            split = "train"
        elif i < int(0.85 * n_samples):
            split = "val"
        else:
            split = "test"

        rec = {
            "option_id": i + 1,
            "split": split,
            "spot_price_S": round(spot_S, 4),
            "strike_price_K": round(strike_K, 4),
            "moneyness_M": round(M, 6),
            "time_to_maturity_tau": round(tau, 6),
            "risk_free_rate_r": round(r, 6),
            "dividend_yield_q": round(q, 4),
            "volatility_sigma": round(sigma, 6),
            "call_price_norm": round(bs["call_price_norm"], 6),
            "call_price_dollar": round(bs["call_price_norm"] * spot_S, 4),
            "intrinsic_norm": round(bs["intrinsic_norm"], 6),
            "time_value_norm": round(bs["time_value_norm"], 6),
            "d1": round(bs["d1"], 6),
            "d2": round(bs["d2"], 6),
            "delta": round(bs["delta"], 6),
            "gamma": round(bs["gamma"], 6),
            "vega": round(bs["vega"], 6),
            "theta": round(bs["theta"], 6),
            "moneyness_regime": regime_label,
            "regime_class": regime_id,
        }
        records.append(rec)

    # Save full dataset CSV and split CSVs
    fieldnames = list(records[0].keys())
    full_csv_path = os.path.join(out_dir, "black_scholes_options_3k.csv")
    with open(full_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    for split_name in ["train", "val", "test"]:
        split_rows = [r for r in records if r["split"] == split_name]
        split_path = os.path.join(out_dir, f"options_{split_name}.csv")
        with open(split_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_rows)

    eda_summary = compute_eda_statistics(records, out_dir=out_dir)
    return records, eda_summary


def _pearson_corr(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x < 1e-12 or den_y < 1e-12:
        return 0.0
    return num / (den_x * den_y)


def _percentile(sorted_vals: List[float], p: float) -> float:
    idx = (len(sorted_vals) - 1) * p
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] * (hi - idx) + sorted_vals[hi] * (idx - lo)


def compute_eda_statistics(records: List[Dict[str, Any]], out_dir: str = "data") -> Dict[str, Any]:
    """
    Performs comprehensive Exploratory Data Analysis (EDA) across predictors and option price:
      - Descriptive statistics (Mean, Std, Min, 25%, Median, 75%, Max, Skewness)
      - Pearson correlation matrix across (M, tau, r, sigma, call_price_norm, delta, vega)
      - Moneyness regime breakdown (ITM vs ATM vs OTM)
    """
    cols = [
        "moneyness_M",
        "time_to_maturity_tau",
        "risk_free_rate_r",
        "volatility_sigma",
        "call_price_norm",
        "intrinsic_norm",
        "time_value_norm",
        "delta",
        "gamma",
        "vega",
    ]

    summary_stats: Dict[str, Dict[str, float]] = {}
    col_vectors: Dict[str, List[float]] = {c: [float(r[c]) for r in records] for c in cols}

    for c in cols:
        vals = col_vectors[c]
        s_vals = sorted(vals)
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        std = math.sqrt(var)
        skew = (sum((v - mean) ** 3 for v in vals) / n) / ((std ** 3) + 1e-12)
        summary_stats[c] = {
            "mean": round(mean, 5),
            "std": round(std, 5),
            "min": round(s_vals[0], 5),
            "q25": round(_percentile(s_vals, 0.25), 5),
            "median": round(_percentile(s_vals, 0.50), 5),
            "q75": round(_percentile(s_vals, 0.75), 5),
            "max": round(s_vals[-1], 5),
            "skewness": round(skew, 4),
        }

    corr_cols = [
        "moneyness_M",
        "time_to_maturity_tau",
        "risk_free_rate_r",
        "volatility_sigma",
        "call_price_norm",
        "delta",
        "vega",
    ]
    corr_matrix: Dict[str, Dict[str, float]] = {}
    for c1 in corr_cols:
        corr_matrix[c1] = {}
        for c2 in corr_cols:
            corr_matrix[c1][c2] = round(_pearson_corr(col_vectors[c1], col_vectors[c2]), 4)

    regime_counts: Dict[str, Dict[str, Any]] = {}
    for reg in ["ITM", "ATM", "OTM"]:
        sub = [r for r in records if r["moneyness_regime"] == reg]
        prices = [float(r["call_price_norm"]) for r in sub]
        vols = [float(r["volatility_sigma"]) for r in sub]
        deltas = [float(r["delta"]) for r in sub]
        regime_counts[reg] = {
            "count": len(sub),
            "percentage": round(100.0 * len(sub) / len(records), 2),
            "mean_price_norm": round(sum(prices) / len(sub), 5),
            "mean_volatility": round(sum(vols) / len(sub), 5),
            "mean_delta": round(sum(deltas) / len(sub), 5),
        }

    eda_payload = {
        "n_total": len(records),
        "n_train": sum(1 for r in records if r["split"] == "train"),
        "n_val": sum(1 for r in records if r["split"] == "val"),
        "n_test": sum(1 for r in records if r["split"] == "test"),
        "volatility_equation": "sigma(M, tau) = 0.15 + 0.08 * tau + 0.65 * (M - 1)^2",
        "summary_statistics": summary_stats,
        "correlation_matrix": corr_matrix,
        "regime_breakdown": regime_counts,
    }

    with open(os.path.join(out_dir, "eda_summary.json"), "w") as f:
        json.dump(eda_payload, f, indent=2)

    return eda_payload
