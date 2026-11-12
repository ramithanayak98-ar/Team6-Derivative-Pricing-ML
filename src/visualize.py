"""
Publication-Grade SVG Chart Generator for Derivative Pricing Case Study (Team 6).
Outputs crisp vector SVG plots for EDA, Volatility Smile Surface, Framework Benchmarking,
Predicted vs Actual Option Prices, and Simplicity-Transparency-Efficacy Radar/Triad Analysis.

Authors (Team 6):
  1. Ramitha MV (PES1UG24AM392)
  2. Gayathri Priya V (PES1UG24AM105)
  3. Gonu Guntla Hema Kowshik (PES1UG24AM106)
"""

import os
import math
from typing import List, Dict, Any


COLORS = {
    "xgboost": "#d97706",   # Amber Gold
    "keras": "#dc2626",     # Crimson Red
    "fastai": "#2563eb",    # Royal Blue
    "tabnet": "#7c3aed",    # Deep Violet
    "flash": "#0891b2",     # Cyan Teal
    "pibsnet": "#059669",   # Emerald Green
}


def generate_all_figures(
    records: List[Dict[str, Any]],
    eda_summary: Dict[str, Any],
    framework_results: List[Dict[str, Any]],
    feature_importance_data: Dict[str, Any],
    out_dir: str = "results/figures",
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = [
        _plot_volatility_smile_and_price_surface(
            records, os.path.join(out_dir, "fig1_volatility_smile_and_bs_surface.svg")
        ),
        _plot_eda_correlation_heatmap(
            eda_summary, os.path.join(out_dir, "fig2_eda_correlation_heatmap.svg")
        ),
        _plot_framework_metrics_comparison(
            framework_results, os.path.join(out_dir, "fig3_framework_rmse_r2_comparison.svg")
        ),
        _plot_simplicity_transparency_efficacy(
            framework_results, os.path.join(out_dir, "fig4_simplicity_transparency_efficacy.svg")
        ),
        _plot_feature_importance_and_convergence(
            framework_results,
            feature_importance_data,
            os.path.join(out_dir, "fig5_transparency_and_convergence.svg"),
        ),
    ]
    return paths


def _plot_volatility_smile_and_price_surface(
    records: List[Dict[str, Any]], filepath: str
) -> str:
    """
    Two-panel SVG figure:
      Left Panel:  Equation 5-2 Volatility Smile/Skew sigma(M, tau) vs Moneyness M across maturities
      Right Panel: Normalized Black-Scholes Option Price C/S vs Moneyness M vs Intrinsic Payoff
    """
    w, h = 880, 420
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" style="background:#ffffff;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif;">',
        f'<text x="{w/2}" y="28" text-anchor="middle" font-size="15.5" font-weight="bold" fill="#0f172a">EDA: Implied Volatility Smile Surface (Eq. 5-2) &amp; Black-Scholes Option Pricing Curve</text>',
    ]

    # Left Panel: Volatility Smile sigma(M, tau) = 0.15 + 0.08*tau + 0.65*(M-1)^2
    pl_x, pl_y, pl_w, pl_h = 68, 58, 345, 295
    lines.append(f'<rect x="{pl_x}" y="{pl_y}" width="{pl_w}" height="{pl_h}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>')
    lines.append(f'<text x="{pl_x + pl_w/2}" y="{pl_y - 10}" text-anchor="middle" font-size="12.5" font-weight="bold" fill="#1e293b">(a) Volatility Smile: σ(M, τ) = σ₀ + ατ + β(M - 1)²</text>')

    # Axes ticks Left Panel: M in [0.65, 1.35], sigma in [0.14, 0.42]
    m_min, m_max = 0.65, 1.35
    s_min, s_max = 0.14, 0.42
    for i in range(6):
        m_val = m_min + i * (m_max - m_min) / 5.0
        gx = pl_x + (m_val - m_min) / (m_max - m_min) * pl_w
        lines.append(f'<line x1="{gx:.1f}" y1="{pl_y}" x2="{gx:.1f}" y2="{pl_y+pl_h}" stroke="#e2e8f0" stroke-dasharray="3,3"/>')
        lines.append(f'<text x="{gx:.1f}" y="{pl_y+pl_h+18}" text-anchor="middle" font-size="10.5" fill="#475569">{m_val:.2f}</text>')

    for i in range(6):
        s_val = s_min + i * (s_max - s_min) / 5.0
        gy = pl_y + pl_h - (s_val - s_min) / (s_max - s_min) * pl_h
        lines.append(f'<line x1="{pl_x}" y1="{gy:.1f}" x2="{pl_x+pl_w}" y2="{gy:.1f}" stroke="#e2e8f0" stroke-dasharray="3,3"/>')
        lines.append(f'<text x="{pl_x-8}" y="{gy+4:.1f}" text-anchor="end" font-size="10.5" fill="#475569">{s_val:.2f}</text>')

    # Scatter sample points on Left Panel
    for rec in records[:220]:
        m_v = float(rec["moneyness_M"])
        s_v = float(rec["volatility_sigma"])
        tau_v = float(rec["time_to_maturity_tau"])
        cx = pl_x + (m_v - m_min) / (m_max - m_min) * pl_w
        cy = pl_y + pl_h - (min(max(s_v, s_min), s_max) - s_min) / (s_max - s_min) * pl_h
        col = "#93c5fd" if tau_v < 0.7 else ("#60a5fa" if tau_v < 1.4 else "#1d4ed8")
        lines.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="2.2" fill="{col}" opacity="0.55"/>')

    # Theoretical Smile curves for tau = 0.2, 1.0, 1.8
    tau_curves = [(0.20, "#059669", "τ = 0.2 yr (Short)"), (1.00, "#d97706", "τ = 1.0 yr (Medium)"), (1.85, "#dc2626", "τ = 1.85 yr (Long)")]
    for idx_c, (tau_val, col, lbl) in enumerate(tau_curves):
        pts = []
        for step in range(41):
            mv = m_min + step * (m_max - m_min) / 40.0
            sv = 0.15 + 0.08 * tau_val + 0.65 * ((mv - 1.0) ** 2)
            px = pl_x + (mv - m_min) / (m_max - m_min) * pl_w
            py = pl_y + pl_h - (sv - s_min) / (s_max - s_min) * pl_h
            pts.append(f"{px:.1f},{py:.1f}")
        lines.append(f'<polyline fill="none" stroke="{col}" stroke-width="2.5" points="{" ".join(pts)}"/>')
        ly = pl_y + 18 + idx_c * 18
        lines.append(f'<line x1="{pl_x+95}" y1="{ly}" x2="{pl_x+115}" y2="{ly}" stroke="{col}" stroke-width="2.5"/>')
        lines.append(f'<text x="{pl_x+120}" y="{ly+4}" font-size="10" font-weight="600" fill="#1e293b">{lbl}</text>')

    lines.append(f'<text x="{pl_x + pl_w/2}" y="{h-22}" text-anchor="middle" font-size="12" font-weight="600" fill="#1e293b">Moneyness M = K / S</text>')
    lines.append(f'<text x="18" y="{pl_y + pl_h/2}" text-anchor="middle" font-size="12" font-weight="600" fill="#1e293b" transform="rotate(-90, 18, {pl_y + pl_h/2})">Implied Volatility σ(M, τ)</text>')

    # Right Panel: Option Price C/S vs Moneyness M
    pr_x, pr_y, pr_w, pr_h = 495, 58, 355, 295
    lines.append(f'<rect x="{pr_x}" y="{pr_y}" width="{pr_w}" height="{pr_h}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>')
    lines.append(f'<text x="{pr_x + pr_w/2}" y="{pr_y - 10}" text-anchor="middle" font-size="12.5" font-weight="bold" fill="#1e293b">(b) Normalized Option Price (C / S) Across Moneyness</text>')

    c_min, c_max = 0.0, 0.45
    for i in range(6):
        m_val = m_min + i * (m_max - m_min) / 5.0
        gx = pr_x + (m_val - m_min) / (m_max - m_min) * pr_w
        lines.append(f'<line x1="{gx:.1f}" y1="{pr_y}" x2="{gx:.1f}" y2="{pr_y+pr_h}" stroke="#e2e8f0" stroke-dasharray="3,3"/>')
        lines.append(f'<text x="{gx:.1f}" y="{pr_y+pr_h+18}" text-anchor="middle" font-size="10.5" fill="#475569">{m_val:.2f}</text>')

    for i in range(6):
        c_val = c_min + i * (c_max - c_min) / 5.0
        gy = pr_y + pr_h - (c_val - c_min) / (c_max - c_min) * pr_h
        lines.append(f'<line x1="{pr_x}" y1="{gy:.1f}" x2="{pr_x+pr_w}" y2="{gy:.1f}" stroke="#e2e8f0" stroke-dasharray="3,3"/>')
        lines.append(f'<text x="{pr_x-8}" y="{gy+4:.1f}" text-anchor="end" font-size="10.5" fill="#475569">{c_val:.2f}</text>')

    # Scatter points colored by regime
    regime_colors = {"ITM": "#059669", "ATM": "#2563eb", "OTM": "#dc2626"}
    for rec in records[:260]:
        m_v = float(rec["moneyness_M"])
        c_v = float(rec["call_price_norm"])
        reg = rec["moneyness_regime"]
        cx = pr_x + (m_v - m_min) / (m_max - m_min) * pr_w
        cy = pr_y + pr_h - (min(max(c_v, c_min), c_max) - c_min) / (c_max - c_min) * pr_h
        lines.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="2.3" fill="{regime_colors[reg]}" opacity="0.62"/>')

    # Intrinsic payoff bound max(1 - M, 0)
    intr_pts = []
    for step in range(41):
        mv = m_min + step * (m_max - m_min) / 40.0
        iv = max(0.0, 1.0 - mv)
        px = pr_x + (mv - m_min) / (m_max - m_min) * pr_w
        py = pr_y + pr_h - (iv - c_min) / (c_max - c_min) * pr_h
        intr_pts.append(f"{px:.1f},{py:.1f}")
    lines.append(f'<polyline fill="none" stroke="#334155" stroke-width="2.0" stroke-dasharray="6,4" points="{" ".join(intr_pts)}"/>')

    # Legend Right Panel
    leg_items = [
        ("#059669", "ITM Regime (M &lt; 0.95)"),
        ("#2563eb", "ATM Regime (0.95 ≤ M ≤ 1.05)"),
        ("#dc2626", "OTM Regime (M &gt; 1.05)"),
        ("#334155", "Intrinsic Lower Bound max(1-M, 0)"),
    ]
    for idx_l, (col, text_l) in enumerate(leg_items):
        ly = pr_y + 18 + idx_l * 18
        lx = pr_x + 135
        if idx_l < 3:
            lines.append(f'<circle cx="{lx+8}" cy="{ly-3}" r="4" fill="{col}"/>')
        else:
            lines.append(f'<line x1="{lx}" y1="{ly-3}" x2="{lx+16}" y2="{ly-3}" stroke="{col}" stroke-width="2" stroke-dasharray="4,2"/>')
        lines.append(f'<text x="{lx+22}" y="{ly+1}" font-size="10" font-weight="600" fill="#1e293b">{text_l}</text>')

    lines.append(f'<text x="{pr_x + pr_w/2}" y="{h-22}" text-anchor="middle" font-size="12" font-weight="600" fill="#1e293b">Moneyness M = K / S</text>')
    lines.append(f'<text x="{pr_x - 42}" y="{pr_y + pr_h/2}" text-anchor="middle" font-size="12" font-weight="600" fill="#1e293b" transform="rotate(-90, {pr_x - 42}, {pr_y + pr_h/2})">Normalized Call Price (C / S)</text>')

    lines.append("</svg>")
    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    return filepath


def _plot_eda_correlation_heatmap(eda_summary: Dict[str, Any], filepath: str) -> str:
    """SVG Correlation Matrix Heatmap across Market Inputs, Option Price, and Greeks."""
    w, h = 760, 470
    corr = eda_summary["correlation_matrix"]
    cols = [
        "moneyness_M",
        "time_to_maturity_tau",
        "risk_free_rate_r",
        "volatility_sigma",
        "call_price_norm",
        "delta",
        "vega",
    ]
    short_names = [
        "Moneyness (M)",
        "Maturity (τ)",
        "Risk-Free (r)",
        "Volatility (σ)",
        "Option Price (C/S)",
        "Delta (Δ)",
        "Vega (ν)",
    ]

    pad_l, pad_t = 155, 65
    cell_w, cell_h = 74, 46

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" style="background:#ffffff;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif;">',
        f'<text x="{w/2}" y="30" text-anchor="middle" font-size="15.5" font-weight="bold" fill="#0f172a">Exploratory Data Analysis: Feature &amp; Greek Pearson Correlation Matrix</text>',
    ]

    for i, r_key in enumerate(cols):
        y = pad_t + i * cell_h
        lines.append(f'<text x="{pad_l - 10}" y="{y + cell_h/2 + 4}" text-anchor="end" font-size="11.5" font-weight="600" fill="#1e293b">{short_names[i]}</text>')
        for j, c_key in enumerate(cols):
            x = pad_l + j * cell_w
            val = corr[r_key][c_key]
            # Diverging blue-white-red color mapping
            if val >= 0:
                inten = min(1.0, val)
                r_c = int(255 - inten * (255 - 30))
                g_c = int(255 - inten * (255 - 90))
                b_c = int(255 - inten * (255 - 185))
            else:
                inten = min(1.0, abs(val))
                r_c = int(255 - inten * (255 - 205))
                g_c = int(255 - inten * (255 - 45))
                b_c = int(255 - inten * (255 - 55))
            fill_hex = f"#{r_c:02x}{g_c:02x}{b_c:02x}"
            txt_col = "#ffffff" if abs(val) > 0.52 else "#0f172a"
            lines.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{fill_hex}" stroke="#ffffff" stroke-width="1.5"/>')
            lines.append(f'<text x="{x + cell_w/2}" y="{y + cell_h/2 + 4}" text-anchor="middle" font-size="11" font-weight="bold" fill="{txt_col}">{val:+.2f}</text>')

    for j, name in enumerate(short_names):
        x = pad_l + j * cell_w + cell_w / 2
        y = pad_t + len(cols) * cell_h + 18
        lines.append(f'<text x="{x}" y="{y}" text-anchor="end" font-size="11" font-weight="600" fill="#1e293b" transform="rotate(-30, {x}, {y})">{name}</text>')

    lines.append("</svg>")
    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    return filepath


def _plot_framework_metrics_comparison(
    framework_results: List[Dict[str, Any]], filepath: str
) -> str:
    """
    Two-panel SVG bar chart comparing all frameworks on:
      (a) Test RMSE & MAE (Lower is better)
      (b) Test R^2 Score (%) & Regime Classification Accuracy (%) (Higher is better)
    """
    w, h = 880, 430
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" style="background:#ffffff;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif;">',
        f'<text x="{w/2}" y="28" text-anchor="middle" font-size="15.5" font-weight="bold" fill="#0f172a">Efficacy Benchmark: Option Pricing Error (RMSE / MAE) &amp; R² Across Frameworks</text>',
    ]

    # Left subplot: RMSE and MAE
    pl_x, pl_y, pl_w, pl_h = 65, 62, 360, 275
    lines.append(f'<rect x="{pl_x}" y="{pl_y}" width="{pl_w}" height="{pl_h}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>')
    lines.append(f'<text x="{pl_x + pl_w/2}" y="{pl_y - 10}" text-anchor="middle" font-size="12.5" font-weight="bold" fill="#1e293b">(a) Pricing Error: Test RMSE &amp; MAE (Lower ↓)</text>')

    max_err = max(r["rmse"] for r in framework_results) * 1.22
    for i in range(6):
        val = i * max_err / 5.0
        gy = pl_y + pl_h - (val / max_err) * pl_h
        lines.append(f'<line x1="{pl_x}" y1="{gy:.1f}" x2="{pl_x+pl_w}" y2="{gy:.1f}" stroke="#e2e8f0" stroke-dasharray="3,3"/>')
        lines.append(f'<text x="{pl_x-6}" y="{gy+4:.1f}" text-anchor="end" font-size="10" fill="#475569">{val:.4f}</text>')

    n_f = len(framework_results)
    group_w = pl_w / n_f
    bar_w = group_w * 0.34

    for idx, r in enumerate(framework_results):
        gx = pl_x + idx * group_w + group_w * 0.14
        rmse_v = r["rmse"]
        mae_v = r["mae"]
        h_rmse = (rmse_v / max_err) * pl_h
        h_mae = (mae_v / max_err) * pl_h
        col = COLORS.get(r["id"], "#2563eb")

        # RMSE bar
        lines.append(f'<rect x="{gx:.1f}" y="{pl_y+pl_h-h_rmse:.1f}" width="{bar_w:.1f}" height="{h_rmse:.1f}" fill="{col}" rx="2"/>')
        lines.append(f'<text x="{gx+bar_w/2:.1f}" y="{pl_y+pl_h-h_rmse-5:.1f}" text-anchor="middle" font-size="8.8" font-weight="bold" fill="#0f172a">{rmse_v:.4f}</text>')

        # MAE bar
        lines.append(f'<rect x="{gx+bar_w+3:.1f}" y="{pl_y+pl_h-h_mae:.1f}" width="{bar_w:.1f}" height="{h_mae:.1f}" fill="{col}" opacity="0.55" rx="2"/>')
        lines.append(f'<text x="{gx+1.5*bar_w+3:.1f}" y="{pl_y+pl_h-h_mae-5:.1f}" text-anchor="middle" font-size="8.5" fill="#475569">{mae_v:.4f}</text>')

        lbl_x = gx + bar_w
        lbl_y = pl_y + pl_h + 16
        lines.append(f'<text x="{lbl_x:.1f}" y="{lbl_y:.1f}" text-anchor="end" font-size="10" font-weight="600" fill="#1e293b" transform="rotate(-28, {lbl_x:.1f}, {lbl_y:.1f})">{r["name"]}</text>')

    # Right subplot: R2 (%) and Classification Accuracy (%)
    pr_x, pr_y, pr_w, pr_h = 490, 62, 360, 275
    lines.append(f'<rect x="{pr_x}" y="{pr_y}" width="{pr_w}" height="{pr_h}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>')
    lines.append(f'<text x="{pr_x + pr_w/2}" y="{pr_y - 10}" text-anchor="middle" font-size="12.5" font-weight="bold" fill="#1e293b">(b) Goodness-of-Fit R² (%) &amp; Regime Acc (%) (Higher ↑)</text>')

    min_pct, max_pct = 90.0, 100.5
    for i in range(6):
        val = min_pct + i * (100.0 - min_pct) / 5.0
        gy = pr_y + pr_h - ((val - min_pct) / (max_pct - min_pct)) * pr_h
        lines.append(f'<line x1="{pr_x}" y1="{gy:.1f}" x2="{pr_x+pr_w}" y2="{gy:.1f}" stroke="#e2e8f0" stroke-dasharray="3,3"/>')
        lines.append(f'<text x="{pr_x-6}" y="{gy+4:.1f}" text-anchor="end" font-size="10" fill="#475569">{val:.1f}%</text>')

    for idx, r in enumerate(framework_results):
        gx = pr_x + idx * group_w + group_w * 0.14
        r2_pct = max(min_pct, r["r2"] * 100.0)
        cls_pct = max(min_pct, r["classification_accuracy_pct"])
        h_r2 = ((r2_pct - min_pct) / (max_pct - min_pct)) * pr_h
        h_cls = ((cls_pct - min_pct) / (max_pct - min_pct)) * pr_h
        col = COLORS.get(r["id"], "#2563eb")

        lines.append(f'<rect x="{gx:.1f}" y="{pr_y+pr_h-h_r2:.1f}" width="{bar_w:.1f}" height="{h_r2:.1f}" fill="{col}" rx="2"/>')
        lines.append(f'<text x="{gx+bar_w/2:.1f}" y="{pr_y+pl_h-h_r2-5:.1f}" text-anchor="middle" font-size="8.8" font-weight="bold" fill="#0f172a">{r2_pct:.2f}%</text>')

        lines.append(f'<rect x="{gx+bar_w+3:.1f}" y="{pr_y+pr_h-h_cls:.1f}" width="{bar_w:.1f}" height="{h_cls:.1f}" fill="#334155" opacity="0.65" rx="2"/>')
        lines.append(f'<text x="{gx+1.5*bar_w+3:.1f}" y="{pr_y+pl_h-h_cls-5:.1f}" text-anchor="middle" font-size="8.5" fill="#334155">{cls_pct:.1f}%</text>')

        lbl_x = gx + bar_w
        lbl_y = pr_y + pr_h + 16
        lines.append(f'<text x="{lbl_x:.1f}" y="{lbl_y:.1f}" text-anchor="end" font-size="10" font-weight="600" fill="#1e293b" transform="rotate(-28, {lbl_x:.1f}, {lbl_y:.1f})">{r["name"]}</text>')

    lines.append("</svg>")
    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    return filepath


def _plot_simplicity_transparency_efficacy(
    framework_results: List[Dict[str, Any]], filepath: str
) -> str:
    """
    Grouped Multi-Criterion Evaluation Chart comparing all frameworks across:
      - Simplicity (API conciseness, boilerplate LOC, ease of tuning)
      - Transparency (Feature importance, Sparsemax masks, Greek interpretability)
      - Efficacy (R2 accuracy, training speed, no-arbitrage adherence)
    """
    w, h = 840, 410
    pad_l, pad_r, pad_t, pad_b = 70, 40, 68, 85
    pw = w - pad_l - pad_r
    ph = h - pad_t - pad_b

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" style="background:#ffffff;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif;">',
        f'<text x="{w/2}" y="28" text-anchor="middle" font-size="15.5" font-weight="bold" fill="#0f172a">Qualitative &amp; Architectural Triad: Simplicity, Transparency, and Efficacy (Score / 10)</text>',
        f'<rect x="{pad_l}" y="{pad_t}" width="{pw}" height="{ph}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>',
    ]

    for i in range(6):
        val = i * 2.0
        gy = pad_t + ph - (val / 10.5) * ph
        lines.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{pad_l+pw}" y2="{gy:.1f}" stroke="#e2e8f0" stroke-dasharray="3,3"/>')
        lines.append(f'<text x="{pad_l-10}" y="{gy+4:.1f}" text-anchor="end" font-size="10.5" fill="#475569">{val:.0f}</text>')

    n_f = len(framework_results)
    group_w = pw / n_f
    bar_w = group_w * 0.22

    crit_colors = [
        ("simplicity_score", "#2563eb", "Simplicity (API &amp; Setup)"),
        ("transparency_score", "#7c3aed", "Transparency (Interpretability)"),
        ("efficacy_score", "#059669", "Efficacy (Accuracy &amp; Speed)"),
    ]

    # Legend at top
    for idx_c, (_, col, lbl) in enumerate(crit_colors):
        lx = pad_l + 65 + idx_c * 225
        ly = pad_t - 16
        lines.append(f'<rect x="{lx}" y="{ly-10}" width="14" height="10" fill="{col}" rx="2"/>')
        lines.append(f'<text x="{lx+20}" y="{ly-1}" font-size="11" font-weight="bold" fill="#1e293b">{lbl}</text>')

    for idx, r in enumerate(framework_results):
        gx = pad_l + idx * group_w + group_w * 0.13
        for c_i, (key, col, _) in enumerate(crit_colors):
            sc = float(r[key])
            bh = (sc / 10.5) * ph
            bx = gx + c_i * (bar_w + 3)
            by = pad_t + ph - bh
            lines.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="{col}" rx="2"/>')
            lines.append(f'<text x="{bx+bar_w/2:.1f}" y="{by-5:.1f}" text-anchor="middle" font-size="9.5" font-weight="bold" fill="#0f172a">{sc:.1f}</text>')

        cx_label = gx + 1.5 * bar_w
        lines.append(f'<text x="{cx_label:.1f}" y="{pad_t+ph+22}" text-anchor="middle" font-size="11" font-weight="bold" fill="#0f172a">{r["name"]}</text>')
        lines.append(f'<text x="{cx_label:.1f}" y="{pad_t+ph+38}" text-anchor="middle" font-size="9.5" fill="#475569">Boilerplate: {r["loc_boilerplate"]} LOC</text>')

    lines.append("</svg>")
    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    return filepath


def _plot_feature_importance_and_convergence(
    framework_results: List[Dict[str, Any]],
    feature_importance_data: Dict[str, Any],
    filepath: str,
) -> str:
    """
    Two-panel SVG:
      Left Panel:  Transparency Comparison — XGBoost Split Gain vs TabNet Sparsemax Attention Mask
      Right Panel: Training Convergence Curves (Validation RMSE vs Epoch across Frameworks)
    """
    w, h = 880, 420
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" style="background:#ffffff;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif;">',
        f'<text x="{w/2}" y="28" text-anchor="middle" font-size="15.5" font-weight="bold" fill="#0f172a">Transparency (Feature Attribution) &amp; Validation Convergence Across Frameworks</text>',
    ]

    # Left Panel: Horizontal Feature Importance (XGBoost Gain vs TabNet Sparsemax)
    pl_x, pl_y, pl_w, pl_h = 148, 60, 290, 295
    lines.append(f'<rect x="{pl_x}" y="{pl_y}" width="{pl_w}" height="{pl_h}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>')
    lines.append(f'<text x="{pl_x + pl_w/2 - 20}" y="{pl_y - 10}" text-anchor="middle" font-size="12" font-weight="bold" fill="#1e293b">(a) Transparency: XGBoost Gain vs TabNet Mask</text>')

    feat_names = feature_importance_data["feature_names"]
    xgb_imp = feature_importance_data["xgboost_gain_pct"]
    tab_imp = feature_importance_data["tabnet_attention_pct"]
    max_imp = max(max(xgb_imp), max(tab_imp)) * 1.18

    row_h = pl_h / len(feat_names)
    for i, fname in enumerate(feat_names):
        ry = pl_y + i * row_h + 6
        bh = row_h * 0.34
        w_xgb = (xgb_imp[i] / max_imp) * pl_w
        w_tab = (tab_imp[i] / max_imp) * pl_w

        lines.append(f'<text x="{pl_x - 8}" y="{ry + bh + 3:.1f}" text-anchor="end" font-size="10" font-weight="600" fill="#1e293b">{fname}</text>')
        lines.append(f'<rect x="{pl_x}" y="{ry:.1f}" width="{w_xgb:.1f}" height="{bh:.1f}" fill="#d97706" rx="2"/>')
        lines.append(f'<text x="{pl_x + w_xgb + 4:.1f}" y="{ry + bh - 1:.1f}" font-size="8.5" font-weight="bold" fill="#b45309">{xgb_imp[i]:.1f}%</text>')

        lines.append(f'<rect x="{pl_x}" y="{ry + bh + 2:.1f}" width="{w_tab:.1f}" height="{bh:.1f}" fill="#7c3aed" rx="2"/>')
        lines.append(f'<text x="{pl_x + w_tab + 4:.1f}" y="{ry + 2*bh + 1:.1f}" font-size="8.5" font-weight="bold" fill="#6d28d9">{tab_imp[i]:.1f}%</text>')

    # Legend for Left Panel
    lines.append(f'<rect x="{pl_x + 120}" y="{pl_y + pl_h - 38}" width="12" height="8" fill="#d97706"/>')
    lines.append(f'<text x="{pl_x + 136}" y="{pl_y + pl_h - 31}" font-size="9.5" font-weight="bold" fill="#1e293b">XGBoost Split Gain (%)</text>')
    lines.append(f'<rect x="{pl_x + 120}" y="{pl_y + pl_h - 22}" width="12" height="8" fill="#7c3aed"/>')
    lines.append(f'<text x="{pl_x + 136}" y="{pl_y + pl_h - 15}" font-size="9.5" font-weight="bold" fill="#1e293b">TabNet Sparsemax Mask (%)</text>')

    # Right Panel: Validation RMSE Convergence Curves
    pr_x, pr_y, pr_w, pr_h = 505, 60, 345, 295
    lines.append(f'<rect x="{pr_x}" y="{pr_y}" width="{pr_w}" height="{pr_h}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>')
    lines.append(f'<text x="{pr_x + pr_w/2}" y="{pr_y - 10}" text-anchor="middle" font-size="12" font-weight="bold" fill="#1e293b">(b) Validation RMSE Convergence Across Epochs</text>')

    y_max = 0.09
    for i in range(6):
        val = i * y_max / 5.0
        gy = pr_y + pr_h - (val / y_max) * pr_h
        lines.append(f'<line x1="{pr_x}" y1="{gy:.1f}" x2="{pr_x+pr_w}" y2="{gy:.1f}" stroke="#e2e8f0" stroke-dasharray="3,3"/>')
        lines.append(f'<text x="{pr_x-6}" y="{gy+4:.1f}" text-anchor="end" font-size="10" fill="#475569">{val:.3f}</text>')

    for idx_r, r in enumerate(framework_results):
        curve = r["val_loss_curve"][:40]
        n_ep = len(curve)
        col = COLORS.get(r["id"], "#2563eb")
        pts = []
        for ep_i, v in enumerate(curve):
            px = pr_x + (ep_i / max(1, n_ep - 1)) * pr_w
            py = pr_y + pr_h - (min(v, y_max) / y_max) * pr_h
            pts.append(f"{px:.1f},{py:.1f}")
        lines.append(f'<polyline fill="none" stroke="{col}" stroke-width="2.2" points="{" ".join(pts)}"/>')

        ly = pr_y + 18 + idx_r * 18
        lx = pr_x + 150
        lines.append(f'<line x1="{lx}" y1="{ly-3}" x2="{lx+18}" y2="{ly-3}" stroke="{col}" stroke-width="2.5"/>')
        lines.append(f'<text x="{lx+24}" y="{ly+1}" font-size="9.5" font-weight="600" fill="#1e293b">{r["name"]} (RMSE={r["rmse"]:.4f})</text>')

    lines.append(f'<text x="{pr_x + pr_w/2}" y="{h-22}" text-anchor="middle" font-size="11.5" font-weight="600" fill="#1e293b">Training Iteration / Epoch</text>')
    lines.append("</svg>")
    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    return filepath
