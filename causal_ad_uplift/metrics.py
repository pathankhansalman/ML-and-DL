"""
Causal Uplift Metrics & Ad Campaign ROI Evaluation.

Implements:
  1. compute_qini_curve: Cumulative incremental conversions across targeting fractions.
  2. compute_qini_score: Qini area relative to random targeting.
  3. compute_decile_table: Net lift and user archetype breakdown per decile.
  4. simulate_campaign_roi: Dollars spent vs incremental dollars earned.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


def compute_qini_curve(
    uplift_scores: np.ndarray,
    treatment: np.ndarray,
    y: np.ndarray,
    n_points: int = 100,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Computes the Radcliffe Qini curve:
      Q(f) = n_{t, 1}(f) - n_{c, 1}(f) * (N_t / N_c)
      
    Args:
        uplift_scores: Model predicted uplift for each user
        treatment: 1 for ad served, 0 for holdout
        y: Observed conversion (1 or 0)
        n_points: Number of evaluation percentiles (default: 100)
        
    Returns:
        fractions: Array of targeted population fractions [0.0 to 1.0]
        qini_curve: Cumulative incremental conversions at each fraction
        qini_score: Area between model curve and random diagonal
    """
    N = len(y)
    Nt = np.sum(treatment == 1)
    Nc = np.sum(treatment == 0)
    ratio = Nt / Nc if Nc > 0 else 1.0

    # Sort descending by predicted uplift
    order = np.argsort(-uplift_scores)
    t_sorted = treatment[order]
    y_sorted = y[order]

    # Cumulative conversions
    cum_yt = np.cumsum((t_sorted == 1) & (y_sorted == 1))
    cum_yc = np.cumsum((t_sorted == 0) & (y_sorted == 1))

    # Evaluate at regular grid
    indices = np.linspace(0, N - 1, n_points, dtype=int)
    fractions = np.linspace(0.0, 1.0, n_points)

    qini_curve = cum_yt[indices] - cum_yc[indices] * ratio

    # Random baseline curve connects (0, 0) to (1.0, qini_curve[-1])
    random_curve = fractions * qini_curve[-1]

    # Qini score is normalized area under the curve minus random
    integrate_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    qini_score = float(integrate_fn(qini_curve - random_curve, fractions))

    return fractions, qini_curve, qini_score


def compute_decile_table(
    uplift_scores: np.ndarray,
    treatment: np.ndarray,
    y: np.ndarray,
    ground_truth_segments: Optional[np.ndarray] = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Groups users into n_bins (deciles) ranked from highest to lowest predicted uplift.
    Computes empirical uplift and archetype composition in each decile.
    """
    df = pd.DataFrame({
        "score": uplift_scores,
        "treatment": treatment,
        "conversion": y,
    })
    if ground_truth_segments is not None:
        df["segment"] = ground_truth_segments

    # Assign decile (1 = highest predicted uplift, 10 = lowest)
    df["decile"] = pd.qcut(-df["score"].rank(method="first"), q=n_bins, labels=False) + 1

    records = []
    for d, group in df.groupby("decile"):
        t_group = group[group["treatment"] == 1]
        c_group = group[group["treatment"] == 0]

        n_t = len(t_group)
        n_c = len(c_group)

        cr_t = t_group["conversion"].mean() if n_t > 0 else 0.0
        cr_c = c_group["conversion"].mean() if n_c > 0 else 0.0
        net_uplift = cr_t - cr_c

        # Incremental conversions scaled to total size
        incremental_conversions = (net_uplift * n_t) if n_t > 0 else 0.0

        row = {
            "Decile": d,
            "Total_Users": len(group),
            "Treated_N": n_t,
            "Control_N": n_c,
            "Treated_CR%": round(cr_t * 100, 2),
            "Control_CR%": round(cr_c * 100, 2),
            "Net_Uplift%": round(net_uplift * 100, 2),
            "Est_Incremental_Conv": round(incremental_conversions, 1),
        }

        if ground_truth_segments is not None:
            seg_counts = group["segment"].value_counts(normalize=True) * 100
            row["Persuadables%"] = round(seg_counts.get("Persuadable", 0.0), 1)
            row["Sure_Things%"] = round(seg_counts.get("Sure Thing", 0.0), 1)
            row["Lost_Causes%"] = round(seg_counts.get("Lost Cause", 0.0), 1)
            row["Sleeping_Dogs%"] = round(seg_counts.get("Sleeping Dog", 0.0), 1)

        records.append(row)

    return pd.DataFrame(records)


def simulate_campaign_roi(
    uplift_scores: np.ndarray,
    treatment: np.ndarray,
    y: np.ndarray,
    cost_per_ad: float = 0.50,
    revenue_per_conversion: float = 40.00,
    targeting_cutoffs: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 0.8, 1.0),
) -> pd.DataFrame:
    """
    Simulates marketing economics for targeting top k% of users.
    
    Net Profit = (Incremental Conversions * Revenue) - (Treated Users * Cost)
    """
    N = len(y)
    Nt = np.sum(treatment == 1)
    Nc = np.sum(treatment == 0)
    ratio = Nt / Nc if Nc > 0 else 1.0

    order = np.argsort(-uplift_scores)
    t_sorted = treatment[order]
    y_sorted = y[order]

    cum_yt = np.cumsum((t_sorted == 1) & (y_sorted == 1))
    cum_yc = np.cumsum((t_sorted == 0) & (y_sorted == 1))
    cum_nt = np.cumsum(t_sorted == 1)

    results = []
    for cutoff in targeting_cutoffs:
        idx = int(cutoff * N) - 1
        idx = max(0, min(idx, N - 1))

        treated_in_top = cum_nt[idx]
        incremental_conv = cum_yt[idx] - (cum_yc[idx] * ratio)

        ad_spend = treated_in_top * cost_per_ad
        incremental_revenue = incremental_conv * revenue_per_conversion
        net_profit = incremental_revenue - ad_spend
        roi_pct = (net_profit / ad_spend * 100) if ad_spend > 0 else 0.0

        results.append({
            "Targeting_Top%": int(cutoff * 100),
            "Ads_Served": treated_in_top,
            "Ad_Spend_$": round(ad_spend, 2),
            "Incremental_Conv": round(incremental_conv, 1),
            "Incremental_Revenue_$": round(incremental_revenue, 2),
            "Net_Profit_$": round(net_profit, 2),
            "ROI%": round(roi_pct, 1),
        })

    return pd.DataFrame(results)
