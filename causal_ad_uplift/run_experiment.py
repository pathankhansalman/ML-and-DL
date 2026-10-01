"""
End-to-End Ad Campaign Uplift Experiment.

Compares:
  1. Naive CTR Model (Standard Supervised ML)
  2. T-Learner (Causal Two-Model)
  3. Class Variable Transformation (Lai's Causal Target)
  4. Random Baseline

Outputs:
  - Decile Breakdown (Persuadables vs Sure Things in each decile)
  - Qini Curve Comparison Plot (saved as qini_curves.png)
  - Financial ROI Analysis (Net profit / loss from ad spend)
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_generator import generate_ad_campaign_data
from models import NaiveCTRModel, TLearner, ClassTransformationLearner, RandomBaseline
from metrics import compute_qini_curve, compute_decile_table, simulate_campaign_roi


def main():
    print("=" * 70)
    print(" [CAUSAL UPLIFT IN ONLINE ADVERTISING: THE A/B TEST EXPERIMENT]")
    print("=" * 70)

    # 1. Generate Dataset
    print("\n[1/4] Generating 50,000 simulated ad impressions...")
    df, gt = generate_ad_campaign_data(n_samples=50000, random_seed=42)

    # Split into Train (60%) and Test (40%)
    split_idx = int(0.60 * len(df))
    feature_cols = [
        "days_since_last_visit",
        "time_on_site_mins",
        "past_spend",
        "cart_value",
        "device_mobile",
        "ad_fatigue",
    ]

    X_train = df.iloc[:split_idx][feature_cols]
    t_train = df.iloc[:split_idx]["treatment"].values
    y_train = df.iloc[:split_idx]["conversion"].values

    X_test = df.iloc[split_idx:][feature_cols]
    t_test = df.iloc[split_idx:]["treatment"].values
    y_test = df.iloc[split_idx:]["conversion"].values
    gt_test = gt.iloc[split_idx:]

    print(f"      Train Set: {len(X_train)} impressions | Test Set: {len(X_test)} impressions")

    # 2. Train Models
    print("\n[2/4] Fitting models...")
    models = {
        "Naive CTR (Non-Causal)": NaiveCTRModel(),
        "T-Learner (Causal 2-Model)": TLearner(),
        "Class Transformation (Lai's Causal)": ClassTransformationLearner(),
        "Random Baseline": RandomBaseline(),
    }

    test_predictions = {}
    for name, model in models.items():
        print(f"      Training: {name}...")
        model.fit(X_train, t_train, y_train)
        test_predictions[name] = model.predict_uplift(X_test)

    # 3. Decile Analysis Comparison
    print("\n[3/4] Comparing Decile 1 (Top 10% Targeted Audience):")
    print("-" * 70)

    for name in ["Naive CTR (Non-Causal)", "Class Transformation (Lai's Causal)"]:
        decile_df = compute_decile_table(
            test_predictions[name],
            t_test,
            y_test,
            ground_truth_segments=gt_test["segment"].values,
            n_bins=10,
        )
        top_decile = decile_df.iloc[0]
        bottom_decile = decile_df.iloc[-1]
        print(f"\nModel: {name}")
        print(f"  -> Top 10% (Decile 1) Targeted:")
        print(f"     - Net Uplift:        +{top_decile['Net_Uplift%']}%")
        print(f"     - Persuadables:      {top_decile.get('Persuadables%', 'N/A')}% of bucket")
        print(f"     - Sure Things:       {top_decile.get('Sure_Things%', 'N/A')}% of bucket")
        print(f"     - Sleeping Dogs:     {top_decile.get('Sleeping_Dogs%', 'N/A')}% of bucket")
        print(f"  -> Bottom 10% (Decile 10) Excluded:")
        print(f"     - Net Uplift:        {bottom_decile['Net_Uplift%']}%")
        print(f"     - Sleeping Dogs:     {bottom_decile.get('Sleeping_Dogs%', 'N/A')}% of bucket")

    # 4. Financial ROI Simulation
    print("\n" + "=" * 70)
    print(" FINANCIAL ROI: AD SPEND VS INCREMENTAL REVENUE")
    print(" (Assumes: $0.50 per ad impression, $40.00 value per incremental conversion)")
    print("=" * 70)

    for name in ["Naive CTR (Non-Causal)", "Class Transformation (Lai's Causal)"]:
        roi_df = simulate_campaign_roi(
            test_predictions[name],
            t_test,
            y_test,
            cost_per_ad=0.50,
            revenue_per_conversion=40.00,
            targeting_cutoffs=(0.10, 0.20, 0.30, 0.50, 1.00),
        )
        print(f"\nModel: {name}")
        print(roi_df[["Targeting_Top%", "Ad_Spend_$", "Incremental_Conv", "Net_Profit_$", "ROI%"]].to_string(index=False))

    # 5. Qini Curve Plotting
    print("\n[4/4] Generating Qini Curves...")
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = {
        "Naive CTR (Non-Causal)": "#E63946",
        "T-Learner (Causal 2-Model)": "#457B9D",
        "Class Transformation (Lai's Causal)": "#2A9D8F",
        "Random Baseline": "#6C757D",
    }

    for name, scores in test_predictions.items():
        fracs, qini, score = compute_qini_curve(scores, t_test, y_test)
        style = "--" if name == "Random Baseline" else "-"
        width = 2.5 if "Class" in name or "Naive" in name else 1.8
        ax.plot(
            fracs * 100,
            qini,
            label=f"{name} (Qini Area: {score:.1f})",
            color=colors[name],
            linestyle=style,
            linewidth=width,
        )

    ax.set_title("Qini Curve: Cumulative Incremental Conversions vs % Audience Targeted", fontsize=12, fontweight="bold")
    ax.set_xlabel("% of Audience Targeted (Sorted by Model)", fontsize=11)
    ax.set_ylabel("Incremental Conversions (Lift * Size)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, fontsize=10)

    output_plot_path = os.path.join(os.path.dirname(__file__), "qini_curves.png")
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    plt.close()
    print(f"      Plot successfully saved to: {output_plot_path}")

    print("\n" + "=" * 70)
    print(" [SUCCESS] Experiment Finished Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
