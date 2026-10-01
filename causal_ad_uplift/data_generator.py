"""
Realistic Ad-Tech Impression Data Generator with Ground-Truth Causal Segments.

Simulates user features and potential outcomes:
  - Y(0): Conversion if NO ad shown (holdout)
  - Y(1): Conversion if ad SHOWN (treatment)
  - True ITE (Uplift) = Y(1) - Y(0)

Includes 4 latent user archetypes:
  1. Sure Things   (high baseline, tau ~ 0)
  2. Persuadables  (low/mod baseline, tau > 0)
  3. Lost Causes   (zero baseline, tau ~ 0)
  4. Sleeping Dogs (high ad fatigue, tau < 0)
"""

import numpy as np
import pandas as pd
from typing import Tuple


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))


def generate_ad_campaign_data(
    n_samples: int = 50000,
    treatment_fraction: float = 0.5,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates synthetic A/B test data for an online ad campaign.

    Returns:
        df: Observed data (Features X, Treatment T, Observed Conversion Y)
        df_ground_truth: Counterfactuals Y0, Y1, True CATE, and Segment Label
    """
    rng = np.random.RandomState(random_seed)

    # 1. Observable User Features
    # Recency: days since last visit (1 to 30 days)
    days_since_last_visit = rng.exponential(scale=7.0, size=n_samples).clip(0.5, 30.0)

    # Past engagement: minutes spent on site in past 30 days
    time_on_site_mins = rng.gamma(shape=2.0, scale=8.0, size=n_samples)

    # Historical spend ($0 to $500)
    past_spend = rng.exponential(scale=45.0, size=n_samples)

    # Cart value ($0 to $200)
    cart_value = rng.choice([0, 25, 60, 120, 200], size=n_samples, p=[0.55, 0.20, 0.15, 0.07, 0.03])

    # Mobile device (0 = Desktop, 1 = Mobile)
    device_mobile = rng.binomial(n=1, p=0.68, size=n_samples)

    # Ad fatigue index (0 to 1, how many ads the user has seen this week)
    ad_fatigue = rng.beta(a=2.0, b=5.0, size=n_samples)

    # 2. Baseline Conversion Probability (without any ad: Y(0))
    # Loyal users with high past spend and recent visits naturally convert without ads!
    base_logit = (
        -3.5
        + 0.02 * past_spend
        + 0.02 * cart_value
        + 0.05 * time_on_site_mins
        - 0.10 * days_since_last_visit
        - 0.20 * device_mobile
    )
    p_y0 = sigmoid(base_logit)

    # 3. Causal Treatment Effect (Uplift: tau = P(Y(1)=1) - P(Y(0)=1))
    # Who responds positively?
    # - Moderate interest (days_since_last_visit between 2 and 10)
    # - Has items in cart
    # - Low ad fatigue
    # Who responds negatively (Sleeping Dogs)?
    # - High ad fatigue (> 0.6) + frequent visits (intrusive ad causes churn/annoyance)
    uplift_raw = (
        0.12 * (cart_value > 0).astype(float)
        + 0.08 * ((days_since_last_visit >= 2) & (days_since_last_visit <= 10)).astype(float)
        - 0.25 * (ad_fatigue > 0.55).astype(float)
        + 0.02 * (time_on_site_mins > 10).astype(float)
        - 0.05
    )

    # Scale uplift so probabilities stay in [0, 1]
    p_y1 = np.clip(p_y0 + uplift_raw, 0.001, 0.999)
    true_tau = p_y1 - p_y0

    # 4. Generate Potential Outcomes Y(0) and Y(1)
    # Using correlated Bernoulli sampling
    u = rng.uniform(0, 1, size=n_samples)
    y0 = (u < p_y0).astype(int)
    y1 = (u < p_y1).astype(int)

    # 5. Randomized Treatment Assignment (A/B Test coin flip)
    # T = 1: Ad served, T = 0: Holdout
    t = rng.binomial(n=1, p=treatment_fraction, size=n_samples)

    # 6. Observed Outcome (The Fundamental Problem of Causal Inference!)
    y = np.where(t == 1, y1, y0)

    # 7. Categorize into the 4 Ground-Truth Archetypes
    segment = np.empty(n_samples, dtype=object)
    segment[(y0 == 1) & (y1 == 1)] = "Sure Thing"
    segment[(y0 == 0) & (y1 == 0)] = "Lost Cause"
    segment[(y0 == 0) & (y1 == 1)] = "Persuadable"
    segment[(y0 == 1) & (y1 == 0)] = "Sleeping Dog"

    features_df = pd.DataFrame({
        "days_since_last_visit": np.round(days_since_last_visit, 1),
        "time_on_site_mins": np.round(time_on_site_mins, 1),
        "past_spend": np.round(past_spend, 2),
        "cart_value": cart_value,
        "device_mobile": device_mobile,
        "ad_fatigue": np.round(ad_fatigue, 3),
        "treatment": t,
        "conversion": y,
    })

    ground_truth_df = pd.DataFrame({
        "p_y0": p_y0,
        "p_y1": p_y1,
        "true_tau": true_tau,
        "y0": y0,
        "y1": y1,
        "segment": segment,
    })

    return features_df, ground_truth_df


if __name__ == "__main__":
    df, gt = generate_ad_campaign_data(n_samples=20000, random_seed=42)
    print("=== Simulated Ad Campaign Dataset ===")
    print(df.head())
    print("\n=== Ground-Truth Segment Distribution ===")
    print(gt["segment"].value_counts(normalize=True) * 100)
    print("\nMean True Uplift by Segment:")
    print(gt.groupby("segment")["true_tau"].mean())
