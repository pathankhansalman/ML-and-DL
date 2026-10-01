"""
Unit tests for Causal Uplift Modeling pipeline.
"""

import numpy as np
import pandas as pd
from data_generator import generate_ad_campaign_data
from models import NaiveCTRModel, TLearner, ClassTransformationLearner, RandomBaseline
from metrics import compute_qini_curve, compute_decile_table, simulate_campaign_roi


def test_data_generator():
    df, gt = generate_ad_campaign_data(n_samples=1000, random_seed=42)
    assert len(df) == 1000
    assert len(gt) == 1000
    assert set(gt["segment"].unique()).issubset({"Sure Thing", "Lost Cause", "Persuadable", "Sleeping Dog"})
    assert set(df["treatment"].unique()).issubset({0, 1})
    assert set(df["conversion"].unique()).issubset({0, 1})
    print("[PASS] test_data_generator")


def test_models_fit_and_predict():
    df, _ = generate_ad_campaign_data(n_samples=2000, random_seed=42)
    feature_cols = ["days_since_last_visit", "time_on_site_mins", "past_spend", "cart_value", "device_mobile", "ad_fatigue"]
    X = df[feature_cols]
    t = df["treatment"].values
    y = df["conversion"].values

    models = [NaiveCTRModel(), TLearner(), ClassTransformationLearner(), RandomBaseline()]
    for model in models:
        model.fit(X, t, y)
        preds = model.predict_uplift(X)
        assert len(preds) == len(X)
        assert not np.isnan(preds).any()
    print("[PASS] test_models_fit_and_predict")


def test_metrics():
    uplift = np.linspace(1, 0, 500)
    treatment = np.random.binomial(1, 0.5, 500)
    y = np.random.binomial(1, 0.1, 500)

    fracs, qini, score = compute_qini_curve(uplift, treatment, y)
    assert len(fracs) == 100
    assert len(qini) == 100

    decile_df = compute_decile_table(uplift, treatment, y, n_bins=5)
    assert len(decile_df) == 5

    roi_df = simulate_campaign_roi(uplift, treatment, y)
    assert len(roi_df) > 0
    print("[PASS] test_metrics")


if __name__ == "__main__":
    test_data_generator()
    test_models_fit_and_predict()
    test_metrics()
    print("\nAll unit tests passed successfully!")
