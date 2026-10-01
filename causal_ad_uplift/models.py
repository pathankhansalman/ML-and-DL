"""
Causal Uplift Models for Online Advertising.

Implements:
  1. NaiveCTRModel: Standard supervised ML (optimizes P(Y=1 | X, T=1))
  2. TLearner: Two-Model approach (mu_1(X) - mu_0(X))
  3. ClassTransformationLearner: Lai's method (direct CATE via Z-target)
  4. RandomBaseline: Unbiased random targeting benchmark
"""

import numpy as np
import pandas as pd
from typing import Optional
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier


class BaseUpliftModel:
    """Abstract interface for all uplift models."""
    def fit(self, X: pd.DataFrame, treatment: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class NaiveCTRModel(BaseUpliftModel):
    """
    Standard Industry Approach (Non-Causal):
    Trains ONLY on exposed users (treatment = 1).
    Ranks users by P(Y=1 | X, T=1).
    
    Trap: Ranks 'Sure Things' highest because their absolute conversion
    probability is very high, even though ad impact is zero.
    """
    def __init__(self, base_estimator=None):
        self.estimator = base_estimator or GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.08, random_state=42
        )

    def fit(self, X: pd.DataFrame, treatment: np.ndarray, y: np.ndarray):
        # Filter only treated impressions
        treated_mask = (treatment == 1)
        X_treated = X[treated_mask]
        y_treated = y[treated_mask]

        self.estimator.fit(X_treated, y_treated)
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        # Returns absolute conversion probability under treatment
        # (This is NOT true uplift, which is why it fails!)
        return self.estimator.predict_proba(X)[:, 1]


class TLearner(BaseUpliftModel):
    """
    Two-Model Approach (Causal Meta-Learner):
    Trains two independent models:
      mu_1(X): P(Y=1 | X, T=1)
      mu_0(X): P(Y=1 | X, T=0)
    
    Predicted Uplift = mu_1(X) - mu_0(X).
    """
    def __init__(self, base_estimator=None):
        base = base_estimator or GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.08, random_state=42
        )
        self.model_t1 = clone(base)
        self.model_t0 = clone(base)

    def fit(self, X: pd.DataFrame, treatment: np.ndarray, y: np.ndarray):
        t1_mask = (treatment == 1)
        t0_mask = (treatment == 0)

        # Fit model on treatment group
        self.model_t1.fit(X[t1_mask], y[t1_mask])

        # Fit model on control group
        self.model_t0.fit(X[t0_mask], y[t0_mask])
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        p1 = self.model_t1.predict_proba(X)[:, 1]
        p0 = self.model_t0.predict_proba(X)[:, 1]
        return p1 - p0


class ClassTransformationLearner(BaseUpliftModel):
    """
    Class Variable Transformation (Lai's Method / Jaskowski & Jaroszewicz 2012):
    
    Transforms the binary outcome using treatment assignment:
      Z_i = Y_i * T_i + (1 - Y_i) * (1 - T_i)
    
    Under balanced A/B testing (P(T=1) = 0.5):
      P(Z=1 | X) = (tau(X) + 1) / 2
      tau(X) = 2 * P(Z=1 | X) - 1
      
    Trains ONE single model to predict Z. Directly optimizes for causal uplift!
    """
    def __init__(self, base_estimator=None):
        self.estimator = base_estimator or GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.08, random_state=42
        )

    def fit(self, X: pd.DataFrame, treatment: np.ndarray, y: np.ndarray):
        # Create transformed target Z
        z = y * treatment + (1 - y) * (1 - treatment)
        self.estimator.fit(X, z)
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        p_z1 = self.estimator.predict_proba(X)[:, 1]
        # Map probability back to causal uplift
        return 2.0 * p_z1 - 1.0


class RandomBaseline(BaseUpliftModel):
    """Baseline: Un-targeted random ad delivery."""
    def fit(self, X: pd.DataFrame, treatment: np.ndarray, y: np.ndarray):
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        rng = np.random.RandomState(42)
        return rng.uniform(size=len(X))
