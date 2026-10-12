"""
Causal Bandit Implementation (LinUCB and Thompson Sampling).

Optimizes incremental lift rather than raw conversion:
  BreakEven = ad_cost / conversion_value
  Action = 1 (Show Ad) if Lift > BreakEven else 0 (Hold Out)
"""

import numpy as np


class CausalBandit:
    def __init__(
        self,
        n_features: int,
        ad_cost: float = 0.05,
        conversion_value: float = 1.0,
        strategy: str = "thompson",
        alpha: float = 1.0,
    ):
        self.d = n_features
        self.ad_cost = ad_cost
        self.conversion_value = conversion_value
        self.break_even = ad_cost / conversion_value
        self.strategy = strategy
        self.alpha = alpha

        # Model 0: Baseline (No Ad)
        self.A0 = np.eye(self.d)
        self.b0 = np.zeros(self.d)

        # Model 1: Treated (Ad)
        self.A1 = np.eye(self.d)
        self.b1 = np.zeros(self.d)

    def choose_action(self, x: np.ndarray) -> int:
        """Decides whether to show the ad (1) or hold out (0)."""
        inv_A0 = np.linalg.inv(self.A0)
        inv_A1 = np.linalg.inv(self.A1)

        mean_0 = inv_A0 @ self.b0
        mean_1 = inv_A1 @ self.b1

        if self.strategy == "thompson":
            # 1. Sample weights from Gaussian posterior
            theta_0 = np.random.multivariate_normal(mean_0, inv_A0)
            theta_1 = np.random.multivariate_normal(mean_1, inv_A1)
            lift = np.dot(x, theta_1) - np.dot(x, theta_0)

        elif self.strategy == "linucb":
            # 2. Mean lift + joint uncertainty bonus
            mean_lift = np.dot(x, mean_1) - np.dot(x, mean_0)
            uncertainty = self.alpha * np.sqrt(x @ inv_A1 @ x + x @ inv_A0 @ x)
            lift = mean_lift + uncertainty

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Show ad only if lift beats the break-even hurdle
        return int(lift > self.break_even)

    def update(self, x: np.ndarray, action: int, y: int):
        """Updates only the model for the action that was actually taken."""
        if action == 1:
            self.A1 += np.outer(x, x)
            self.b1 += y * x
        else:
            self.A0 += np.outer(x, x)
            self.b0 += y * x
