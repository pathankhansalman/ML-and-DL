"""
Day 4: Pearl's 3-Step Counterfactual Algorithm & Unit-Level Counterfactuals
===========================================================================

This script implements:
1. A Structural Causal Model (SCM) with latent exogenous noise U.
2. Observational Sampling (Rung 1).
3. Population-level Interventional Sampling (Rung 2: do(X = x*)).
4. Pearl's 3-Step Counterfactual Engine (Rung 3):
     - Step 1 (Abduction): Deduce latent noise U for a specific individual from factual observations.
     - Step 2 (Action): Perform graph surgery do(X = x*) on structural equations.
     - Step 3 (Prediction): Re-evaluate the outcome under the modified model using the abducted U.
5. Contrast: Why an individual counterfactual Y_{x*}(u) differs from the population average E[Y | do(X = x*)].
"""

import numpy as np


class LinearGaussianSCM:
    """
    SCM Definition:
      U_Z ~ N(0, 1)          # Unobserved background lifestyle / baseline health
      U_X ~ N(0, 1)          # Unobserved treatment inclination noise
      U_Y ~ N(0, 0.5^2)      # Unobserved biological outcome noise

      Z = U_Z                # Confounder (e.g., socioeconomic status)
      X = 0.8 * Z + U_X      # Treatment (e.g., drug dosage or training hours)
      Y = 2.0 * X + 1.5 * Z + U_Y  # Outcome (e.g., recovery speed or performance)
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def sample_observational(self, n: int = 5000):
        """Rung 1: Sample from joint observational distribution P(Z, X, Y)."""
        u_z = self.rng.normal(0, 1.0, size=n)
        u_x = self.rng.normal(0, 1.0, size=n)
        u_y = self.rng.normal(0, 0.5, size=n)

        z = u_z
        x = 0.8 * z + u_x
        y = 2.0 * x + 1.5 * z + u_y

        return {"Z": z, "X": x, "Y": y, "U_Z": u_z, "U_X": u_x, "U_Y": u_y}

    def simulate_intervention(self, do_x_val: float, n: int = 5000):
        """
        Rung 2: Simulate population-level intervention do(X = do_x_val).
        Graph surgery severs the link from Z -> X. New individuals are drawn from P(U).
        """
        u_z = self.rng.normal(0, 1.0, size=n)
        u_y = self.rng.normal(0, 0.5, size=n)

        z = u_z
        x = np.full(n, do_x_val)  # X is externally set by intervention
        y = 2.0 * x + 1.5 * z + u_y  # Downstream structural mechanism intact

        return {"Z": z, "X": x, "Y": y}

    def counterfactual_query(self, observed_evidence: dict, counterfactual_x: float) -> dict:
        """
        Rung 3: Pearl's 3-Step Counterfactual Algorithm for a specific individual.

        Query: "Given that this individual had X = x_obs, Y = y_obs, and Z = z_obs,
                what would Y have been had they received X = counterfactual_x?"
        """
        x_obs = observed_evidence["X"]
        y_obs = observed_evidence["Y"]
        z_obs = observed_evidence["Z"]

        # =====================================================================
        # Step 1: ABDUCTION
        # Infer the specific exogenous noise vector (u_z, u_x, u_y) that generated
        # this exact factual individual.
        # From structural equations:
        #   Z = U_Z              => U_Z = Z
        #   X = 0.8 * Z + U_X    => U_X = X - 0.8 * Z
        #   Y = 2.0*X + 1.5*Z + U_Y => U_Y = Y - 2.0*X - 1.5*Z
        # =====================================================================
        abducted_u_z = z_obs
        abducted_u_x = x_obs - 0.8 * z_obs
        abducted_u_y = y_obs - 2.0 * x_obs - 1.5 * z_obs

        # =====================================================================
        # Step 2: ACTION
        # Modify the structural model: sever incoming edges into X, set X = counterfactual_x.
        # =====================================================================
        cf_x = counterfactual_x

        # =====================================================================
        # Step 3: PREDICTION
        # Re-compute the outcome using the modified model and the abducted noise.
        # =====================================================================
        cf_z = abducted_u_z  # Z is upstream of X, unchanged
        cf_y = 2.0 * cf_x + 1.5 * cf_z + abducted_u_y

        return {
            "factual": {"Z": z_obs, "X": x_obs, "Y": y_obs},
            "abducted_noise": {"U_Z": abducted_u_z, "U_X": abducted_u_x, "U_Y": abducted_u_y},
            "counterfactual": {"X*": cf_x, "Y*": cf_y},
        }


def main():
    scm = LinearGaussianSCM(seed=42)

    print("=" * 70)
    print("DAY 4: PEARL'S 3-STEP COUNTERFACTUAL ENGINE")
    print("=" * 70)

    # 1. Sample an actual observed patient (Alice)
    obs_data = scm.sample_observational(n=100)
    idx = 12  # Pick unit #12 (Alice)
    alice_obs = {
        "Z": float(obs_data["Z"][idx]),
        "X": float(obs_data["X"][idx]),
        "Y": float(obs_data["Y"][idx]),
    }

    print("\n[FACTUAL WORLD: ALICE]")
    print(f"  Observed Confounder (Z) : {alice_obs['Z']:.3f}")
    print(f"  Observed Treatment  (X) : {alice_obs['X']:.3f}")
    print(f"  Observed Outcome    (Y) : {alice_obs['Y']:.3f}")

    # 2. Run Counterfactual Query: What if Alice had received X = 0.0?
    counterfactual_treatment = 0.0
    result = scm.counterfactual_query(alice_obs, counterfactual_x=counterfactual_treatment)

    print("\n[STEP 1: ABDUCTION]")
    u = result["abducted_noise"]
    print(f"  Abduced Latent U_Z : {u['U_Z']:.3f}")
    print(f"  Abduced Latent U_X : {u['U_X']:.3f}")
    print(f"  Abduced Latent U_Y : {u['U_Y']:.3f} (Alice's unique individual biology)")

    print("\n[STEP 2: ACTION]")
    print(f"  Intervention applied: do(X = {counterfactual_treatment})")

    print("\n[STEP 3: PREDICTION]")
    cf_y = result["counterfactual"]["Y*"]
    print(f"  Counterfactual Outcome Y_{{X={counterfactual_treatment}}}(Alice) : {cf_y:.3f}")

    factual_y = alice_obs["Y"]
    unit_treatment_effect = factual_y - cf_y
    print(f"  Individual Causal Effect for Alice (Y - Y*): {unit_treatment_effect:.3f}")

    # 3. Contrast with Population Average under Intervention (Rung 2)
    interv_data = scm.simulate_intervention(do_x_val=counterfactual_treatment, n=10000)
    pop_avg_y_under_interv = float(np.mean(interv_data["Y"]))

    print("\n" + "-" * 70)
    print("[CRITICAL COMPARISON: RUNG 2 (INTERVENTION) vs. RUNG 3 (COUNTERFACTUAL)]")
    print("-" * 70)
    print(f"  Population Mean under do(X = 0)   : E[Y | do(X = 0)] = {pop_avg_y_under_interv:.3f}")
    print(f"  Alice's Specific Counterfactual    : Y_{{X=0}}(Alice)  = {cf_y:.3f}")
    print(
        "\n  -> WHY THEY DIFFER: The population mean draws random individuals from P(U).\n"
        "     Alice's counterfactual PRESERVES her specific unobserved noise (U_Z, U_Y).\n"
        "     Her high baseline health (U_Z) remains fixed in her parallel universe!"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
