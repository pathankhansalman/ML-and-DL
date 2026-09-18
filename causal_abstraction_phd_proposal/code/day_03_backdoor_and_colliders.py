"""
Day 3 Coding Lab: d-Separation, Backdoor Criterion, and the Danger of Bad Controls

Objective:
Demonstrate empirically:
1. Ground Truth Causal Effect via Graph Surgery: do(X = 1) vs do(X = 0).
2. Naive Observational Association: Biased by confounder Z.
3. Correct Backdoor Adjustment: Conditioning on {Z} blocks the backdoor path and recovers true effect.
4. Collider Bias (Berkson's Paradox): Conditioning on collider C introduces artificial bias.
5. Over-controlling (Mediator Blocking): Conditioning on mediator M blocks the causal mechanism.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def generate_day3_scm_data(num_samples=100000, seed=42):
    """
    SCM Structure:
      Z (Confounder): Z ~ Bernoulli(0.5)
      X (Treatment): X ~ Bernoulli(sigmoid(1.5 * Z - 0.5))
      M (Mediator): M = 2.0 * X + N(0, 1)
      Y (Outcome): Y = 3.0 * M - 4.0 * Z + N(0, 1)
         -> Total Causal Effect of X on Y is: dY/dM * dM/dX = 3.0 * 2.0 = +6.0
      C (Collider): C = 2.5 * X + 1.5 * Y + N(0, 1)  (Common effect of X and Y)
    """
    np.random.seed(seed)

    # Exogenous noise
    U_Z = np.random.uniform(0, 1, size=num_samples)
    U_X = np.random.uniform(0, 1, size=num_samples)
    U_M = np.random.normal(0, 1, size=num_samples)
    U_Y = np.random.normal(0, 1, size=num_samples)
    U_C = np.random.normal(0, 1, size=num_samples)

    # Endogenous variables
    Z = (U_Z < 0.5).astype(int)
    prob_X = 1 / (1 + np.exp(-(1.5 * Z - 0.5)))
    X = (U_X < prob_X).astype(int)
    M = 2.0 * X + U_M
    Y = 3.0 * M - 4.0 * Z + U_Y
    C = 2.5 * X + 1.5 * Y + U_C

    return pd.DataFrame({'Z': Z, 'X': X, 'M': M, 'Y': Y, 'C': C})


def simulate_graph_surgery_effect(num_samples=100000, seed=42):
    """
    Computes true causal effect E[Y | do(X=1)] - E[Y | do(X=0)] via physical graph surgery:
    Sever incoming edge into X, setting X := 1 and X := 0.
    """
    np.random.seed(seed)
    U_Z = np.random.uniform(0, 1, size=num_samples)
    U_M = np.random.normal(0, 1, size=num_samples)
    U_Y = np.random.normal(0, 1, size=num_samples)

    Z = (U_Z < 0.5).astype(int)

    # do(X = 1)
    X1 = np.ones(num_samples)
    M1 = 2.0 * X1 + U_M
    Y1 = 3.0 * M1 - 4.0 * Z + U_Y

    # do(X = 0)
    X0 = np.zeros(num_samples)
    M0 = 2.0 * X0 + U_M
    Y0 = 3.0 * M0 - 4.0 * Z + U_Y

    true_ace = np.mean(Y1) - np.mean(Y0)
    return true_ace


def run_regression(df, formula, variable_of_interest='X'):
    """Helper to fit OLS and extract coefficient & standard error for variable_of_interest."""
    import statsmodels.formula.api as smf
    model = smf.ols(formula, data=df).fit()
    coef = model.params[variable_of_interest]
    se = model.bse[variable_of_interest]
    p_val = model.pvalues[variable_of_interest]
    return coef, se, p_val


if __name__ == '__main__':
    print("=" * 75)
    print("DAY 3 CODING LAB: d-SEPARATION, BACKDOOR CRITERION, & BAD CONTROLS")
    print("=" * 75)

    # 1. Physical Graph Surgery Ground Truth
    true_ace = simulate_graph_surgery_effect()
    print(f"\n[GROUND TRUTH] Physical Graph Surgery ACE = {true_ace:.3f} (Theoretical: +6.000)")

    # Generate Observational Dataset
    df = generate_day3_scm_data()

    # 2. Naive Observational Regression: Y ~ X (Unadjusted)
    coef_naive, se_naive, _ = run_regression(df, "Y ~ X")
    print("\n1. NAIVE OBSERVATIONAL REGRESSION:  Y ~ X")
    print(f"   Estimated X Coeff = {coef_naive:.3f} (SE: {se_naive:.3f})")
    print("   -> BIASED: Confounder Z opens the backdoor path X <- Z -> Y.")

    # 3. Valid Backdoor Adjustment: Y ~ X + Z (Condition on Backdoor Set {Z})
    coef_adj, se_adj, _ = run_regression(df, "Y ~ X + Z")
    print("\n2. VALID BACKDOOR ADJUSTMENT:       Y ~ X + Z")
    print(f"   Estimated X Coeff = {coef_adj:.3f} (SE: {se_adj:.3f})")
    print("   -> UNBIASED: Conditioning on Z blocks the backdoor path. Recovers true ACE (+6.0)!")

    # 4. Bad Control #1: Collider Conditioning: Y ~ X + Z + C
    coef_coll, se_coll, _ = run_regression(df, "Y ~ X + Z + C")
    print("\n3. BAD CONTROL #1 (COLLIDER BIAS): Y ~ X + Z + C")
    print(f"   Estimated X Coeff = {coef_coll:.3f} (SE: {se_coll:.3f})")
    print("   -> SEVERELY BIASED: C is a collider (X -> C <- Y). Conditioning on C OPENS")
    print("      a spurious non-causal path, inducing Berkson's Paradox!")

    # 5. Bad Control #2: Overcontrolling / Mediator: Y ~ X + Z + M
    coef_med, se_med, _ = run_regression(df, "Y ~ X + Z + M")
    print("\n4. BAD CONTROL #2 (MEDIATOR BLOCK): Y ~ X + Z + M")
    print(f"   Estimated X Coeff = {coef_med:.3f} (SE: {se_med:.3f})")
    print("   -> ZEROED OUT: M is a mediator (X -> M -> Y). Conditioning on M blocks the")
    print("      actual transmission of the causal effect, leaving only direct effect (0.0)!")

    print("\n" + "=" * 75)
    print("SUMMARY OF EMPIRICAL VERIFICATION:")
    print(f"   Ground Truth Total Effect: {true_ace:.3f}")
    print(f"   Backdoor Adjusted:         {coef_adj:.3f}  [Matches Ground Truth? {abs(coef_adj - true_ace) < 0.05}]")
    print(f"   Collider Conditioned:      {coef_coll:.3f}  [Matches Ground Truth? {abs(coef_coll - true_ace) < 0.05}]")
    print(f"   Mediator Conditioned:      {coef_med:.3f}  [Matches Ground Truth? {abs(coef_med - true_ace) < 0.05}]")
    print("=" * 75)
