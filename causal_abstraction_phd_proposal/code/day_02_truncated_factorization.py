"""
Day 2 Coding Lab: The do(.) Operator & Truncated Factorization Formula

Goal:
1. Define a confounded SCM: Z -> X, Z -> Y, X -> Y.
2. Verify Truncated Factorization: P(Y | do(X=x)) = sum_z P(Y | X=x, Z=z) * P(Z=z).
3. Demonstrate that Truncated Factorization mathematically matches Graph Surgery simulation.
"""

import numpy as np
import pandas as pd


def generate_scm_data(num_samples=100000, seed=42):
    """
    Observational Data Generation from SCM:
    Z: Confounder ~ Bernoulli(p=0.4) (e.g., High Risk Patient = 1)
    X: Treatment ~ Bernoulli(p = sigmoid(1.5 * Z - 0.5)) (High risk patients get treatment more often)
    Y: Recovery ~ Normal(mu = 50 + 20 * X - 15 * Z, sigma = 5)
    """
    np.random.seed(seed)
    
    # Exogenous noise
    U_Z = np.random.uniform(0, 1, size=num_samples)
    U_X = np.random.uniform(0, 1, size=num_samples)
    U_Y = np.random.normal(0, 5, size=num_samples)
    
    # Endogenous variables (Observational)
    Z = (U_Z < 0.4).astype(int)
    prob_X = 1 / (1 + np.exp(-(1.5 * Z - 0.5)))
    X = (U_X < prob_X).astype(int)
    Y = 50 + 20 * X - 15 * Z + U_Y
    
    return pd.DataFrame({'Z': Z, 'X': X, 'Y': Y})


def simulate_graph_surgery_do(do_x, num_samples=100000, seed=42):
    """
    Interventional Data via Physical Graph Surgery do(X = do_x):
    Removes edge Z -> X. Overwrites X := do_x.
    """
    np.random.seed(seed)
    
    U_Z = np.random.uniform(0, 1, size=num_samples)
    U_Y = np.random.normal(0, 5, size=num_samples)
    
    Z = (U_Z < 0.4).astype(int)
    
    # GRAPH SURGERY: Force X = do_x (ignoring Z)
    X = np.full(num_samples, do_x)
    Y = 50 + 20 * X - 15 * Z + U_Y
    
    return pd.DataFrame({'Z': Z, 'X': X, 'Y': Y})


def compute_truncated_factorization_expectation(df_obs, do_x_value):
    """
    Computes E[Y | do(X = do_x)] using ONLY observational data via Truncated Factorization:
    E[Y | do(X = x)] = sum_z E[Y | X = x, Z = z] * P(Z = z)
    """
    # 1. P(Z = z) from observational population
    p_z1 = (df_obs['Z'] == 1).mean()
    p_z0 = 1.0 - p_z1
    
    # 2. E[Y | X = x, Z = z] conditional means from observational data
    e_y_given_x_z0 = df_obs[(df_obs['X'] == do_x_value) & (df_obs['Z'] == 0)]['Y'].mean()
    e_y_given_x_z1 = df_obs[(df_obs['X'] == do_x_value) & (df_obs['Z'] == 1)]['Y'].mean()
    
    # 3. Truncated Factorization / Adjustment Sum
    e_y_do = (e_y_given_x_z0 * p_z0) + (e_y_given_x_z1 * p_z1)
    return e_y_do, (e_y_given_x_z0, e_y_given_x_z1, p_z0, p_z1)


if __name__ == '__main__':
    print("=" * 70)
    print("DAY 2 CODING LAB: TRUNCATED FACTORIZATION vs GRAPH SURGERY")
    print("=" * 70)
    
    # 1. Generate Observational Data
    df_obs = generate_scm_data()
    
    # Naive observational estimation: E[Y | X = 1] - E[Y | X = 0]
    naive_e_y1 = df_obs[df_obs['X'] == 1]['Y'].mean()
    naive_e_y0 = df_obs[df_obs['X'] == 0]['Y'].mean()
    naive_ace = naive_e_y1 - naive_e_y0
    
    print("\n1. NAIVE OBSERVATIONAL ESTIMATE (P(Y | X=x)):")
    print(f"   E[Y | X = 1] = {naive_e_y1:.3f}")
    print(f"   E[Y | X = 0] = {naive_e_y0:.3f}")
    print(f"   Naive Observational ACE = {naive_ace:.3f}  (Biased due to confounding!)")
    
    # 2. Physical Graph Surgery do(X = 1) and do(X = 0)
    df_do1 = simulate_graph_surgery_do(do_x=1)
    df_do0 = simulate_graph_surgery_do(do_x=0)
    
    true_do1 = df_do1['Y'].mean()
    true_do0 = df_do0['Y'].mean()
    true_ace = true_do1 - true_do0
    
    print("\n2. TRUE INTERVENTIONAL SIMULATION (Graph Surgery do(X=x)):")
    print(f"   E[Y | do(X = 1)] = {true_do1:.3f}")
    print(f"   E[Y | do(X = 0)] = {true_do0:.3f}")
    print(f"   True Interventional ACE = {true_ace:.3f}  (Matches true structural effect +20.0!)")
    
    # 3. Truncated Factorization Formula from Observational Data
    trunc_do1, details1 = compute_truncated_factorization_expectation(df_obs, do_x_value=1)
    trunc_do0, details0 = compute_truncated_factorization_expectation(df_obs, do_x_value=0)
    trunc_ace = trunc_do1 - trunc_do0
    
    print("\n3. TRUNCATED FACTORIZATION FORMULA (From Observational Data Only):")
    print(f"   Formula: sum_z E[Y | X=x, Z=z] * P(Z=z)")
    print(f"   Estimated E[Y | do(X = 1)] via Truncated Factorization = {trunc_do1:.3f}")
    print(f"   Estimated E[Y | do(X = 0)] via Truncated Factorization = {trunc_do0:.3f}")
    print(f"   Calculated Interventional ACE = {trunc_ace:.3f}")
    
    print("\n" + "=" * 70)
    print("VERIFICATION RESULT:")
    print(f"   | True Graph Surgery ACE ({true_ace:.3f}) - Truncated Factorization ACE ({trunc_ace:.3f}) | < 0.05?")
    if abs(true_ace - trunc_ace) < 0.05:
        print("   [SUCCESS] Truncated Factorization perfectly recovers true interventional ACE!")
    else:
        print("   [FAILURE] Discrepancy detected.")
    print("=" * 70)
