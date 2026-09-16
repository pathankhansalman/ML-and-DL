"""
Day 1 Coding Lab: Building a Structural Causal Model (SCM) from scratch in Python.

Scenario: Medical Treatment & Disease Severity
Variables:
- A: Age
- D: Disease Severity
- T: Treatment Assigned
- Y: Health Recovery Outcome
"""

import numpy as np
import pandas as pd


def generate_observational_data(num_samples=10000, seed=42):
    """
    Simulates observational data from the true SCM:
    1. Age (A) ~ Uniform(20, 80)
    2. Disease Severity (D) = 0.5 * Age + U_D
    3. Treatment (T) = 1 if (1.2 * D - 0.4 * Age + U_T > 25) else 0  (Confounded!)
    4. Recovery (Y) = 100 - 0.8 * D + 15 * T + U_Y  (True Causal Effect of Treatment = +15)
    """
    np.random.seed(seed)
    
    # Exogenous noise
    U_A = np.random.uniform(20, 80, size=num_samples)
    U_D = np.random.normal(0, 5, size=num_samples)
    U_T = np.random.normal(0, 5, size=num_samples)
    U_Y = np.random.normal(0, 5, size=num_samples)
    
    # Endogenous variables (Observational SCM)
    A = U_A
    D = 0.5 * A + U_D
    T = (1.2 * D - 0.4 * A + U_T > 25).astype(int)
    Y = 100 - 0.8 * D + 15 * T + U_Y
    
    return pd.DataFrame({'Age': A, 'Disease_Severity': D, 'Treatment': T, 'Recovery': Y})


def generate_interventional_data(do_treatment_value, num_samples=10000, seed=42):
    """
    Simulates interventional data under Graph Surgery: do(Treatment = do_treatment_value).
    Structural equation for T is OVERWRITTEN by a constant!
    """
    np.random.seed(seed)
    
    # Exogenous noise
    U_A = np.random.uniform(20, 80, size=num_samples)
    U_D = np.random.normal(0, 5, size=num_samples)
    U_Y = np.random.normal(0, 5, size=num_samples)
    
    # Endogenous variables (Interventional SCM under Graph Surgery)
    A = U_A
    D = 0.5 * A + U_D
    
    # INTERVENTION: Force T = do_treatment_value (Graph Surgery collapses input edges into T)
    T = np.full(num_samples, do_treatment_value)
    
    # Health recovery given forced treatment
    Y = 100 - 0.8 * D + 15 * T + U_Y
    
    return pd.DataFrame({'Age': A, 'Disease_Severity': D, 'Treatment': T, 'Recovery': Y})


if __name__ == '__main__':
    print("=" * 60)
    print("DAY 1 CODING LAB: OBSERVATIONAL VS INTERVENTIONAL SCM")
    print("=" * 60)
    
    # 1. Observational Analysis
    df_obs = generate_observational_data()
    obs_mean_treated = df_obs[df_obs['Treatment'] == 1]['Recovery'].mean()
    obs_mean_control = df_obs[df_obs['Treatment'] == 0]['Recovery'].mean()
    obs_diff = obs_mean_treated - obs_mean_control
    
    print("\n1. OBSERVATIONAL ESTIMATE (Naive ML Correlation):")
    print(f"   Mean Recovery (Observed Treated T=1): {obs_mean_treated:.2f}")
    print(f"   Mean Recovery (Observed Control T=0): {obs_mean_control:.2f}")
    print(f"   Naively Estimated Treatment Effect (Treated - Control): {obs_diff:.2f}")
    print("   -> Notice: Naive comparison under-estimates treatment benefit due to confounding!")
    
    # 2. Interventional Analysis (True Causal Effect)
    df_do1 = generate_interventional_data(do_treatment_value=1)
    df_do0 = generate_interventional_data(do_treatment_value=0)
    
    true_causal_effect = df_do1['Recovery'].mean() - df_do0['Recovery'].mean()
    
    print("\n2. INTERVENTIONAL ESTIMATE (Graph Surgery do(T=1) vs do(T=0)):")
    print(f"   E[Recovery | do(T=1)]: {df_do1['Recovery'].mean():.2f}")
    print(f"   E[Recovery | do(T=0)]: {df_do0['Recovery'].mean():.2f}")
    print(f"   TRUE Average Causal Effect (ACE): {true_causal_effect:.2f}")
    print("   -> Notice: True ACE is exactly +15.00, matching the structural coefficient!")
    print("=" * 60)
