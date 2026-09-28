"""
Day 5 Coding Lab: Minimal Differentiable SCM in PyTorch.

DAG Structure:
    Z (Confounder) -> X (Treatment) -> Y (Outcome)
    Z --------------> Y
"""

import torch
import torch.nn as nn


class DifferentiableSCM(nn.Module):
    """
    Parametric SCM with explicit exogenous noise inputs and do-intervention support.
    """
    def __init__(self):
        super().__init__()
        # Structural mechanisms: fi(PA_i, U_i)
        # 1. Z = f_Z(U_Z)
        self.f_Z = nn.Linear(1, 1)
        # 2. X = f_X(Z, U_X)
        self.f_X = nn.Linear(2, 1)
        # 3. Y = f_Y(Z, X, U_Y)
        self.f_Y = nn.Linear(3, 1)

    def forward(self, u_dict: dict, do: dict = None) -> dict:
        """
        Forward pass with Graph Surgery support:
        - do: dict mapping variable names to forced values, e.g. {'X': torch.tensor([1.0])}
        """
        do = do or {}

        # 1. Evaluate Z
        if 'Z' in do:
            z = do['Z']
        else:
            z = self.f_Z(u_dict['U_Z'])

        # 2. Evaluate X (Graph Surgery cuts incoming edge Z -> X if X in do)
        if 'X' in do:
            x = do['X']
        else:
            x = self.f_X(torch.cat([z, u_dict['U_X']], dim=-1))

        # 3. Evaluate Y
        if 'Y' in do:
            y = do['Y']
        else:
            y = self.f_Y(torch.cat([z, x, u_dict['U_Y']], dim=-1))

        return {'Z': z, 'X': x, 'Y': y}


if __name__ == '__main__':
    torch.manual_seed(42)
    scm = DifferentiableSCM()

    # =========================================================
    # RUNG 2: POPULATION INTERVENTIONAL EFFECT (ATE)
    # E[Y | do(X=1)] - E[Y | do(X=0)] across a population
    # =========================================================
    N = 10000
    u_population = {
        'U_Z': torch.randn(N, 1),
        'U_X': torch.randn(N, 1),
        'U_Y': torch.randn(N, 1)
    }

    out_do1 = scm(u_population, do={'X': torch.ones(N, 1)})
    out_do0 = scm(u_population, do={'X': torch.zeros(N, 1)})
    ate = (out_do1['Y'] - out_do0['Y']).mean().item()

    # =========================================================
    # RUNG 3: UNIT-LEVEL COUNTERFACTUAL (ITE)
    # Fixed single unit's exogenous state (one patient/system)
    # =========================================================
    u_unit = {
        'U_Z': torch.tensor([[1.5]]),
        'U_X': torch.tensor([[-0.5]]),
        'U_Y': torch.tensor([[0.8]])
    }

    # 1. Factual world
    factual = scm(u_unit)
    # 2. Counterfactual world for the SAME unit
    counterfactual = scm(u_unit, do={'X': torch.tensor([[1.0]])})
    ite = counterfactual['Y'].item() - factual['Y'].item()

    print("=" * 60)
    print("PYTORCH DIFFERENTIABLE SCM: RUNG 2 VS RUNG 3")
    print("=" * 60)
    print("1. RUNG 2 (Population Interventional ATE):")
    print(f"   E[Y | do(X=1)]: {out_do1['Y'].mean().item():.3f}")
    print(f"   E[Y | do(X=0)]: {out_do0['Y'].mean().item():.3f}")
    print(f"   Average Treatment Effect (ATE): {ate:.3f}")
    print("\n2. RUNG 3 (Single-Unit Counterfactual ITE):")
    print(f"   Factual:        X = {factual['X'].item():.3f}  -->  Y = {factual['Y'].item():.3f}")
    print(f"   Counterfactual: do(X = 1.000)  -->  Y = {counterfactual['Y'].item():.3f}")
    print(f"   Individual Treatment Effect (ITE): {ite:.3f}")
    print("=" * 60)

