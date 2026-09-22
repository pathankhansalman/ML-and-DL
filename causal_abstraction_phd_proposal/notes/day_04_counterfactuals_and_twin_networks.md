# Day 4: Exogenous Noise ($U$), Counterfactuals, & Twin Networks

> **PhD Proposal Context:** Causal Abstractions (CA) and Interventionally Consistent Surrogates (ICS4CSM) must preserve unit-level causal mechanisms across abstraction levels. Today we formalize **Rung 3 of Pearl's Causal Hierarchy**: Counterfactuals ($Y_{X=x^*}(u)$) and the structural role of exogenous noise $U$.

---

## 1. Interventions vs. Counterfactuals

| Property | Rung 2: Interventions ($do(\cdot)$) | Rung 3: Counterfactuals ($Y_{X=x^*}$) |
|:---|:---|:---|
| **Query Level** | Population-level average | Unit-level / Individual-level |
| **Question** | *"If we treat everyone, what is the expected outcome?"* | *"Given that Alice was treated and recovered, would **she** have recovered without treatment?"* |
| **Mathematical Form** | $P(Y \mid do(X = x^*))$ | $P(Y_{X=x^*} = y^* \mid X = x, Y = y)$ |
| **Exogenous Noise ($U$)** | Averaged out over prior distribution $P(U)$ | **Conditioned upon / frozen** for that specific unit |
| **Model Requirements** | DAG + Causal Markov Condition | Full SCM $\langle V, U, F, P(U) \rangle$ |

### The Physical Meaning of Exogenous Noise $U$
In an SCM, each endogenous variable $V_i$ is determined by:
$$V_i = f_i(PA_i, U_i)$$
* $PA_i$: Observable parent causes in the system.
* $U_i$: **Exogenous background factors**—the unobserved genetic predisposition, microscopic environment, and individual uniqueness of that unit.
* In a randomized experiment ($do(X)$), we sample new individuals from $P(U)$.
* In a counterfactual ($Y_{x^*}$), we **freeze unit $i$'s exact $U$**, rewind time, and intervene only on $X$.

---

## 2. Pearl's 3-Step Counterfactual Algorithm

Every structural counterfactual query can be solved algorithmically through three steps:

```
                      [ Factual Evidence: E = e ]
                                   │
                                   ▼  STEP 1: ABDUCTION
                      [ Compute P(U | E = e) ]
                                   │
                                   ▼  STEP 2: ACTION
                   [ Substitute X = x* into SCM M ]
                                   │
                                   ▼  STEP 3: PREDICTION
               [ Compute Y_x* using M_x* and updated P(U|e) ]
```

### Step 1: Abduction
Use the factual evidence $E = e$ to update the distribution of exogenous variables $U$:
$$P(U \mid E = e) = \frac{P(E = e \mid U) P(U)}{P(E = e)}$$
*(Deduce what unobserved characteristics the individual must have possessed to produce the observed facts).*

### Step 2: Action
Perform graph surgery on SCM $\mathcal{M}$ to construct the sub-model $\mathcal{M}_{x^*}$:
* Replace the equation $X = f_X(PA_X, U_X)$ with the constant assignment:
$$X \leftarrow x^*$$

### Step 3: Prediction
Using the modified model $\mathcal{M}_{x^*}$ and the abducted posterior distribution $P(U \mid E = e)$, compute the counterfactual distribution:
$$P(Y_{X = x^*} = y \mid E = e) = \int_{\{u \mid f_Y^{\mathcal{M}_{x^*}}(PA_Y, u) = y\}} P(u \mid E = e) \, du$$

---

## 3. Twin Networks

A **Twin Network** (Balke & Pearl, 1994) is a graphical representation that brings the factual world and the counterfactual world onto the same DAG.

### Architecture:
1. **Factual World (Left):** Nodes $X, Y, M$ representing the observed reality.
2. **Counterfactual World (Right):** Nodes $X^*, Y^*, M^*$ representing the parallel counterfactual reality under $do(X = x^*)$.
3. **Shared Latent Anchor ($U$):** The factual and counterfactual nodes **share the exact same exogenous variables $U$**.

```
    Factual World              Shared Noise             Counterfactual World
   
      [ X ] <───────────────────── ( U_X ) ─────────────────────> [ X* = x* ]
        │                                                              │
        ▼                                                              ▼
      [ M ] <───────────────────── ( U_M ) ─────────────────────> [ M* ]
        │                                                              │
        ▼                                                              ▼
      [ Y ] <───────────────────── ( U_Y ) ─────────────────────> [ Y* ]
```

* Because $U$ connects both worlds, observing facts in the factual world ($X=x, Y=y$) transmits information through $U$ into the counterfactual world ($Y^*$).
* Graphically, we can run $d$-separation directly on the Twin Network to test counterfactual independencies!

---

## 4. Potential Outcomes vs. Structural Counterfactuals

* **Rubin's Potential Outcomes ($Y(x)$):**
  * Treats potential outcomes as primitive random variables indexed by treatment $x$.
  * Relies on the **SUTVA** assumption (Stable Unit Treatment Value Assumption) and Ignorability: $(Y(0), Y(1)) \perp X \mid Z$.
* **Pearl's Structural Counterfactuals ($Y_x(u)$):**
  * Potential outcomes are **derived mathematically** from the deterministic equations of an SCM:
    $$Y_x(u) \triangleq f_Y^{\mathcal{M}_x}(PA_Y, u)$$
  * Connects Rubin's potential outcomes directly to graph surgery and DAG topologies.

---

## 5. Connection to PhD Research: Causal Abstractions & DRO

1. **Interventional vs. Counterfactual Consistency in Causal Abstractions:**
   * **Interventional Consistency (ICS4CSM):** Ensures that intervening on a macro-model yields the abstracted result of intervening on the micro-model ($\tau(\mathcal{M}_{\text{micro}, \omega(x)}) = \mathcal{M}_{\text{macro}, x}$).
   * **Counterfactual Consistency:** Stricter condition—requires that an individual unit's counterfactual in the micro-model maps to their counterfactual in the macro-model (requires aligning latent noise spaces $U_{\text{micro}} \to U_{\text{macro}}$).
2. **Distributionally Robust Optimization (DRO & DiRoCA):**
   * Worst-case domain shift can be modeled as an adversary perturbing the distribution of exogenous noise $P(U) \to Q(U)$ within a Wasserstein ambiguity set $\mathcal{W}_\epsilon(P(U))$.
   * Robust surrogates must generalize under shifts in $P(U)$ while keeping structural mechanisms $F$ invariant.
