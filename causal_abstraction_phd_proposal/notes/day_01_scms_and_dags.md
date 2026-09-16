# Day 1: Structural Causal Models (SCMs) & Directed Acyclic Graphs (DAGs)

## Core Concepts Summary

### 1. Observational vs. Interventional Inference
* **Observational:** $P(Y \mid X = x)$ — Passive observation of $X=x$. Subject to confounding.
* **Interventional:** $P(Y \mid do(X = x))$ — Active intervention forcing $X=x$. Removes edge incoming into $X$.

### 2. SCM Definition
An SCM $\mathcal{M} = \langle V, U, F, P(U) \rangle$ consists of:
* **Endogenous variables ($V$):** Internal observed variables.
* **Exogenous variables ($U$):** External background noise variables.
* **Structural equations ($F$):** $V_i := f_i(PA_i, U_i)$ where $PA_i$ are the parents (direct endogenous causes) of $V_i$.

---

## Completed Exercises

### Exercise 1: Structural Equation Assignment & Graph Surgery
Given system:
$$X := U_X$$
$$Y := 3X + U_Y$$

* **Intervention $do(X = 5)$:**
  $$X := 5$$
  $$Y := 3(5) + U_Y = 15 + U_Y$$
* **Graph Effect:** Graph surgery collapses incoming edge $U_X \rightarrow X$. $X$ becomes independent of its natural exogenous cause.

### Exercise 2: Markov Decision Process (MDP) DAG
Given RL Agent:
* $S_t = f_{S_t}(U_{S_t})$
* $A_t = f_{A_t}(S_t, U_{A_t})$
* $R_t = f_{R_t}(S_t, A_t, U_{R_t})$
* $S_{t+1} = f_{S_{t+1}}(S_t, A_t, U_{S_{t+1}})$

#### Induced DAG Structure:
- $PA(A_t) = \{S_t\}$
- $PA(R_t) = \{S_t, A_t\}$
- $PA(S_{t+1}) = \{S_t, A_t\}$
