# Day 2: The $do(\cdot)$ Operator & Truncated Factorization

## 📚 Core Theoretical Concepts

### 1. The $do(\cdot)$ Operator (Graph Surgery)
In a Bayesian Network or SCM $\mathcal{M} = \langle V, U, F, P(U) \rangle$, an intervention $do(X_k = x_k)$ represents an exogenous forced action rather than passive observation.

* **Observational Conditioning $P(Y \mid X_k = x_k)$:** Filters the population to sub-population where $X_k = x_k$. All incoming edges into $X_k$ remain intact, allowing spurious correlation/confounding flow through $PA_k \rightarrow X_k$.
* **Interventional Conditioning $P(Y \mid do(X_k = x_k))$:** Performs **graph surgery** on DAG $\mathcal{G}$:
  1. Removes all incoming edges into $X_k$ (severing $PA_k \rightarrow X_k$).
  2. Overwrites structural equation $f_k(PA_k, U_k)$ with constant assignment $X_k := x_k$.
  3. Leaves all other structural equations $f_i$ ($i \neq k$) unchanged.

---

### 2. Truncated Factorization Formula (G-Formula / Manipulated Graph Rule)
By the Causal Markov Condition, the observational joint distribution factorizes over DAG $\mathcal{G}$:
$$P(X_1, X_2, \dots, X_n) = \prod_{i=1}^n P(X_i \mid PA_i)$$

Under intervention $do(X_k = x_k)$, the conditional probability factor $P(X_k \mid PA_k)$ is replaced by $1$ (for $X_k = x_k$) and dropped from the product for all other variables. The interventional joint distribution over remaining variables $V \setminus \{X_k\}$ becomes:

$$P(X_1, \dots, X_{k-1}, X_{k+1}, \dots, X_n \mid do(X_k = x_k)) = \prod_{i \neq k} P(X_i \mid PA_i) \Big|_{X_k = x_k}$$

This is the **Truncated Factorization Formula** (also called Pearl's *Manipulated Density Formula* or Robins' *g-formula*).

---

## ✏️ Mathematical Derivation & Proof

### Setup: Classic Confounder DAG ($Z \rightarrow X$, $Z \rightarrow Y$, $X \rightarrow Y$)
Let $Z$ be age/severity (confounder), $X$ be treatment, and $Y$ be recovery outcome.

1. **Observational Joint Distribution:**
   $$P(Z, X, Y) = P(Z) P(X \mid Z) P(Y \mid X, Z)$$

2. **Interventional Joint under $do(X = x^*)$:**
   By Truncated Factorization, we remove $P(X \mid Z)$:
   $$P(Z, Y \mid do(X = x^*)) = P(Z) P(Y \mid X = x^*, Z)$$

3. **Marginal Interventional Distribution $P(Y \mid do(X = x^*))$:**
   Summing/Integrating over confounder $Z$:
   $$P(Y \mid do(X = x^*)) = \sum_{z} P(Z = z, Y \mid do(X = x^*)) = \sum_{z} P(Y \mid X = x^*, Z = z) P(Z = z)$$

> **Key Insight:** This exact formula is the **Backdoor Adjustment Formula**! Truncated factorization proves why adjusting for confounders $Z$ recovers the true interventional distribution from observational data without conducting physical randomized trials.
