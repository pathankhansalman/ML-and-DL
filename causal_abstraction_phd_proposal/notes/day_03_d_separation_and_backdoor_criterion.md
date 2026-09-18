# Day 3: $d$-Separation, The Backdoor Criterion, & Collider Bias

## 📚 Core Theoretical Concepts

In observational causal inference, our central goal is to estimate $P(Y \mid do(X = x))$ from passive observational data. To know **which variables we must adjust for** (and which we must **never** adjust for), we rely on graph theory: **$d$-separation** and the **Backdoor Criterion**.

---

### 1. The Three Fundamental Graphical Junctions

Any path in a Directed Acyclic Graph (DAG) can be decomposed into combinations of three local 3-node structures. The behavior of information flow under conditioning differs critically across them:

#### A. Chain (Mediation): $A \rightarrow B \rightarrow C$
* **Unconditioned:** Open (information flows from $A$ to $C$ via mediator $B$).
* **Conditioned on $B$ ($A \perp C \mid B$):** **Blocked**. Knowing the intermediate mechanism completely shields $C$ from $A$.
* *Caution:* If $X \rightarrow M \rightarrow Y$, conditioning on mediator $M$ blocks the very causal mechanism we wish to measure!

#### B. Fork (Common Cause / Confounding): $A \leftarrow B \rightarrow C$
* **Unconditioned:** Open (spurious association flows between $A$ and $C$ due to mutual parent $B$).
* **Conditioned on $B$ ($A \perp C \mid B$):** **Blocked**. Once we hold the common cause constant, $A$ and $C$ become conditionally independent.
* *Rule:* Conditioning on confounder $B$ eliminates confounding bias.

#### C. Collider (Common Effect / Inverted Fork): $A \rightarrow B \leftarrow C$
* **Unconditioned:** **BLOCKED** naturally! $A$ and $C$ are marginally independent ($A \perp C$). Two independent causes of a common effect do not correlate.
* **Conditioned on $B$ (or any descendant of $B$):** **OPENS** the path! Conditioning on a collider introduces spurious correlation between its causes ($A \not\perp C \mid B$).
* *Phenomenon:* **Berkson's Paradox** or **Collider Bias**.
* *Example:* Talent ($A$) $\rightarrow$ Celebrity Status ($B$) $\leftarrow$ Wealthy Family ($C$). In the general population, talent and family wealth are uncorrelated. But among celebrities (conditioned on $B=1$), finding someone with zero family wealth makes it virtually certain they possess immense talent.

---

### 2. Formal Definition of $d$-Separation

A path $p$ between node $X$ and node $Y$ is **blocked** (or $d$-separated) by a set of conditioning variables $Z$ if and only if:
1. $p$ contains a **chain** ($A \rightarrow B \rightarrow C$) or a **fork** ($A \leftarrow B \rightarrow C$) such that middle node $B \in Z$.
2. **OR** $p$ contains a **collider** ($A \rightarrow B \leftarrow C$) such that middle node $B \notin Z$ **and** no descendant of $B$ is in $Z$.

If all paths between set $X$ and set $Y$ are blocked by $Z$, then $X$ and $Y$ are **$d$-separated by $Z$** (written $(X \perp Y \mid Z)_{\mathcal{G}}$). By the Causal Markov Condition:
$$(X \perp Y \mid Z)_{\mathcal{G}} \implies X \perp Y \mid Z \quad \text{in probability } P$$

---

### 3. Pearl's Backdoor Criterion

To compute $P(Y \mid do(X = x))$ from observational data via adjustment, we must block all non-causal spurious paths while leaving all causal paths unobstructed.

#### Definition
A set of variables $S$ satisfies the **Backdoor Criterion** relative to an ordered pair of variables $(X, Y)$ in DAG $\mathcal{G}$ if:
1. **No Descendants:** No node in $S$ is a descendant of $X$ (i.e., $S \cap \text{Desc}(X) = \emptyset$).
2. **Blocks Backdoor Paths:** $S$ blocks (d-separates) every path between $X$ and $Y$ that has an **arrow pointing into $X$** (paths of the form $X \leftarrow \dots \rightarrow Y$).

#### The Backdoor Adjustment Formula
If a set $S$ satisfies the Backdoor Criterion with respect to $(X, Y)$, the causal effect is identified by:
$$P(Y \mid do(X = x)) = \sum_{s} P(Y \mid X = x, S = s) P(S = s)$$

---

### 4. Pathological Pitfalls in Causal Adjustment ("Bad Controls")

Not all variables are safe to include in a regression or conditioning set!

1. **Conditioning on a Mediator ($X \rightarrow M \rightarrow Y$):**
   * Blocks the direct/indirect causal transmission of $X \rightarrow Y$.
   * Estimating $Y \sim X + M$ extinguishes the total causal effect of $X$.

2. **Conditioning on a Collider ($X \rightarrow C \leftarrow Y$):**
   * Opens an otherwise inactive path, introducing artificial association where none existed.

3. **M-Bias Structure:**
   ```
     U1       U2
    /  \     /  \
   v    v   v    v
   X      M      Y
   ```
   * Here, $M$ is a collider on the backdoor path $X \leftarrow U_1 \rightarrow M \leftarrow U_2 \rightarrow Y$.
   * Path is **naturally closed** without conditioning!
   * If an analyst includes $M$ as a "control variable", the path **opens**, introducing confounding that was not originally present!

---

## 🎯 Summary Cheatsheet

| Structure | Unconditioned State | Conditioned State | Purpose in Causal Adjustment |
|:---|:---:|:---:|:---|
| **Chain** ($X \rightarrow M \rightarrow Y$) | Open | Blocked | **Do NOT adjust** if you want total causal effect. |
| **Fork** ($X \leftarrow Z \rightarrow Y$) | Open | Blocked | **MUST adjust** to eliminate confounding. |
| **Collider** ($X \rightarrow C \leftarrow Y$) | Blocked | Open | **NEVER adjust**; creates selection/Berkson bias. |
| **M-Structure Collider** ($X \leftarrow U_1 \rightarrow M \leftarrow U_2 \rightarrow Y$) | Blocked | Open | **NEVER adjust** for $M$. |
