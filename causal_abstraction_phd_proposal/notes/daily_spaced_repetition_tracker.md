# 🧠 Daily Spaced Repetition & Active Recall Tracker

> **Purpose:** A progressive, active recall question bank to test core causal inference, SCM, causal abstraction, and DRO concepts over the 4-week PhD proposal preparation. Questions are added each day and reviewed periodically.

---

## 📅 Question Bank by Day

### Day 1: SCMs, DAGs, & Graph Surgery
* **Q1.1:** What is the formal 4-tuple definition of an SCM $\mathcal{M}$, and what does each component represent?
* **Q1.2:** In an SCM, what is the difference between endogenous ($V$) and exogenous ($U$) variables? Can an exogenous variable have parents?
* **Q1.3:** Explain "graph surgery" in physical terms: what happens to the DAG and structural equations when we apply $do(X = x)$?

---

### Day 2: The $do(\cdot)$ Operator & Truncated Factorization
* **Q2.1:** What does it mean for an unblocked confounder $Z$ to "distort" the observed relationship between $X$ and $Y$?
* **Q2.2:** What are the three distinct probability distributions involved in causal identification, and what does each represent?
  1. Observational Joint: $P(Z, X, Y)$
  2. Interventional Joint: $P(Z, Y \mid do(X = x^*))$
  3. Marginal Interventional: $P(Y \mid do(X = x^*))$
* **Q2.3:** What mathematical operation takes you from (1) Observational Joint to (2) Interventional Joint?
* **Q2.4:** What mathematical operation takes you from (2) Interventional Joint to (3) Marginal Interventional Distribution?
* **Q2.5:** Write down the final Backdoor Adjustment formula resulting from this 3-step derivation.

---

### Day 3: $d$-Separation, Backdoor Criterion, & Bad Controls
* **Q3.1:** What are the 3 canonical 3-node junctions in a DAG? Which ones are naturally open vs. naturally blocked?
* **Q3.2:** What happens to information flow through a collider ($A \rightarrow B \leftarrow C$) when you condition on $B$ (or a descendant of $B$)? What is this phenomenon called?
* **Q3.3:** State the two conditions required for a set of variables $S$ to satisfy Pearl's Backdoor Criterion relative to $(X, Y)$.
* **Q3.4:** Why is conditioning on a mediator ($X \rightarrow M \rightarrow Y$) considered a "bad control" when attempting to estimate the total causal effect of $X$ on $Y$?
* **Q3.5:** What is M-bias, and why does controlling for the middle node in an M-structure introduce bias?

---

## 📈 Review Schedule & Recall Log

| Review Date | Questions Tested | Concept / Focus | Mastery Level (1–5) | Notes / Gaps |
|:---:|:---|:---|:---:|:---|
| *Day 2* | Q2.2 – Q2.5 | 3 Distributions & Truncated Factorization | 5 / 5 ✅ | Mastered intuition: (1) passive observation, (2) intervention with upstream intact, (3) marginalizing out upstream. |
| *Day 3 (Today)* | Q1.1–Q1.3, Q2.1–Q2.5 | SCMs, Exogenous vs Endogenous, Confounding & Truncated Factorization | 5 / 5 ✅ | Perfect recall: exogenous variables have no parents, graph surgery cuts incoming edges, marginalization yields backdoor formula. |
| *Day 4* | Day 3 questions, Counterfactuals | d-Separation, Colliders, Exogenous Noise ($U$) & Counterfactuals | ⬜ Scheduled | Next session |
| *Day 7 (W1 Review)* | All Week 1 Questions | Full Week 1 Synthesis | ⬜ Scheduled | |
| *Day 14 (W2 Review)* | Week 1 & Week 2 Questions | Abstraction & Commutative Diagrams | ⬜ Scheduled | |
