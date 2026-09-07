# ML and DL Research

This repository contains machine learning, deep learning, and AI alignment research projects.

---

## 🌟 Featured Project: Mechanistic Interpretability & AI Safety

### **[Causal Circuit Mapping and Activation Steering of Evasion Intent in Trading Agents](mats_probing/)**
* **Author:** Salman Khan Pathan
* **Focus:** Reverse-engineering regulatory evasion circuits in LLMs (`Qwen2.5-0.5B`) using Linear Probing, Causal Activation Patching, Direct Logit Attribution (DLA), and Representation Engineering (PCA Activation Steering).
* **Key Findings:**
  * Intent emerges in **middle layers (Layers 6–15)** with $100\%$ probe accuracy.
  * **Late-layer MLPs (Layers 19–21)** act as the primary vocabulary writers, driving $>83\%$ of positive logit attribution.
  * **Low-Rank PCA Steering** achieves a **$4.58\times$ efficacy gain** over mean difference, successfully enforcing compliance at runtime.
  * Middle-layer steering (**Layer 12**) provides robust buffer protection against out-of-distribution language collapse.

👉 **[Read the Full Project Documentation & Methodology in `mats_probing/`](mats_probing/README.md)**
