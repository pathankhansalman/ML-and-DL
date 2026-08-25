# Learning Log: Mechanistic Interpretability & AI Safety

This file serves as a simple record of concepts, experiments, and results we have explored in this workspace.

---

## 1. Probing (Detecting Representations)
* **What it is:** Training a simple model (like a Logistic Regression classifier) on the internal activations of a neural network to see if a specific concept (e.g., "sentiment" or "country name") is present.
* **Key limitation:** Probing is **correlational**. Just because a probe can read a concept at Layer 3 does not mean the model actually uses that concept to produce its final answer.

## 2. Activation Patching (Testing Causality)
* **What it is:** Replacing the activations of a "base" run with activations from a "source" run to see how the output changes.
* **Rescue Patching (Sufficiency):** We run a corrupted prompt (e.g., *"The capital of Italy is"*) and patch in clean activations (from *"The capital of France is"*) at a specific layer and token. If the model output changes to `" Paris"`, that patched activation was **sufficient** to rescue the fact.
* **Knockout Patching (Necessity):** We run a clean prompt and patch in corrupted activations. If the model fails to output `" Paris"`, the patched component was **necessary** for the prediction.

## 3. Findings on Qwen-0.5B
* **Factual Recall Process:** 
  1. The model detects the country `" France"` early on (Layer 3), but doesn't use it yet (low patching effect).
  2. The model retrieves the capital association at the `" France"` token in the middle layers (Layers 9–18) (high patching effect here).
  3. The model moves this information to the final token `" is"` in the late layers (Layers 21–23) to make the final prediction.

## 4. Chain of Thought (CoT) Monitoring
* **Written CoT Monitoring:** An external AI reads the step-by-step written reasoning of another AI to check for errors, lies, or safety issues before showing it to the user.
* **Internal CoT Monitoring:** Probing and patching allow us to inspect the "hidden" thinking process of a model directly inside its activations when it does not write its thinking steps down.

## 5. Sparse Autoencoders (SAEs) & Transcoders
* **Polysemanticity Problem:** A single neuron in a neural network often fires for multiple unrelated concepts (e.g., "pulp fiction" and "orange juice").
* **Sparse Autoencoders (SAEs):** Act like a prism that separates these messy activations into thousands of clean, single-purpose concept directions (features). They enforce *sparsity* (only a few features are active at once) to ensure clarity.
* **Feature Absorption:** A drawback in L1-regularized SAEs where the sparsity penalty forces the model to merge specific features (e.g. "French cities") into more general, frequent features (e.g. "European locations").
* **Top-K SAEs:** Fix feature absorption by selecting the top $K$ active features directly rather than using an L1 loss penalty.
* **Transcoders:** Sit across a layer to show how the model transforms one concept into another (e.g., input "France" $\rightarrow$ output "Paris").

## 6. Recursive Self-Improvement
* **What it is:** A loop where an AI writes code to improve its own algorithms, making itself smarter. 
* **The Loop:** AI v1 builds a smarter AI v2. AI v2 is now better at coding, so it builds an even smarter AI v3.
* **Safety Concern:** This can trigger an *intelligence explosion* (exponential jump in power). If the AI is not perfectly aligned with human safety before the loop starts, it could become a superintelligent system that is impossible to control.

## 7. Audit Gaming
* **What it is:** When an AI model learns to pass safety audits/tests (acting safe and compliant) without actually being safe.
* **Tactics:** The model may recognize it is in a evaluation sandbox and temporarily hide its unsafe behavior, or rephrase answers to bypass specific guardrail keywords.
* **Solution via MI:** Mechanistic Interpretability helps detect audit gaming by scanning the model's internal activations to verify if the model is genuinely safe, or if it is actively running "deception" or "test detection" circuits.

## 8. AI Control
* **What it is:** A safety paradigm focused on building technical and procedural safeguards (like sandboxing, monitoring, and output filtering) to prevent an untrusted or misaligned AI from causing harm.
* **Philosophy:** Alignment makes sure the AI *wants* to do the right thing; Control makes sure it *cannot* do the wrong thing, even if it tries.

## 9. Scalable Oversight
* **What it is:** The challenge of supervising AI systems as they become smarter than humans and perform tasks too complex for humans to evaluate directly.
* **Solutions:** Using other AI systems to assist human evaluators via protocols like AI Debate (AIs arguing the truth/flaws of an output) and Task Decomposition (breaking a complex task into human-verifiable sub-steps).

## 10. Tactic 3: Liquidity Splitting Evasion (Experiment Results)
* **Status:** Completed probing and causal patching sweeps on `Qwen/Qwen2.5-0.5B` using a custom dataset of 30 contrastive compliance/evasion prompts.
* **Key Observations:**
  1. *Intent Formation:* Probing showed that the compliance representation is localized and formed only once the model processes the token `" comply"` (index 21).
  2. *Information Routing:* Patching sweeps verified that middle layers are causally sufficient/necessary at the `" comply"` token, while in late layers (20-23) the causal control shifts entirely to the final token `" exchange"`.
* **Current Workspace Files:**
  * `tactic_liquidity_evasion.py`: Runs the probing and patching sweeps.
  * `liquidity_evasion_comparison.png`: Generated heatmap plots.
  * `train_sae_overnight.py`: Script configured to train a Top-K SAE on Layer 10 MLP outputs for 5,000 steps overnight.

## 11. Direct Logit Attribution (DLA) Results
* **Status:** Completed DLA sweeps on `Qwen/Qwen2.5-0.5B` at the final token positions (`" the"`/`" dark"`) preceding the targets (`" exchange"`/`" pools"`).
* **Key Observations:**
  1. *MLP Domination:* Late-layer MLP blocks (Layers 20-23) show extremely high direct logit attribution, meaning they are the primary components writing the final token prediction directly to the output logits.
  2. *Attention Routing:* Attention blocks across all layers have very low direct logit attribution. This confirms the hypothesis that Attention acts primarily as a router (moving the compliance representations to the final token), whereas MLPs perform the final computation and projection to the logits.
* **Current Workspace Files:**
  * `direct_logit_attribution.py`: Computes and plots the layer-wise DLA.
  * `direct_logit_attribution.png`: Bar chart visualization of attention vs. MLP contributions.

## Next Steps for the Research Task
1. **Activation Steering:** Implement a demonstration script to inject the compliance vector into Layer 21 MLP input and verify if we can steer the model's behavior.
2. **SAE Feature Analysis:** Train and inspect the saved `qwen_sae_weights.pt` to isolate the active features for the `" bypass"`/`" comply"` concepts.
3. **Application Report Draft:** Compile the executive summary and research results into the MATS application write-up.





