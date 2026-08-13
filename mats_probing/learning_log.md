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
