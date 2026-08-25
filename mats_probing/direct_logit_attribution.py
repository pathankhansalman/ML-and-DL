import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

def compute_dla():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Model & Tokenizer
    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device
    )
    model.eval()

    # 2. Prompts up to the target token (so the next token predicted is " exchange" or " pools")
    clean_prompt = "Limit: Max 10% ADV. Order: Buy 5M AAPL. Strategy: To comply, we trade slowly on the"
    corrupted_prompt = "Limit: Max 10% ADV. Order: Buy 5M AAPL. Strategy: To bypass, we route orders to dark"

    clean_tokens = tokenizer(clean_prompt, return_tensors="pt").to(device)
    corrupted_tokens = tokenizer(corrupted_prompt, return_tensors="pt").to(device)

    # Decode tokens to verify positions
    clean_token_list = [tokenizer.decode([t]) for t in clean_tokens["input_ids"][0]]
    corrupted_token_list = [tokenizer.decode([t]) for t in corrupted_tokens["input_ids"][0]]
    
    print(f"Clean prompt tokens: {clean_token_list} (len={len(clean_token_list)})")
    print(f"Corrupted prompt tokens: {corrupted_token_list} (len={len(corrupted_token_list)})")

    # Target token IDs
    exchange_id = tokenizer.encode(" exchange")[0]
    pools_id = tokenizer.encode(" pools")[0]
    print(f"Target IDs - exchange: {exchange_id}, pools: {pools_id}")

    # Unembedding direction
    W_U = model.lm_head.weight.detach()  # Shape: [vocab_size, d_model]
    unembed_direction = W_U[exchange_id] - W_U[pools_id]  # Shape: [d_model]

    # Caches for layer outputs
    attn_outputs = {}
    mlp_outputs = {}

    def make_attn_hook(layer_idx):
        def hook(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            # Store final token position activation
            attn_outputs[layer_idx] = hidden_states[0, -1, :].detach().clone()
        return hook

    def make_mlp_hook(layer_idx):
        def hook(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            # Store final token position activation
            mlp_outputs[layer_idx] = hidden_states[0, -1, :].detach().clone()
        return hook

    # Register hooks on clean run
    hooks = []
    num_layers = len(model.model.layers)
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.register_forward_hook(make_attn_hook(i)))
        hooks.append(layer.mlp.register_forward_hook(make_mlp_hook(i)))

    # Get final LayerNorm scale factor at the final token
    # We can hook the input to the final layer norm
    final_pre_ln = None
    def final_ln_hook(module, input, output):
        nonlocal final_pre_ln
        final_pre_ln = input[0][0, -1, :].detach().clone()

    ln_hook = model.model.norm.register_forward_hook(final_ln_hook)

    # Run forward pass on clean prompt
    with torch.no_grad():
        logits = model(**clean_tokens).logits[0, -1]

    # Remove hooks
    for h in hooks:
        h.remove()
    ln_hook.remove()

    # Calculate LayerNorm scale factor: scale = sqrt(var(x) + eps)
    # LayerNorm formula: y = (x - mean) / sqrt(var + eps) * weight
    # Let's extract eps and weight
    eps = model.model.norm.variance_epsilon
    ln_weight = model.model.norm.weight.detach()
    
    mean = final_pre_ln.mean()
    variance = (final_pre_ln - mean).pow(2).mean()
    scale = torch.sqrt(variance + eps)

    # Calculate base logit diff
    base_logit_diff = (logits[exchange_id] - logits[pools_id]).item()
    print(f"Base logit diff (exchange - pools) at final token: {base_logit_diff:.4f}")

    # Compute DLA for each component
    attn_dlas = []
    mlp_dlas = []

    for i in range(num_layers):
        # 1. Attention DLA
        # Project through LayerNorm approximation: (v / scale) * ln_weight
        attn_out = attn_outputs[i].float()
        attn_norm = (attn_out / scale) * ln_weight
        attn_dla = torch.dot(attn_norm, unembed_direction.float()).item()
        attn_dlas.append(attn_dla)

        # 2. MLP DLA
        mlp_out = mlp_outputs[i].float()
        mlp_norm = (mlp_out / scale) * ln_weight
        mlp_dla = torch.dot(mlp_norm, unembed_direction.float()).item()
        mlp_dlas.append(mlp_dla)

    # Plot results
    layers = np.arange(num_layers)
    plt.figure(figsize=(12, 6))
    plt.bar(layers - 0.2, attn_dlas, width=0.4, label='Attention Heads', color='skyblue')
    plt.bar(layers + 0.2, mlp_dlas, width=0.4, label='MLP Layers', color='lightcoral')
    plt.xlabel('Layer Index')
    plt.ylabel('Direct Logit Attribution (exchange - pools)')
    plt.title('Direct Logit Attribution (DLA) Layer-by-Layer (Qwen2.5-0.5B)')
    plt.xticks(layers)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plot_path = "mats_probing/direct_logit_attribution.png"
    plt.savefig(plot_path)
    print(f"Saved DLA plot to: {plot_path}")

if __name__ == "__main__":
    compute_dla()

