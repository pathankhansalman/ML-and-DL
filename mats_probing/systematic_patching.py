import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import ActivationCache, patch_hook_builder

def run_systematic_patching():
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

    # 2. Define Prompts (Must tokenize to the exact same length)
    clean_text = "The capital of France is"
    corrupted_text = "The capital of Italy is"
    
    clean_tokens = tokenizer(clean_text, return_tensors="pt").to(device)
    corrupted_tokens = tokenizer(corrupted_text, return_tensors="pt").to(device)
    
    # Get token representations for printing/labeling
    token_labels = [tokenizer.decode([t]) for t in clean_tokens["input_ids"][0]]
    seq_len = len(token_labels)
    num_layers = len(model.model.layers)
    
    print(f"Sequence length: {seq_len} tokens -> {token_labels}")
    print(f"Model layers: {num_layers}")

    # Target logits to compare
    clean_target_id = tokenizer.encode(" Paris")[0]
    corrupted_target_id = tokenizer.encode(" Rome")[0]
    print(f"Clean Target (' Paris' ID): {clean_target_id}")
    print(f"Corrupted Target (' Rome' ID): {corrupted_target_id}")

    # 3. Establish Baselines
    model.eval()
    with torch.no_grad():
        clean_logits = model(**clean_tokens).logits[0, -1]
        corrupted_logits = model(**corrupted_tokens).logits[0, -1]
        
    clean_logit_diff = (clean_logits[clean_target_id] - clean_logits[corrupted_target_id]).item()
    corrupted_logit_diff = (corrupted_logits[clean_target_id] - corrupted_logits[corrupted_target_id]).item()
    
    print(f"Baseline Clean Logit Diff (Paris - Rome): {clean_logit_diff:.4f}")
    print(f"Baseline Corrupted Logit Diff (Paris - Rome): {corrupted_logit_diff:.4f}")

    # 4. Cache Clean Activations
    clean_cache = ActivationCache()
    handles = []
    for layer_idx, layer in enumerate(model.model.layers):
        hook = clean_cache.get_hook(layer_idx)
        handles.append(layer.register_forward_hook(hook))
        
    with torch.no_grad():
        model(**clean_tokens)
        
    for handle in handles:
        handle.remove()
        
    print("Successfully cached clean activations.")

    # 5. Run the 2D Sweep (Layers x Positions)
    # Initialize results grid
    results = torch.zeros((num_layers, seq_len))

    for layer_idx in range(num_layers):
        print(f"Sweeping Layer {layer_idx}/{num_layers}...")
        for token_idx in range(seq_len):
            # Build patch hook
            clean_act = clean_cache.cache[layer_idx]
            hook = patch_hook_builder(token_idx, clean_act)
            
            # Register hook on target layer
            handle = model.model.layers[layer_idx].register_forward_hook(hook)
            
            # Run model on corrupted input
            with torch.no_grad():
                patched_logits = model(**corrupted_tokens).logits[0, -1]
                
            # Remove hook immediately
            handle.remove()
            
            # Calculate normalized rescue effect
            patched_logit_diff = (patched_logits[clean_target_id] - patched_logits[corrupted_target_id]).item()
            
            # Normalization formula
            normalized_effect = (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)
            results[layer_idx, token_idx] = normalized_effect

    # 6. Plot and Save Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        results.numpy(), 
        xticklabels=token_labels, 
        yticklabels=list(range(num_layers)), 
        cmap="RdBu_r", 
        center=0.0,
        annot=True,
        fmt=".2f"
    )
    plt.title("Activation Patching Heatmap (Rescue Effect on Qwen-0.5B)")
    plt.xlabel("Token Position patched with Clean Activation")
    plt.ylabel("Layer Index")
    plt.gca().invert_yaxis()  # Put Layer 0 at the bottom
    
    heatmap_path = "patching_heatmap.png"
    plt.savefig(heatmap_path, bbox_inches="tight")
    print(f"Saved systematic patching heatmap to: {heatmap_path}")

if __name__ == "__main__":
    run_systematic_patching()
