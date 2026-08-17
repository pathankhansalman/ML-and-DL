import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression

# 1. Dataset Generation for Probing
FRANCE_SENTENCES = [
    "I visited France last summer.",
    "People in France love baguettes.",
    "France has a rich history.",
    "The geography of France is diverse.",
    "Tourism in France is very popular.",
    "Many artists lived in France.",
    "The economy of France is strong.",
    "We traveled across France by train.",
    "France is known for fashion.",
    "The cuisine of France is world-famous."
]

ITALY_SENTENCES = [
    "I visited Italy last summer.",
    "People in Italy love pasta.",
    "Italy has a rich history.",
    "The geography of Italy is diverse.",
    "Tourism in Italy is very popular.",
    "Many artists lived in Italy.",
    "The economy of Italy is strong.",
    "We traveled across Italy by train.",
    "Italy is known for fashion.",
    "The cuisine of Italy is world-famous."
]

from utils import ActivationCache, patch_hook_builder, find_token_idx

def run_holistic_analysis():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Model & Tokenizer
    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device
    )
    model.eval()

    # Prompts for Patching
    clean_text = "The capital of France is"
    corrupted_text = "The capital of Italy is"
    
    clean_tokens = tokenizer(clean_text, return_tensors="pt").to(device)
    corrupted_tokens = tokenizer(corrupted_text, return_tensors="pt").to(device)
    
    token_labels = [tokenizer.decode([t]) for t in clean_tokens["input_ids"][0]]
    seq_len = len(token_labels)
    num_layers = len(model.model.layers)

    clean_target_id = tokenizer.encode(" Paris")[0]
    corrupted_target_id = tokenizer.encode(" Rome")[0]

    # Baselines
    with torch.no_grad():
        clean_logits = model(**clean_tokens).logits[0, -1]
        corrupted_logits = model(**corrupted_tokens).logits[0, -1]
        
    clean_logit_diff = (clean_logits[clean_target_id] - clean_logits[corrupted_target_id]).item()
    corrupted_logit_diff = (corrupted_logits[clean_target_id] - corrupted_logits[corrupted_target_id]).item()
    
    print(f"Clean Logit Diff: {clean_logit_diff:.4f}")
    print(f"Corrupted Logit Diff: {corrupted_logit_diff:.4f}")

    # Cache clean & corrupted activations for patching
    clean_cache = ActivationCache()
    corrupted_cache = ActivationCache()

    # Cache Clean
    handles = [layer.register_forward_hook(clean_cache.get_hook(i)) for i, layer in enumerate(model.model.layers)]
    with torch.no_grad():
        model(**clean_tokens)
    for h in handles:
        h.remove()

    # Cache Corrupted
    handles = [layer.register_forward_hook(corrupted_cache.get_hook(i)) for i, layer in enumerate(model.model.layers)]
    with torch.no_grad():
        model(**corrupted_tokens)
    for h in handles:
        h.remove()

    # --- 1. Train Probes ---
    print("Training linear probes on country representation...")
    probes = {}
    
    # We will extract activations from the Country token position in training sentences
    # France = 0, Italy = 1
    train_texts = FRANCE_SENTENCES + ITALY_SENTENCES
    train_labels = [0] * len(FRANCE_SENTENCES) + [1] * len(ITALY_SENTENCES)
    
    # Store activations for all layers: shape [num_layers, num_samples, hidden_dim]
    train_activations = {layer_idx: [] for layer_idx in range(num_layers)}

    for text in train_texts:
        # Find which word to locate
        target_word = " France" if "France" in text else " Italy"
        word_idx = find_token_idx(tokenizer, text, target_word)
        tokens = tokenizer(text, return_tensors="pt").to(device)
        
        cache = ActivationCache()
        handles = [layer.register_forward_hook(cache.get_hook(i)) for i, layer in enumerate(model.model.layers)]
        with torch.no_grad():
            model(**tokens)
        for h in handles:
            h.remove()
            
        for layer_idx in range(num_layers):
            # Extract activation at the target word index
            act = cache.cache[layer_idx][0, word_idx, :].cpu().float().numpy()
            train_activations[layer_idx].append(act)

    # Train a probe at each layer
    for layer_idx in range(num_layers):
        X = np.stack(train_activations[layer_idx])
        y = np.array(train_labels)
        
        # Simple Logistic Regression with high C to prevent excessive regularization bias
        clf = LogisticRegression(C=1.0, max_iter=200)
        clf.fit(X, y)
        probes[layer_idx] = clf

    # --- 2. Run Sweeps ---
    rescue_matrix = np.zeros((num_layers, seq_len))
    knockout_matrix = np.zeros((num_layers, seq_len))
    probe_matrix = np.zeros((num_layers, seq_len))

    for layer_idx in range(num_layers):
        print(f"Processing Layer {layer_idx}/{num_layers}...")
        probe = probes[layer_idx]
        
        for token_idx in range(seq_len):
            # A. Probing: Evaluate the trained probe on the clean activation
            clean_act = clean_cache.cache[layer_idx][0, token_idx, :].cpu().float().numpy()
            # Predict probability of class 0 ("France")
            # predict_proba returns [P(France), P(Italy)]
            prob_france = probe.predict_proba([clean_act])[0][0]
            probe_matrix[layer_idx, token_idx] = prob_france

            # B. Rescue Patching: corrupted run, patch in clean
            hook = patch_hook_builder(token_idx, clean_cache.cache[layer_idx])
            handle = model.model.layers[layer_idx].register_forward_hook(hook)
            with torch.no_grad():
                logits = model(**corrupted_tokens).logits[0, -1]
            handle.remove()
            diff = (logits[clean_target_id] - logits[corrupted_target_id]).item()
            rescue_matrix[layer_idx, token_idx] = (diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)

            # C. Knockout Patching: clean run, patch in corrupted
            hook = patch_hook_builder(token_idx, corrupted_cache.cache[layer_idx])
            handle = model.model.layers[layer_idx].register_forward_hook(hook)
            with torch.no_grad():
                logits = model(**clean_tokens).logits[0, -1]
            handle.remove()
            diff = (logits[clean_target_id] - logits[corrupted_target_id]).item()
            knockout_matrix[layer_idx, token_idx] = (clean_logit_diff - diff) / (clean_logit_diff - corrupted_logit_diff)

    # --- 3. Visualization ---
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # Plot 1: Probing Probability
    sns.heatmap(
        probe_matrix, 
        xticklabels=token_labels, 
        yticklabels=list(range(num_layers)), 
        cmap="Purples", 
        annot=True, 
        fmt=".2f", 
        ax=axes[0],
        vmin=0, 
        vmax=1
    )
    axes[0].set_title("Probing: Probability of 'France'")
    axes[0].set_xlabel("Token Position")
    axes[0].set_ylabel("Layer")
    axes[0].invert_yaxis()

    # Plot 2: Rescue Patching
    sns.heatmap(
        rescue_matrix, 
        xticklabels=token_labels, 
        yticklabels=list(range(num_layers)), 
        cmap="RdBu_r", 
        annot=True, 
        fmt=".2f", 
        ax=axes[1],
        center=0.0
    )
    axes[1].set_title("Rescue Patching: Corrupted + Clean Act (Sufficient)")
    axes[1].set_xlabel("Token Position")
    axes[1].set_ylabel("Layer")
    axes[1].invert_yaxis()

    # Plot 3: Knockout Patching
    sns.heatmap(
        knockout_matrix, 
        xticklabels=token_labels, 
        yticklabels=list(range(num_layers)), 
        cmap="Oranges", 
        annot=True, 
        fmt=".2f", 
        ax=axes[2],
        vmin=0, 
        vmax=1
    )
    axes[2].set_title("Knockout Patching: Clean + Corrupted Act (Necessary)")
    axes[2].set_xlabel("Token Position")
    axes[2].set_ylabel("Layer")
    axes[2].invert_yaxis()

    plt.tight_layout()
    output_path = "holistic_comparison.png"
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Successfully generated and saved comparison heatmaps to: {output_path}")

if __name__ == "__main__":
    run_holistic_analysis()
