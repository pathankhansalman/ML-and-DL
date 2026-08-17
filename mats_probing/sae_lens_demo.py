import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

def run_sae_lens_demo():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load the HookedTransformer (GPT-2 small)
    model_name = "gpt2-small"
    print(f"Loading model {model_name}...")
    model = HookedTransformer.from_pretrained(model_name, device=device)

    # 2. Load the pre-trained SAE for Layer 8 (Residual Stream - Pre)
    release_name = "gpt2-small-res-jb"
    sae_id = "blocks.8.hook_resid_pre"
    
    print(f"Loading SAE {sae_id} from release {release_name}...")
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=release_name,
        sae_id=sae_id,
        device=device
    )
    
    # 3. Define prompt and run model with SAE activation tracking
    prompt = "The capital of France is Paris."
    print(f"\nRunning model on prompt: '{prompt}'")
    
    # Run the model and get activations at Layer 8 residual pre hook
    logits, cache = model.run_with_cache(prompt)
    layer_activation = cache[sae_id] # Shape: [batch, seq_len, d_model]
    
    # 4. Pass activations through the pre-trained SAE
    tokens = model.to_tokens(prompt)[0]
    token_strings = [model.tokenizer.decode([t]) for t in tokens]
    
    # Shape of inputs to SAE: [seq_len, d_model]
    sae_in = layer_activation[0]
    
    # Encode activations using the SAE
    feature_acts = sae.encode(sae_in) # Shape: [seq_len, num_features]
    
    print("\n--- Top Active Features Per Token ---")
    for token_idx, token_str in enumerate(token_strings):
        print(f"\nToken: '{token_str}' (Index {token_idx})")
        acts = feature_acts[token_idx]
        
        # Find features with non-zero activations
        active_indices = torch.nonzero(acts).squeeze(-1)
        if len(active_indices) == 0:
            print("  No features active.")
            continue
            
        active_values = acts[active_indices]
        sorted_indices = active_indices[torch.argsort(active_values, descending=True)]
        
        # Print top 3 active features
        for rank, feat_idx in enumerate(sorted_indices[:3]):
            val = acts[feat_idx].item()
            print(f"  Rank {rank+1}: Feature #{feat_idx.item()} | Activation: {val:.4f}")

if __name__ == "__main__":
    run_sae_lens_demo()
