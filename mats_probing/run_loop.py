import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataset import get_train_test_split
from probe import extract_last_token_activations, train_probe

def run_layer_sweep(step_size=4, use_huggingface=False):
    """
    Sweeps through model layers with a configurable step size,
    extracts activations, trains probes, and outputs results.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running layer sweep (step size = {step_size}) on device: {device}")
    
    # 1. Load data
    train_texts, test_texts, train_labels, test_labels = get_train_test_split(use_huggingface=use_huggingface)
    
    # 2. Load model & tokenizer once
    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"Loading {model_name} in float16...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        device_map=device
    )
    
    # Determine number of layers
    num_layers = len(model.model.layers)
    layers_to_sweep = list(range(0, num_layers, step_size))
    
    # Ensure the final layer is included
    if (num_layers - 1) not in layers_to_sweep:
        layers_to_sweep.append(num_layers - 1)
        
    print(f"Layers to sweep: {layers_to_sweep}")
    
    results = []
    
    # 3. Sweep layers
    for layer_num in layers_to_sweep:
        print(f"\n--- Sweeping Layer {layer_num} ---")
        
        # Extract activations at current layer
        train_acts = extract_last_token_activations(model, tokenizer, train_texts, layer_num, device)
        test_acts = extract_last_token_activations(model, tokenizer, test_texts, layer_num, device)
        
        # Train and evaluate probes
        acc = train_probe("logistic", train_acts, train_labels, test_acts, test_labels, epochs=150)
        mse = train_probe("linear", train_acts, train_labels, test_acts, test_labels, epochs=150)
        
        results.append({
            "layer": layer_num,
            "logistic_accuracy": acc,
            "linear_mse": mse
        })
        
    # 4. Print text-based table
    print("\n" + "="*50)
    print(f"{'Layer':<8} | {'Logistic Accuracy':<20} | {'Linear MSE':<12}")
    print("="*50)
    for res in results:
        print(f"{res['layer']:<8} | {res['logistic_accuracy']*100:>17.2f}% | {res['linear_mse']:>10.4f}")
    print("="*50)
    
    # 5. Plot and save results
    layers = [res["layer"] for res in results]
    accs = [res["logistic_accuracy"] for res in results]
    mses = [res["linear_mse"] for res in results]
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    # Primary axis for Accuracy
    color = 'tab:blue'
    ax1.set_xlabel('Layer Number')
    ax1.set_ylabel('Logistic Probe Accuracy', color=color)
    ax1.plot(layers, accs, marker='o', color=color, label='Logistic Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0.4, 1.05)
    
    # Secondary axis for MSE
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Linear Probe MSE Loss', color=color)
    ax2.plot(layers, mses, marker='s', linestyle='--', color=color, label='Linear MSE')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Probing Sentiment Concept Across Qwen Layers')
    fig.tight_layout()
    
    plot_path = "probing_results.png"
    plt.savefig(plot_path)
    print(f"\nPlot saved successfully to: {plot_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=4, help="Step size for sweeping layers")
    parser.add_argument("--huggingface", action="store_true", help="Use Rotten Tomatoes dataset from Hugging Face")
    args = parser.parse_args()
    
    run_layer_sweep(step_size=args.step, use_huggingface=args.huggingface)
