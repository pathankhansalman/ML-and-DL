import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataset import get_train_test_split

class PyTorchLogisticRegression(nn.Module):
    """
    Logistic Regression implemented from scratch in PyTorch.
    Uses Sigmoid activation for binary classification.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

class PyTorchLinearRegression(nn.Module):
    """
    Linear Regression implemented from scratch in PyTorch.
    Predicts continuous sentiment output (0 to 1).
    """
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

class ActivationExtractor:
    """
    Helper class to manage forward hooks and store activations.
    """
    def __init__(self):
        self.activations = {}

    def get_hook(self, layer_num):
        def hook(module, input, output):
            # Qwen output is a tuple (hidden_states, optional_caches_etc)
            # The residual stream tensor is the first element
            hidden_states = output[0] if isinstance(output, tuple) else output
            self.activations[layer_num] = hidden_states.detach()
        return hook

def extract_last_token_activations(model, tokenizer, texts, layer_num, device):
    """
    Runs text batches through the model and extracts residual stream activations
    for the last token position at the specified layer_num.
    """
    extractor = ActivationExtractor()
    
    # Qwen layer path in Hugging Face: model.model.layers[layer_num]
    target_layer = model.model.layers[layer_num]
    hook_handle = target_layer.register_forward_hook(extractor.get_hook(layer_num))

    all_activations = []
    
    model.eval()
    with torch.no_grad():
        for text in texts:
            # Tokenize single sentence (batch size = 1 for simplicity and low VRAM)
            inputs = tokenizer(text, return_tensors="pt").to(device)
            # Forward pass
            model(**inputs)
            
            # Retrieve cached activations: Shape [batch_size, sequence_length, hidden_dim]
            layer_act = extractor.activations[layer_num]
            
            # Extract last token's activation vector: Shape [hidden_dim]
            # Since batch_size = 1, sequence_length - 1 is the last token position
            last_token_act = layer_act[0, -1, :]
            
            all_activations.append(last_token_act.cpu())
            
    # Clean up the hook so we don't leak memory
    hook_handle.remove()
    
    return torch.stack(all_activations)

def train_probe(model_type, train_acts, train_labels, test_acts, test_labels, epochs=100, lr=0.005, weight_decay=0.01):
    """
    Trains a PyTorch probe (Logistic or Linear) using Adam and L2 regularization (weight_decay).
    """
    train_acts = train_acts.float()
    test_acts = test_acts.float()
    input_dim = train_acts.shape[1]
    
    if model_type == "logistic":
        probe = PyTorchLogisticRegression(input_dim)
        criterion = nn.BCELoss()
    else:
        probe = PyTorchLinearRegression(input_dim)
        criterion = nn.MSELoss()
        
    optimizer = optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Convert labels to PyTorch tensors
    y_train = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(test_labels, dtype=torch.float32).unsqueeze(1)
    
    # Training loop
    for epoch in range(epochs):
        probe.train()
        optimizer.zero_grad()
        outputs = probe(train_acts)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # Evaluation
    probe.eval()
    with torch.no_grad():
        test_outputs = probe(test_acts)
        if model_type == "logistic":
            predictions = (test_outputs >= 0.5).float()
            accuracy = (predictions == y_test).float().mean().item()
            print(f"[{model_type.upper()} PROBE] Test Accuracy: {accuracy * 100:.2f}% | Loss: {loss.item():.4f}")
            return accuracy
        else:
            test_loss = criterion(test_outputs, y_test).item()
            print(f"[{model_type.upper()} PROBE] Test MSE Loss: {test_loss:.4f} | Train Loss: {loss.item():.4f}")
            return test_loss

def run_probing_experiment(layer_num=12, use_huggingface=False):
    """
    Complete pipeline: Loads Qwen, extracts activations at layer_num, and trains/evaluates probes.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load data
    train_texts, test_texts, train_labels, test_labels = get_train_test_split(use_huggingface=use_huggingface)
    
    # 2. Load model & tokenizer
    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"Loading {model_name} in float16...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        device_map=device
    )
    
    # 3. Extract activations
    print(f"Extracting activations from Layer {layer_num} at the last token position...")
    train_acts = extract_last_token_activations(model, tokenizer, train_texts, layer_num, device)
    test_acts = extract_last_token_activations(model, tokenizer, test_texts, layer_num, device)
    
    print(f"Activation tensor shape: {train_acts.shape} (Num sentences, Hidden dimension)")
    
    # 4. Train probes
    train_probe("logistic", train_acts, train_labels, test_acts, test_labels)
    train_probe("linear", train_acts, train_labels, test_acts, test_labels)

if __name__ == "__main__":
    # Test on Layer 12, using the local 100-sentence dataset
    run_probing_experiment(layer_num=12, use_huggingface=False)
