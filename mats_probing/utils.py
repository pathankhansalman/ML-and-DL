import torch

class ActivationCache:
    """
    A generic helper class to manage forward hooks and store activations.
    Works for standard PyTorch runs and Hugging Face model forward passes.
    """
    def __init__(self):
        self.cache = {}
        # Alias for backward compatibility
        self.activations = self.cache

    def get_hook(self, layer_idx):
        def hook(module, input, output):
            # hidden_states is typically the first element of output tuple
            hidden_states = output[0] if isinstance(output, tuple) else output
            self.cache[layer_idx] = hidden_states.detach().clone()
        return hook

def patch_hook_builder(target_token_idx, source_activation):
    """
    Builds a hook that replaces the activation at a specific token position
    with the corresponding source activation vector.
    """
    def hook(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0].clone()
            hidden_states[:, target_token_idx, :] = source_activation[:, target_token_idx, :]
            return (hidden_states,) + output[1:]
        else:
            out = output.clone()
            out[:, target_token_idx, :] = source_activation[:, target_token_idx, :]
            return out
    return hook

def find_token_idx(tokenizer, text, target_word):
    """
    Finds the token index of a specific target word in the tokenized text.
    """
    tokens = tokenizer.tokenize(text)
    for idx, token in enumerate(tokens):
        decoded = tokenizer.convert_tokens_to_string([token])
        if target_word.strip().lower() in decoded.strip().lower():
            return idx
    return len(tokens) - 1  # Fallback to last token if not found

class ActivationCollector:
    """
    Collects activations sequentially across multiple inputs (e.g. for training).
    """
    def __init__(self):
        self.activations = []

    def hook(self, module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        self.activations.append(hidden_states.detach().clone())

