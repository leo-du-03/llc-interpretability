"""
Utility functions for loading saved trained models.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

class SimpleTrainedTransformer(nn.Module):
    """Transformer architecture that matches the structure used in final_comparison.py"""

    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, d_ff: int, max_seq_len: int = 10):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Custom transformer layers to match TRACR structure
        self.layers = nn.ModuleList([
            self._make_layer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

    def _make_layer(self, d_model, num_heads, d_ff):
        """Create a single transformer layer."""
        return nn.ModuleDict({
            'attention': nn.MultiheadAttention(d_model, num_heads, batch_first=True),
            'norm1': nn.LayerNorm(d_model),
            'mlp': nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model)
            ),
            'norm2': nn.LayerNorm(d_model)
        })

    def forward(self, x):
        # x shape: [batch_size, seq_len]
        B, T = x.shape

        # Embeddings
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        x = self.token_embedding(x) + self.position_embedding(positions)

        # Apply transformer layers
        for layer in self.layers:
            # Attention
            residual = x
            x = layer['norm1'](x)
            attn_out, _ = layer['attention'](x, x, x)
            x = residual + attn_out

            # MLP
            residual = x
            x = layer['norm2'](x)
            x = residual + layer['mlp'](x)

        # Output projection
        return self.output_proj(x)


def load_trained_model(model_path: str = "trained_reverse_model.pth") -> Tuple[SimpleTrainedTransformer, Dict[str, Any]]:
    """
    Load a saved trained model with its metadata.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        model: The loaded trained model
        metadata: Dictionary containing vocab mappings and model parameters
    """
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extract metadata
    vocab_to_idx = checkpoint['vocab_to_idx']
    idx_to_vocab = checkpoint['idx_to_vocab'] 
    vocab = checkpoint['vocab']
    model_params = checkpoint['model_params']
    
    # Create model with saved parameters
    model = SimpleTrainedTransformer(
        vocab_size=model_params['vocab_size'],
        d_model=model_params['d_model'],
        n_layers=model_params['n_layers'],
        n_heads=model_params['n_heads'],
        d_ff=model_params['d_ff']
    )
    
    # Load the model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    metadata = {
        'vocab_to_idx': vocab_to_idx,
        'idx_to_vocab': idx_to_vocab,
        'vocab': vocab,
        'model_params': model_params
    }
    
    return model, metadata


def test_loaded_model(model: SimpleTrainedTransformer, metadata: Dict[str, Any], test_cases: list = None):
    """
    Test a loaded model with some example cases.
    
    Args:
        model: The loaded trained model
        metadata: Metadata dictionary from load_trained_model
        test_cases: Optional list of test cases, if None uses default reverse cases
    """
    
    model.eval()
    vocab_to_idx = metadata['vocab_to_idx']
    idx_to_vocab = metadata['idx_to_vocab']
    
    if test_cases is None:
        test_cases = [
            ['BOS', 'a'],
            ['BOS', 'a', 'b'],
            ['BOS', 'a', 'b', 'c'],
            ['BOS', 'c', 'd', 'e']
        ]
    
    print("Testing loaded model:")
    for test_input in test_cases:
        # Convert to indices
        input_ids = [vocab_to_idx[token] for token in test_input]
        max_len = 7  # Should match training max length
        input_padded = input_ids + [0] * (max_len - len(input_ids))
        input_tensor = torch.tensor([input_padded], dtype=torch.long)
        
        # Get prediction
        with torch.no_grad():
            logits = model(input_tensor)
            predicted_ids = torch.argmax(logits[0], dim=-1)[:len(test_input)]
            predicted_seq = [idx_to_vocab[idx.item()] for idx in predicted_ids]
        
        # Expected reverse sequence
        expected = ["BOS"] + test_input[1:][::-1]
        correct = "✅" if predicted_seq == expected else "❌"
        
        print(f"  {test_input} -> {predicted_seq} {correct}")
        if predicted_seq != expected:
            print(f"    Expected: {expected}")


if __name__ == "__main__":
    # Example usage
    try:
        model, metadata = load_trained_model()
        print(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"Vocabulary: {metadata['vocab']}")
        print(f"Model architecture: {metadata['model_params']}")
        print()
        test_loaded_model(model, metadata)
        
    except FileNotFoundError:
        print("No saved model found. Run final_comparison.py first to train and save a model.")
    except Exception as e:
        print(f"Error loading model: {e}")
