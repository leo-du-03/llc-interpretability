#!/usr/bin/env python3
"""
Final comparison between TRACR compiled and traditionally trained transformers.

This script creates trained transformers that match the TRACR compiled models
in terms of architecture and parameters, then compares their performance
on the reverse sequence task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import random
from itertools import product

from reverse_models import check_reverse_small
from tracr.tracr.haiku_to_pytorch import haiku_to_pytorch, haiku_params_to_torch, infer_transformer_hparams
from torchinfo import summary

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class MatchingTransformer(nn.Module):
    """Transformer architecture that closely matches TRACR's structure."""
    
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len=10):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Custom transformer layers to match TRACR structure
        self.layers = nn.ModuleList([
            self._make_layer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
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
    
    def _init_weights(self):
        """Initialize weights similar to TRACR."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x):
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

def generate_training_data(vocab, num_samples=2000, max_len=6):
    """Generate training data for reverse sequence task."""
    data = []
    
    # Generate all short sequences exhaustively
    for length in range(1, 4):
        for seq_tuple in product(vocab, repeat=length):
            if len(data) >= num_samples:
                break
            seq = list(seq_tuple)
            input_seq = ["BOS"] + seq
            target_seq = ["BOS"] + seq[::-1]
            
            data.append((input_seq, target_seq))
    
    # Fill with random sequences
    while len(data) < num_samples:
        length = random.randint(1, min(max_len, 5))
        seq = [random.choice(list(vocab)) for _ in range(length)]
        input_seq = ["BOS"] + seq
        target_seq = ["BOS"] + seq[::-1]
        
        data.append((input_seq, target_seq))
    
    return data

def train_matching_model(vocab_size, d_model, num_layers, num_heads, d_ff, train_data, epochs=160, lr=1e-3):
    """Train a transformer model to match TRACR performance."""
    
    model = MatchingTransformer(vocab_size, d_model, num_layers, num_heads, d_ff)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    print(f"Training model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    model.train()
    best_loss = float('inf')
    
    # For accuracy testing during training
    vocab = {"a", "b", "c", "d", "e"}
    vocab_list = ["BOS"] + list(vocab)
    vocab_to_idx = {token: idx for idx, token in enumerate(vocab_list)}
    
    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(train_data)
        
        for input_seq, target_seq in train_data:
            input_ids = [vocab_to_idx[token] for token in input_seq]
            target_ids = [vocab_to_idx[token] for token in target_seq]
            
            max_len = 7
            input_padded = input_ids + [0] * (max_len - len(input_ids))
            target_padded = target_ids + [0] * (max_len - len(target_ids))
            
            input_tensor = torch.tensor([input_padded], dtype=torch.long)
            target_tensor = torch.tensor([target_padded], dtype=torch.long)
            
            optimizer.zero_grad()
            
            logits = model(input_tensor)
            
            # Calculate loss only on actual sequence positions
            loss = 0
            seq_len = len(input_ids)
            for i in range(seq_len):
                loss += criterion(logits[0, i:i+1], target_tensor[0, i:i+1])
            loss = loss / seq_len
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_data)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        # Print progress every 40 epochs (5 data points total: 0, 40, 80, 120, 159)
        if epoch % 10 == 0 or epoch == epochs - 1:
            # Test accuracy
            test_acc = test_model_accuracy(model, vocab, num_tests=100)
            print(f"  Epoch {epoch:3d}: Loss = {avg_loss:.4f}, Test Accuracy = {test_acc:.3f} (Best Loss: {best_loss:.4f})")
    
    model.eval()
    return model

def test_model_accuracy(model, vocab, num_tests=200):
    """Test model accuracy on random sequences."""
    vocab_list = ["BOS"] + list(vocab)
    vocab_to_idx = {token: idx for idx, token in enumerate(vocab_list)}
    idx_to_vocab = {idx: token for token, idx in vocab_to_idx.items()}
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for _ in range(num_tests):
            # Generate test sequence
            length = random.randint(1, 4)
            seq = [random.choice(list(vocab)) for _ in range(length)]
            input_seq = ["BOS"] + seq
            expected_seq = ["BOS"] + seq[::-1]
            
            # Convert to tensor
            input_ids = [vocab_to_idx[token] for token in input_seq]
            max_len = 7
            input_padded = input_ids + [0] * (max_len - len(input_ids))
            input_tensor = torch.tensor([input_padded], dtype=torch.long)
            
            # Get prediction
            logits = model(input_tensor)
            predicted_ids = torch.argmax(logits[0], dim=-1)[:len(input_seq)]
            predicted_seq = [idx_to_vocab[idx.item()] for idx in predicted_ids]
            
            # Check accuracy
            if predicted_seq == expected_seq:
                correct += 1
            total += 1
    
    return correct / total

def main():
    """Main comparison function."""
    set_seed(42)
    
    print("=" * 80)
    print("TRACR COMPILED vs TRADITIONALLY TRAINED TRANSFORMER COMPARISON")
    print("=" * 80)
    
    vocab = {"a", "b", "c", "d", "e"}
    vocab_list = ["BOS"] + list(vocab)
    vocab_to_idx = {token: idx for idx, token in enumerate(vocab_list)}
    idx_to_vocab = {idx: token for token, idx in vocab_to_idx.items()}
    
    # 1. TRACR Model Analysis
    print("\\n1. TRACR COMPILED MODEL")
    print("-" * 40)
    
    tracr_model = check_reverse_small()
    tracr_pytorch = haiku_to_pytorch(tracr_model)
    
    hk_params = haiku_params_to_torch(tracr_model.params)
    hparams = infer_transformer_hparams(hk_params)
    
    print(f"Architecture: {hparams['n_layers']} layers, {hparams['n_heads']} heads, d_model={hparams['d_model']}")
    print(f"Parameters: {sum(p.numel() for p in tracr_pytorch.parameters()):,}")
    
    # Test TRACR model
    test_cases = [
        ['BOS', 'a'],
        ['BOS', 'a', 'b'], 
        ['BOS', 'a', 'b', 'c'],
        ['BOS', 'c', 'd', 'e', 'a']
    ]
    
    print("\\nTRACR Test Results:")
    for test_input in test_cases:
        result = tracr_model.apply(test_input)
        expected = ["BOS"] + test_input[1:][::-1]
        correct = "✅" if result.decoded == expected else "❌"
        print(f"  {test_input} -> {result.decoded} {correct}")
    
    # 2. Train Matching Model
    print("\\n2. TRADITIONALLY TRAINED MODEL")
    print("-" * 40)
    
    # Generate training data
    print("Generating training data...")
    train_data = generate_training_data(vocab, num_samples=3000)
    print(f"Generated {len(train_data)} training examples")
    
    # Train model
    print("\\nTraining model...")
    trained_model = train_matching_model(
        vocab_size=len(vocab_list),
        d_model=hparams['d_model'],
        num_layers=hparams['n_layers'],
        num_heads=hparams['n_heads'],
        d_ff=hparams['d_ff'],
        train_data=train_data,
        epochs=50,
        lr=1e-3
    )
    
    print(f"\\nTrained model parameters: {sum(p.numel() for p in trained_model.parameters()):,}")
    
    # Save the trained model
    model_save_path = "trained_reverse_model.pth"
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'vocab_to_idx': vocab_to_idx,
        'idx_to_vocab': idx_to_vocab,
        'vocab': vocab,
        'model_params': {
            'd_model': hparams['d_model'],
            'n_layers': hparams['n_layers'], 
            'n_heads': hparams['n_heads'],
            'd_ff': hparams['d_ff'],
            'vocab_size': len(vocab_to_idx)
        }
    }, model_save_path)
    print(f"\\nTrained model saved to: {model_save_path}")
    
    # Test trained model
    print("\\nTrained Model Test Results:")
    for test_input in test_cases:
        if len(test_input) > 6:  # Skip if too long
            continue
            
        input_ids = [vocab_to_idx[token] for token in test_input]
        max_len = 7
        input_padded = input_ids + [0] * (max_len - len(input_ids))
        input_tensor = torch.tensor([input_padded], dtype=torch.long)
        
        with torch.no_grad():
            logits = trained_model(input_tensor)
            predicted_ids = torch.argmax(logits[0], dim=-1)[:len(test_input)]
            predicted_seq = [idx_to_vocab[idx.item()] for idx in predicted_ids]
        
        expected = ["BOS"] + test_input[1:][::-1]
        correct = "✅" if predicted_seq == expected else "❌"
        print(f"  {test_input} -> {predicted_seq} {correct}")
        if predicted_seq != expected:
            print(f"    Expected: {expected}")
    
    # 3. Accuracy Comparison
    print("\\n3. ACCURACY COMPARISON")
    print("-" * 40)
    
    # TRACR accuracy (should be 100%)
    tracr_correct = 0
    tracr_total = 0
    for _ in range(500):
        length = random.randint(1, 4)
        seq = [random.choice(list(vocab)) for _ in range(length)]
        test_input = ["BOS"] + seq
        expected = ["BOS"] + seq[::-1]
        
        result = tracr_model.apply(test_input)
        if result.decoded == expected:
            tracr_correct += 1
        tracr_total += 1
    
    tracr_accuracy = tracr_correct / tracr_total
    
    # Trained model accuracy
    trained_accuracy = test_model_accuracy(trained_model, vocab, num_tests=500)
    
    print(f"TRACR Model Accuracy:  {tracr_accuracy:.3f} ({tracr_correct}/{tracr_total})")
    print(f"Trained Model Accuracy: {trained_accuracy:.3f}")
    
    print("\\n" + "=" * 80)
    print("CONCLUSION:")
    print(f"  • TRACR provides perfect algorithmic solution ({tracr_accuracy:.1%} accuracy)")
    print(f"  • Trained model learns approximate solution ({trained_accuracy:.1%} accuracy)")
    print(f"  • Both models have similar parameter counts (~{sum(p.numel() for p in tracr_pytorch.parameters())//1000}K parameters)")
    print(f"  • TRACR guarantees correctness, trained model learns from data")
    print("=" * 80)

if __name__ == "__main__":
    main()
