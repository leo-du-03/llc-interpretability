import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import copy

from reverse_models import check_reverse_small
from reverse_training_data import create_reverse_dataloader, test_reverse_accuracy
from tracr.tracr.haiku_to_pytorch import haiku_to_pytorch, haiku_params_to_torch, infer_transformer_hparams
from torchinfo import summary

"""Attention block matching TRACR architecture"""
class TrainedAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_k, d_v):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_v
        self.d_k = d_k

        self.query = nn.Linear(d_model, d_k, bias=True)
        self.key = nn.Linear(d_model, d_k, bias=True)
        self.value = nn.Linear(d_model, d_v, bias=True)
        self.output = nn.Linear(num_heads * d_v, d_model, bias=True)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            att = att.masked_fill(mask == 0, -1e9)
            
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.output(out)

"""MLP block matching TRACR architecture"""
class TrainedMLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

"""Transformer block matching TRACR architecture"""
class TrainedBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, d_k, d_v):
        super().__init__()
        self.attn = TrainedAttention(d_model, num_heads, d_k, d_v)
        self.mlp = TrainedMLP(d_model, d_ff)

    def forward(self, x, mask=None):
        x = x + self.attn(x, mask)
        x = x + self.mlp(x)
        return x

"""Trained transformer matching TRACR architecture exactly"""
class TrainedTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, d_k, d_v, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TrainedBlock(d_model, num_heads, d_ff, d_k, d_v) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, mask=None):
        B, T = x.shape
        
        # Create embeddings
        positions = torch.arange(0, T, device=x.device).unsqueeze(0).expand(B, T)
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        x = token_emb + pos_emb
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Output projection
        logits = self.output_projection(x)
        return logits

def create_trained_reverse_model(vocab, max_seq_len=10):
    """Create a trained transformer that matches TRACR reverse model architecture."""
    
    # Get TRACR model architecture parameters
    tracr_model = check_reverse_small()
    hk_params = haiku_params_to_torch(tracr_model.params)
    hparams = infer_transformer_hparams(hk_params)
    
    vocab_size = len(vocab) + 1  # +1 for BOS token
    
    print("TRACR Model Hyperparameters:")
    for key, value in hparams.items():
        print(f"  {key}: {value}")
    print(f"  vocab_size: {vocab_size}")
    print(f"  max_seq_len: {max_seq_len}")
    
    # Create matching trained model
    model = TrainedTransformer(
        vocab_size=vocab_size,
        d_model=hparams['d_model'],
        num_layers=hparams['n_layers'],
        num_heads=hparams['n_heads'],
        d_ff=hparams['d_ff'],
        d_k=hparams['d_k'],
        d_v=hparams['d_v'],
        max_seq_len=max_seq_len
    )
    
    return model, hparams

def train_reverse_model(vocab, epochs=10, max_seq_len=10, num_samples=2000, batch_size=32, lr=1e-3):
    """Train a transformer to perform reverse sequence task."""
    
    # Create model and dataloader
    model, hparams = create_trained_reverse_model(vocab, max_seq_len)
    dataloader = create_reverse_dataloader(vocab, max_seq_len, num_samples, batch_size)
    
    print(f"\nTraining transformer with {sum(p.numel() for p in model.parameters())} parameters")
    print("Model Architecture:")
    summary(model, input_size=(batch_size, max_seq_len), dtypes=[torch.long])
    
    # Training setup
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    model.train()
    
    print(f"\nTraining for {epochs} epochs...")
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (inputs, targets, lengths) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(inputs)
            
            # Create mask for loss calculation (only calculate loss on actual sequence positions)
            mask = torch.zeros_like(targets, dtype=torch.bool)
            for i, length in enumerate(lengths):
                mask[i, :length] = True
            
            # Calculate loss only on non-padded positions
            loss = criterion(logits[mask], targets[mask])
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Test accuracy every few epochs
        if epoch % 5 == 0 or epoch == epochs - 1:
            accuracy = test_reverse_accuracy(model, vocab, max_seq_len, num_test=100)
            print(f"Epoch {epoch + 1:2d}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.3f}")
        else:
            print(f"Epoch {epoch + 1:2d}: Loss = {avg_loss:.4f}")
    
    model.eval()
    return model

def compare_models():
    """Compare TRACR compiled model vs trained model."""
    vocab = {"a", "b", "c", "d", "e"}
    max_seq_len = 10
    
    print("="*60)
    print("COMPARING TRACR COMPILED vs TRAINED TRANSFORMERS")
    print("="*60)
    
    # TRACR model
    print("\n1. TRACR COMPILED MODEL:")
    tracr_model = check_reverse_small()
    tracr_pytorch = haiku_to_pytorch(tracr_model)
    print("Architecture:")
    summary(tracr_pytorch)
    
    # Test TRACR model
    test_cases = [
        ['BOS', 'a', 'b'],
        ['BOS', 'c', 'd', 'e'],
        ['BOS', 'a', 'b', 'c', 'd']
    ]
    
    print("\nTRACR Model Test Results:")
    for test_input in test_cases:
        result = tracr_model.apply(test_input)
        print(f"  {test_input} -> {result.decoded}")
    
    # Trained model
    print(f"\n2. TRAINED MODEL:")
    trained_model = train_reverse_model(vocab, epochs=50, max_seq_len=max_seq_len, num_samples=3000, batch_size=16, lr=1e-3)
    
    # Test trained model
    print("\nTrained Model Test Results:")
    vocab_to_idx = {token: idx for idx, token in enumerate(["BOS"] + list(vocab))}
    idx_to_vocab = {idx: token for token, idx in vocab_to_idx.items()}
    
    trained_model.eval()
    with torch.no_grad():
        for test_input in test_cases:
            # Convert to indices
            input_indices = [vocab_to_idx[token] for token in test_input]
            input_padded = input_indices + [0] * (max_seq_len - len(input_indices))
            input_tensor = torch.tensor([input_padded], dtype=torch.long)
            
            # Get prediction
            output = trained_model(input_tensor)
            predicted_indices = torch.argmax(output[0], dim=-1)[:len(test_input)]
            predicted_seq = [idx_to_vocab[idx.item()] for idx in predicted_indices]
            
            # Expected output
            expected = ["BOS"] + test_input[1:][::-1]
            correct = "✅" if predicted_seq == expected else "❌"
            
            print(f"  {test_input} -> {predicted_seq} {correct}")
            if predicted_seq != expected:
                print(f"    Expected: {expected}")
    
    # Final accuracy test
    final_accuracy = test_reverse_accuracy(trained_model, vocab, max_seq_len, num_test=500)
    print(f"\nFinal Trained Model Accuracy: {final_accuracy:.3f}")
    
    return tracr_model, trained_model

if __name__ == "__main__":
    compare_models()
