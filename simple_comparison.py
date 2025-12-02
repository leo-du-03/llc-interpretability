import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np

from reverse_models import check_reverse_small
from tracr.tracr.haiku_to_pytorch import haiku_to_pytorch, haiku_params_to_torch, infer_transformer_hparams
from torchinfo import summary

def simple_comparison():
    """Simple comparison focusing on architecture matching."""
    
    print("="*60)
    print("SIMPLE TRACR vs TRAINED TRANSFORMER COMPARISON")
    print("="*60)
    
    vocab = {"a", "b", "c", "d", "e"}
    vocab_list = ["BOS"] + list(vocab)
    vocab_to_idx = {token: idx for idx, token in enumerate(vocab_list)}
    idx_to_vocab = {idx: token for token, idx in vocab_to_idx.items()}
    
    # 1. Get TRACR model details
    print("\n1. TRACR MODEL ANALYSIS:")
    tracr_model = check_reverse_small()
    tracr_pytorch = haiku_to_pytorch(tracr_model)
    
    print(f"TRACR Model Architecture:")
    summary(tracr_pytorch)
    
    hk_params = haiku_params_to_torch(tracr_model.params)
    hparams = infer_transformer_hparams(hk_params)
    
    print(f"\nTRACR Hyperparameters:")
    for key, value in hparams.items():
        print(f"  {key}: {value}")
    
    # Test TRACR on simple examples
    test_cases = [['BOS', 'a'], ['BOS', 'a', 'b'], ['BOS', 'a', 'b', 'c']]
    print(f"\nTRACR Test Results:")
    for test_input in test_cases:
        result = tracr_model.apply(test_input)
        print(f"  {test_input} -> {result.decoded}")
    
    # 2. Create matching trained model architecture
    print(f"\n2. TRAINED MODEL WITH MATCHING ARCHITECTURE:")
    
    class SimpleTrainedTransformer(nn.Module):
        def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff):
            super().__init__()
            self.d_model = d_model
            
            # Embeddings (simplified - no positional for now)
            self.embedding = nn.Embedding(vocab_size, d_model)
            
            # Transformer layers (simplified)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_ff,
                dropout=0.0,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            # Output projection
            self.output_proj = nn.Linear(d_model, vocab_size)
            
        def forward(self, x):
            # Embed
            x = self.embedding(x) * np.sqrt(self.d_model)
            # Transform
            x = self.transformer(x)
            # Project to vocab
            return self.output_proj(x)
    
    # Create model with same parameters as TRACR
    trained_model = SimpleTrainedTransformer(
        vocab_size=len(vocab_list),
        d_model=hparams['d_model'],
        num_layers=hparams['n_layers'],
        num_heads=hparams['n_heads'],
        d_ff=hparams['d_ff']
    )
    
    print(f"Trained Model Architecture:")
    summary(trained_model, input_size=(1, 5), dtypes=[torch.long])
    
    print(f"\nParameter Comparison:")
    tracr_params = sum(p.numel() for p in tracr_pytorch.parameters())
    trained_params = sum(p.numel() for p in trained_model.parameters())
    print(f"  TRACR model: {tracr_params:,} parameters")
    print(f"  Trained model: {trained_params:,} parameters")
    print(f"  Difference: {abs(tracr_params - trained_params):,} parameters")
    
    # 3. Quick training test
    print(f"\n3. QUICK TRAINING TEST:")
    
    # Create simple training data
    train_data = []
    for _ in range(500):
        # Random sequence of length 2-4
        length = np.random.randint(2, 5)
        seq = [np.random.choice(list(vocab)) for _ in range(length)]
        input_seq = ["BOS"] + seq
        target_seq = ["BOS"] + seq[::-1]
        
        input_ids = [vocab_to_idx[token] for token in input_seq]
        target_ids = [vocab_to_idx[token] for token in target_seq]
        
        train_data.append((input_ids, target_ids))
    
    # Simple training loop
    optimizer = optim.Adam(trained_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    trained_model.train()
    for epoch in range(100):
        total_loss = 0
        for input_ids, target_ids in train_data:
            # Pad to same length
            max_len = 5
            input_padded = input_ids + [0] * (max_len - len(input_ids))
            target_padded = target_ids + [0] * (max_len - len(target_ids))
            
            input_tensor = torch.tensor([input_padded], dtype=torch.long)
            target_tensor = torch.tensor([target_padded], dtype=torch.long)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = trained_model(input_tensor)
            
            # Loss only on actual sequence positions
            loss = 0
            for i in range(len(input_ids)):
                loss += criterion(logits[0, i:i+1], target_tensor[0, i:i+1])
            loss = loss / len(input_ids)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: Average Loss = {total_loss/len(train_data):.4f}")
    
    # Test trained model
    print(f"\nTrained Model Test Results:")
    trained_model.eval()
    with torch.no_grad():
        for test_input in test_cases:
            if len(test_input) > 5:
                continue
                
            input_ids = [vocab_to_idx[token] for token in test_input]
            input_padded = input_ids + [0] * (5 - len(input_ids))
            input_tensor = torch.tensor([input_padded], dtype=torch.long)
            
            logits = trained_model(input_tensor)
            predicted_ids = torch.argmax(logits[0], dim=-1)[:len(test_input)]
            predicted_tokens = [idx_to_vocab[idx.item()] for idx in predicted_ids]
            
            expected = ["BOS"] + test_input[1:][::-1]
            correct = "✅" if predicted_tokens == expected else "❌"
            
            print(f"  {test_input} -> {predicted_tokens} {correct}")
            if predicted_tokens != expected:
                print(f"    Expected: {expected}")

if __name__ == "__main__":
    simple_comparison()
