import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from dataloaders import makeFractokTrainDataLoader, CHAR2IDX

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TransformerForFractok(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_seq_len,
        d_model=64,
        n_heads=4,
        n_layers=4,
        dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output head for regression (fraction prediction)
        self.output_head = nn.Linear(d_model, 1)
        
    def forward(self, x):
        # x: (batch, seq_len) of token indices
        batch_size, seq_len = x.shape
        
        # Embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(x) + self.pos_embedding(positions)
        
        # Transformer with causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        x = self.transformer(x, mask=causal_mask)
        
        # Output: predict fraction at each position
        output = self.output_head(x).squeeze(-1)  # (batch, seq_len)
        output = torch.sigmoid(output)  # Ensure [0, 1] range
        
        return output


def train_fractok_model(
    model,
    train_loader,
    epochs=100,
    lr=1e-3,
    weight_decay=0.01,
    device='cuda',
    save_path='fractok_trained.pt'
):
    """
    Train a transformer model on the fractok task.
    
    Args:
        model: TransformerForFractok model
        train_loader: DataLoader with tokenized inputs
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for AdamW
        device: 'cuda' or 'cpu'
        save_path: Path to save trained model
        
    Returns:
        model: Trained model
        train_losses: List of training losses per epoch
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        for input_ids, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids)
            
            # MSE loss for regression
            loss = F.mse_loss(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        train_losses.append(avg_loss)
        
        print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}, LR = {scheduler.get_last_lr()[0]:.6f}")
        
        scheduler.step()
    
    # Save the trained model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    return model, train_losses


def test_model_accuracy(model, test_loader, device='cuda'):
    """
    Test model accuracy on fractok task.
    
    Returns:
        avg_loss: Average MSE loss
        avg_mae: Average mean absolute error
    """
    model.eval()
    model = model.to(device)
    
    total_loss = 0
    total_mae = 0
    n_batches = 0
    
    with torch.no_grad():
        for input_ids, targets in test_loader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            
            outputs = model(input_ids)
            
            loss = F.mse_loss(outputs, targets)
            mae = F.l1_loss(outputs, targets)
            
            total_loss += loss.item()
            total_mae += mae.item()
            n_batches += 1
    
    avg_loss = total_loss / n_batches
    avg_mae = total_mae / n_batches
    
    print(f"Test Loss (MSE): {avg_loss:.6f}")
    print(f"Test MAE: {avg_mae:.6f}")
    
    return avg_loss, avg_mae


if __name__ == "__main__":
    # Configuration
    MAX_SEQ_LEN = 10
    VOCAB_SIZE = 'medium'
    BATCH_SIZE = 32
    EPOCHS = 200
    
    # Create data loader
    train_loader = makeFractokTrainDataLoader(
        max_seq_len=MAX_SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # Create model
    model = TransformerForFractok(
        vocab_size=len(CHAR2IDX),
        max_seq_len=MAX_SEQ_LEN,
        d_model=64,
        n_heads=4,
        n_layers=4,
        dropout=0.1
    )
    
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    trained_model, losses = train_fractok_model(
        model,
        train_loader,
        epochs=EPOCHS,
        lr=1e-3,
        device=DEVICE,
        save_path='fractok_trained.pt'
    )
    
    # Test model
    test_loader = makeFractokTrainDataLoader(
        max_seq_len=MAX_SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        batch_size=BATCH_SIZE
    )
    
    test_loss, test_mae = test_model_accuracy(trained_model, test_loader, device=DEVICE)
    
    # Plot training curve
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.savefig('training_curve.png', dpi=300)
    print("Training curve saved to training_curve.png")