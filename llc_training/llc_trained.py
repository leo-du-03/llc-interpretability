import torch
from torch.nn import functional as F
from devinterp.optim import SGLD
from devinterp.slt.sampler import estimate_learning_coeff_with_summary
from devinterp.utils import default_nbeta
from datasets.dataloaders import makeFractokDataLoader, CHAR2IDX
from train_fractok import TransformerForFractok
from rasp_models.fractok import check_fractok
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load your trained model
model = TransformerForFractok(
    vocab_size=len(CHAR2IDX),
    max_seq_len=10,
    d_model=64,
    n_heads=4,
    n_layers=4,
    dropout=0.1
).to(DEVICE)

model.load_state_dict(torch.load('fractok_trained.pt'))
model.eval()

# Create the compiled model to figure out which dimension is fractok
compiled_model = check_fractok(max_seq_len=10, vocab_size='medium')

# Find the fractok dimension by checking residual labels
# This assumes your model has a way to access residual stream labels
# You might need to inspect compiled_model to find the right attribute
try:
    # Try to find the frac_prevs dimension
    # This is model-specific - adjust based on your fractok.py implementation
    residual_labels = compiled_model.residual_labels  # or whatever attribute has the labels
    frac_prevs_idx = residual_labels.index('frac_prevs')
    print(f"Found frac_prevs at dimension {frac_prevs_idx}")
except:
    # If we can't find it automatically, we'll compute ground truth
    print("Could not find frac_prevs dimension, computing ground truth instead")
    frac_prevs_idx = None

# Create test loader
test_loader = makeFractokDataLoader(max_seq_len=10, vocab_size='medium')


print("\n=== Checking Test Loss ===")
model.eval()
test_losses = []
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        inputs, outputs = batch[0]
        
        # Tokenize inputs
        input_ids = torch.tensor(
            [CHAR2IDX.get(token, 0) for token in inputs], 
            dtype=torch.long
        ).unsqueeze(0).to(DEVICE)
        
        # Compute ground truth
        fractok_values = []
        for j in range(len(inputs)):
            if j == 0:  # BOS
                fractok_values.append(0.0)
            else:
                num_x = sum(1 for t in inputs[1:j+1] if t == 'x')
                fractok_values.append(num_x / j)
        targets = torch.tensor(fractok_values, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        # Forward pass
        model_output = model.forward(input_ids)
        
        # Calculate loss (with and without scaling)
        raw_loss = F.mse_loss(model_output, targets)
        scaled_loss = raw_loss * 1000
        
        test_losses.append(raw_loss.item())
        
        if i == 0:  # Print first batch details
            print(f"\nFirst batch example:")
            print(f"  Input: {inputs}")
            print(f"  Targets: {targets.cpu().numpy()}")
            print(f"  Predictions: {model_output.cpu().numpy()}")
            print(f"  Raw loss: {raw_loss.item():.6f}")
            print(f"  Scaled loss (1000x): {scaled_loss.item():.6f}")
        
        if i >= 9:  # Check first 10 batches
            break

avg_test_loss = sum(test_losses) / len(test_losses)
print(f"\nAverage test loss (raw): {avg_test_loss:.6f}")
print(f"Average test loss (scaled 1000x): {avg_test_loss * 1000:.6f}")

# Evaluation function for trained model
def evaluate_trained(model, batch):
    inputs, outputs = batch[0]
    
    # Tokenize string inputs to IDs
    input_ids = torch.tensor(
        [CHAR2IDX.get(token, 0) for token in inputs], 
        dtype=torch.long
    ).unsqueeze(0).to(DEVICE)
    
    # Extract just the fractok values from the compiled model output
    if frac_prevs_idx is not None:
        # Extract the specific dimension
        targets = outputs[:, frac_prevs_idx].to(DEVICE)
    else:
        # Compute ground truth directly from inputs
        fractok_values = []
        for i in range(len(inputs)):
            if i == 0:  # BOS
                fractok_values.append(0.0)
            else:
                num_x = sum(1 for t in inputs[1:i+1] if t == 'x')
                fractok_values.append(num_x / i)
        targets = torch.tensor(fractok_values, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    # Forward pass
    model_output = model.forward(input_ids)
    
    # Same loss scaling as compiled model (1000x)
    loss = F.mse_loss(model_output, targets) * 1000
    
    return loss, {"logits": model_output}

# Estimate LLC
print("Estimating LLC for trained model...")
learning_coeff_stats = estimate_learning_coeff_with_summary(
    model,
    loader=test_loader,
    evaluate=evaluate_trained,
    sampling_method=SGLD,
    optimizer_kwargs=dict(
        lr=1e-5, 
        localization=1.0, 
        nbeta=default_nbeta(test_loader)
    ),
    num_chains=10,
    num_draws=100,
    num_burnin_steps=0,
    num_steps_bw_draws=1,
    device=DEVICE,
    online=True,
)

avg_llc = sum(learning_coeff_stats['llc/means']) / len(learning_coeff_stats['llc/means'])
print(f"\nTrained Model LLC: {avg_llc:.2f}")

# Optional: plot the trace
from devinterp.utils import plot_trace

trace = learning_coeff_stats["loss/trace"]
plot_trace(
    trace,
    "Loss",
    x_axis="Step",
    title=f"Trained Model Loss Trace, avg LLC = {avg_llc}",
    plot_mean=False,
    plot_std=False,
    fig_size=(12, 9),
    true_lc=None,
)