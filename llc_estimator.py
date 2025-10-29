import torch
from torch.utils.data import DataLoader, TensorDataset
from devinterp.slt.sampler import estimate_learning_coeff_with_summary
from devinterp.optim import SGLD
from devinterp.utils import plot_trace, default_nbeta
from dataloaders import makePeakDataLoader
from tracr.haiku_to_pytorch import haiku_to_pytorch
from peak import get_peak_model

class DifferentiableTracr(torch.nn.Module):
    def __init__(self, tracr_pytorch_model):
        super().__init__()
        self.tracr_model = tracr_pytorch_model
        # Store all parameters as a list to avoid name issues
        self.param_list = torch.nn.ParameterList([
            torch.nn.Parameter(param.clone()) 
            for param in tracr_pytorch_model.parameters()
        ])
    
    def forward(self, input_seq):
        # Apply model with current parameters - just use the model as-is
        # The issue is tracr models aren't really differentiable
        output_dict = self.tracr_model.model.apply(input_seq)
        decoded = output_dict.decoded[1:]  # Skip BOS
        return torch.tensor([float(x) for x in decoded], dtype=torch.float32)

def estimate_llc(model, dataloader, sampling_method='local', num_chains=5,
                 num_draws=100, device='cpu'):
    
    def loss_fn(model, batch):
        inputs, targets = batch
        
        batch_outputs = []
        for input_seq in inputs:
            # Pass raw sequence directly - model expects ['BOS', 1, 2, 3] format
            output = model(input_seq)  # DifferentiableTracr.forward handles this
            batch_outputs.append(output)
        
        outputs = torch.stack(batch_outputs).to(device)
        targets = torch.tensor(targets) if not isinstance(targets, torch.Tensor) else targets
        targets = targets.to(device).float()
        
        # MSE loss
        loss = torch.nn.functional.mse_loss(outputs, targets)
        return loss
    
    result = estimate_learning_coeff_with_summary(
        model,
        loader=dataloader,
        evaluate=loss_fn,
        optimizer_kwargs=dict(lr=4e-4, localization=100.0, nbeta=default_nbeta(dataloader)),
        sampling_method=SGLD,
        num_chains=num_chains,
        num_draws=num_draws,
        device=device
    )
    return result



def print_llc_results(result, model_name):
    print(f"LLC Results for {model_name}")
    print(f"Learning Coefficient (λ): {result['lambda']:.4f}")
    print(f"Standard Error: {result['lambda_stderr']:.4f}")
    print(f"RLCT: {result['rlct']:.4f}")

def plot_llc_trace(result, model_name, save_path=None):
    trace = result.get('trace', None)
    if trace is None:
        print("No trace data available in results.")
        return
    
    # Calculate average LLC from the trace
    llc_means = result.get('llc_means', [])
    if llc_means:
        avg_llc = sum(llc_means) / len(llc_means)
    else:
        avg_llc = result.get('lambda', 0)
    
    # Plot the trace
    plot_trace(
        trace,
        "Loss",
        x_axis="Step",
        title=f"{model_name} - Loss Trace, avg LLC = {avg_llc:.2f}",
        plot_mean=False,
        plot_std=False,
        fig_size=(12, 9),
        true_lc=None,
    )
    
    # Save if path provided
    if save_path:
        import matplotlib.pyplot as plt
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    import matplotlib.pyplot as plt
    plt.show()
    
    

# pal_loader = makePalindromeDataLoader()
peak_loader = makePeakDataLoader()
# pal_model = haiku_to_pytorch(check_palindrome())
peak_model = DifferentiableTracr(haiku_to_pytorch(get_peak_model()))
# pal_result = estimate_llc(pal_model, pal_loader, num_chains=3, num_draws=50)
peak_result = estimate_llc(peak_model, peak_loader, num_chains=3, num_draws=50)
print_llc_results(peak_result, "Peak Model")
