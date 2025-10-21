import torch
from torch.utils.data import DataLoader, TensorDataset
from devinterp.slt.sampler import estimate_learning_coeff_with_summary
from devinterp.optim import SGLD
from devinterp.utils import plot_trace, default_nbeta



def estimate_llc(model, dataloader, sampling_method='local', num_chains=5,
                 num_draws=100, device='cpu'):
    result = estimate_learning_coeff_with_summary(
        model,
        loader=dataloader,
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
    

