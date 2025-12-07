from devinterp.slt.sampler import estimate_learning_coeff_with_summary
from tracr.rasp import rasp
from tracr.compiler import compiling


from llc_estimator import (
    test_to_tensors,
    create_dataloader,
    estimate_llc,
    print_llc_results,
    plot_llc_trace
)

from rasp_models import peak
def get_peak_model():
    bos = "BOS"
    model = compiling.compile_rasp_to_model(
        peak.peak(),
        vocab={0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        max_seq_len=10,
        compiler_bos=bos,
    )
    return model

from datasets.peak_data import get_peak_test_cases
def get_peak_test_data(): # get the test_cases from test_peak.py (cleaner)
    return get_peak_test_cases() 


def main():
    model = get_peak_model()
    test_data = get_peak_test_data()
    inputs, targets = test_to_tensors(test_data)
    dataloader = create_dataloader(inputs, targets, batch_size=8)
    print("Starting LLC estimation (this may take a few minutes)...\n")

    result = estimate_llc(
        model=model,
        dataloader=dataloader,
        sampling_method='local',  # Use 'hmc' for more accurate but slower results
        num_chains=5,
        num_draws=100,
        device='cpu'
    )
    print_llc_results(result, model_name="Peak Detection")    
    plot_llc_trace(result, model_name="Peak Detection", save_path="peak_trace.png")

if __name__ == "__main__":
    main()