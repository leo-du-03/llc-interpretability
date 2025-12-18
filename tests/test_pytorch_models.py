"""
Tests the loading of models declared solely in pytorch. Makes sure they can accept proper input.
"""
from llc_training.grokking.peak_models import Small, Medium_Large, Nano
import torch

def testLoadingGrokkingPeakModels():
    '''
    Makes sure the custom peak models used to test training 
    using the Timaeus Grokking notebook can load correctly.
    '''
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    class ExperimentParams:
        p: int = 100
        l: int = 5
        v = 12
        n_batches: int = 1000
        n_save_model_checkpoints: int = 100
        print_times: int = 100
        lr: float = 3e-3
        batch_size: int = 128
        hidden_size: int = 48
        linear_hidden_size: int = 200
        embed_dim: int = 127
        train_frac: float = 0.4
        random_seed: int = 0
        device: str = DEVICE
        weight_decay: float = 2e-5
        blocks: int = 6
    params = ExperimentParams()
    torch.manual_seed(params.random_seed)
    nano = Nano(params).to(params.device)
    small = Small(params).to(params.device)
    medium_large = Medium_Large(params).to(params.device)

    testList = torch.tensor([[0, 1, 1, 0, 0]])
    assert(nano(testList) != None)
    print(".", end="", flush=True)
    assert(small(testList) != None)
    print(".", end="", flush=True)
    assert(medium_large(testList) != None)
    print(".", end="", flush=True)