# LLC Interpretability
This is project aims to investigate the use of the Local Learning Coefficient (LLC) in interpreting Transformer-based models.

## Project Goals
We aim to investigate three things:
- LLC's ability to indicate algorithmic complexity stored within a model
- Using LLC as a metric in model training and validation
- Measuring the effect of ablations on models by measuring LLC before and after the ablation

## Repository Structure
### datasets
This folder contains python scripts that generate datasets and dataloaders for the various models we use.
Files:
- `dataloaders.py`: Creates dataloaders using the dataset generators in this folder, and helper functions to assist in making dataloaders.
- `fractok_data.py`: Create datasets for the previous fractions is token problem.
- `palindrome_data.py`: Create datasets for the palindrome detection problem.
- `peak_data.py`: Contains list of hard coded data for the peak detection problem.
### llc_estim
This folder contains jupyter notebooks to estimate LLCs for various compiled transformers.
Files:
- `haiku_to_pytorch_rg.py`: Contains useful functions for converting haiku models to pytorch.
- `llc_dev.ipynb`: A test notebook to develop the workflow for estimating LLCs. Used the palindrome compiled transformer.
- `llc_fractok.ipynb`: Estimates LLC for the previous fraction is token problem.
- `main_peak.py`: Python script to estimate LLC for the peak detection problem.
- `pal_llc_estim.ipynb`: Estimates LLC for the palindrome detection problem.
- `pal_llc_no_rand_head.ipynb`: Estimates LLC for a transformer that adds an extra embedding head to the compiled transformer 
- `peak_llc_estim.ipynb`: Estimates LLC for the peak detection problem.
- `rg_new_llc.ipynb`: A development notebook using the palindrome compiled transformer.
### llc_training
Contains notebooks that investigate the use of LLC as a metric in model training.
Files:
- `llc_trained.py`: Python script for training a model on the previous fraction is token problem and estimating LLC during training
- `palindrome_llc_training.ipynb`: Notebook for training a model with the same structure as a compiled transformer of equivalent function on the palindrome detection problem and estimating LLC during training.
- `peak_llc_training.ipynb`: Notebook for training a model with the same structure as a compiled transformer of equivalent function on the peak detection problem and estimating LLC during training.
- `train_fractok.py`: Python script for training a model on the previous fraction is token problem and estimating LLC during training
- `trainedmodels.py`: Contains useful functions for training, LLC estimation, and performance analysis
- `trainedmodelsdemo.ipynb`: Development notebook for training workflow. Uses the palindrome detection model.
#### grokking
Contains notebooks that use the flow from from Timaeus' grokking demonstration notebook. These notebooks aim to both stabilize LLC during training and investigate if LLC can predict future validation loss.
Files:
- `mod_add.ipynb`: Directly adapted from Timaeus' grokking demonstration notebook. Trains a model on modular addition.
- `peak_big.ipynb`: Trains a model on the peak detection problem. Uses a large sequential transformer-based architecture with >100,000 parameters.
- `peak_small.ipynb`: Trains a model on the peak detection problem. Uses a small parallel transformer-based architecture.
#### models
Folder to store trained models from the various training notebooks
### rasp_models
Contains python scripts to define and instantiate compiled transformers using the RASP language.
Files:
- `fractok.py`: Python script to define and initialize a compiled transformer for the previous fraction is token problem.
- `palindrome.py`: Python script to define and initialize a compiled transformer for the palindrome detection problem.
- `peak.py`: Python script to define and initialize a compiled transformer for the peak detection problem.
### results
Folder to store various results of interest for future presentations.
### tests
Contains unit tests for model loading and model accuracy.
Models created using tracr should maintain a 100% accuracy on tests.
### tracr
A modified version of the original tracr codebase.
Code has been added to convert compiled transformers from JAX to PyTorch. This can be found in `tracr/tracr/haiku_to_pytorch.py`.