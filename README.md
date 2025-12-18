# LLC Interpretability
This is project aims to investigate the use of the Local Learning Coefficient (LLC) in interpreting Transformer-based models.

## Project Goals
We aim to investigate three things:
- LLC's ability to indicate algorithmic complexity stored within a model
- Using LLC as a metric in model training and validation
- Measuring the effect of ablations on models by measuring LLC before and after the ablation

## User Guide
Navigate to the top level of this repository.
From there, run the following commands:
```
cd tracr
pip install .
cd ..
pip install -r requirements.txt
pip install -e .
```

You should have everything you need now!
Use the below command to run our full test suite. 
```
python tests/run_tests.py
```

You may see a warning like this: `An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.`
This is safe to ignore.

You can now run any python script or jupyter notebook in this repository.

## RASP Algorithms
Here is a description of the algorithms that we implemented in RASP and studied:
- Fraction of Previous Tokens: given a list of characters, returns a float list of the same size with a float at an index corresponding to the fraction of previous tokens in the input sequence that was a particular token.
- Reverse: given a list, returns the reversed list.
- Palindrome Detection: given a list of characters, returns a boolean list of the same size with all “True” if the list represents a palindrome and all “False” otherwise.
- Peak Detection: given a list of integers, returns a boolean list of the same size with “True” if the token in that index is greater than the tokens in immediately adjacent indices.
- Dominant Peak Detection: a variant of peak detection in which an input token is a “dominant peak” if it is at least two greater than the two adjacent tokens on either side and doesn’t immediately rise back up.
- Histogram: given a list of tokens, returns an integer list of the same size where each position contains the frequency of that token in the entire sequence.
- Rank Sort: given a list of tokens, returns the list sorted in ascending order. Each element’s position is determined by counting how many elements are strictly smaller than it.
- Triplets: given a list of tokens, returns an integer list where each position k contains the product of (number of positions before k) × (number of tokens smaller than token at k)

## Repository Structure
### datasets
This folder contains python scripts that generate datasets and dataloaders for the various models we use.
Run `dataloaders.py` to make sure all dataloaders create data and instantiate properly.
Files:
- `dataloaders.py`: Creates dataloaders using the dataset generators in this folder, and helper functions to assist in making dataloaders. Also contains tests for instantiation and data creation.
- `fractok_data.py`: Creates datasets for the fraction of previous tokens algorithm.
- `histogram_data.py`: Creates datasets for the histogram algorithm.
- `palindrome_data.py`: Creates datasets for the palindrome detection algorithm.
- `peak_data.py`: Contains methods to create datasets for the peak and dominant peak detection algorithm. Also has a list of hard coded data for testing.
- `reverse_data.py`: Contains methods to create datasets for the reverse algorithm.
- `sort_data.py`: Contains methods to create datasets for the sort algorithm.
- `triplets_data.py`: Contains methods to create datasets for the triplets algorithm.
### llc_estim
This folder contains jupyter notebooks to estimate LLCs for various compiled transformers.
Files:
- `llc_dev.ipynb`: A test notebook to develop the workflow for estimating LLCs. Used the palindrome algorithm for testing.
- `llc_dompeak.ipynb`: Estimates LLC for dominant peak detection algorithm.
- `llc_fractok.ipynb`: Estimates LLC for the fraction of previous tokens algorithm.
- `llc_histogram.ipynb`: Estimates LLC for the histogram algorithm.
- `llc_peak.ipynb`: Estimates LLC for peak detection algorithm.
- `llc_reverse.ipynb`: Estimates LLC for reverse algorithm.
- `llc_sort.ipynb`: Estimates LLC for sort algorithm.
- `llc_triplets.ipynb`: Estimates LLC for triplets algorithm.
- `main_peak.py`: Python script to estimate LLC for the peak detection algorithm.
- `pal_llc_estim.ipynb`: Estimates LLC for the palindrome detection algorithm.
- `pal_llc_no_rand_head.ipynb`: Estimates LLC for a transformer that adds an extra embedding head to the compiled transformer
### llc_training
Contains notebooks that investigate the use of LLC as a metric in model training.
Files:
- `llc_trained.py`: Python script for training a model on the fraction of previous tokens algorithm and estimating LLC during training
- `palindrome_llc_training.ipynb`: Notebook for training a model with the same structure as a compiled transformer of equivalent function on the palindrome detection algorithm and estimating LLC during training.
- `peak_llc_training.ipynb`: Notebook for training a model with the same structure as a compiled transformer of equivalent function on the peak detection algorithm and estimating LLC during training.
- `train_fractok.py`: Python script for training a model on the fraction of previous tokens algorithm and estimating LLC during training
- `training_utils.py`: Contains useful functions for training, LLC estimation, and performance analysis
#### llc_training/grokking
Contains notebooks that use the flow from Timaeus' grokking demonstration notebook. These notebooks aim to both stabilize LLC during training and investigate if LLC can predict future validation loss.
Files:
- `grokking_fractok.ipynb`: Trains a model on the fraction of previous tokens algorithm.
- `mod_add.ipynb`: Directly adapted from Timaeus' grokking demonstration notebook. Trains a model on modular addition.
- `peak_models.py`: Contains a variety of pytorch models for use in training. Instantiation is tested in `tests/test_pytorch_models.py`.
- `peak.ipynb`: Trains a model on the peak detection algorithm. Pulls models from `peak_models.py`
#### llc_training/models
Folder for temporarily storing trained models.
### rasp_models
Contains python scripts to define and instantiate compiled transformers using the RASP language.
Files:
- `dominantpeak.py`:  Python script to define and initialize a compiled transformer for the dominant peak detection algorithm (which builds off of the peak detection algorithm).
- `fractok.py`: Python script to define and initialize a compiled transformer for the fraction of previous tokens algorithm. Tests are contained in `tests/test_fractok.py`.
- `histogram.py`: Python script to define and initialize a compiled transformer for the histogram algorithm.
- `palindrome.py`: Python script to define and initialize a compiled transformer for the palindrome detection algorithm. Tests are contained in `tests/test_palindrome.py`.
- `peak.py`: Python script to define and initialize a compiled transformer for the peak detection algorithm. Tests are contained in `tests/test_peak.py`.
- `reverse.py`: Python script to define and initialize a compiled transformer for the reverse algorithm.
- `sort.py`: Python script to define and initialize a compiled transformer for the sort algorithm.
- `triplets.py`: Python script to define and initialize a compiled transformer for the triplets algorithm.
### results
Folder to store various results of interest for presentations.
More results can be found in [this google sheet](https://docs.google.com/spreadsheets/d/1cm1KNR_QvzNsIX6buWDczwk_XJ4tRUn-BvJ0xuCGMNE/edit?usp=sharing).
### tests
Contains unit tests for model loading and model accuracy.
Models created using tracr should maintain a 100% accuracy on tests.
- `non_palindromes.txt`: Contains hard-coded test cases for the palindrome model
- `palindromes.txt`: Contains hard-coded test cases for the palindrome model 
- `run_tests.py`: Runs all tests
- `test_fractok.py`: Contains tests for the fraction of previous tokens model.
- `test_haiku_to_pytorch.py`: Contains tests for the method that converts haiku models to pytorch.
- `test_palindrome.py`: Contains tests for the palindrome model.
- `test_peak.py`: Contains tests for the peak model.
- `test_pytorch_models.py`: Contains tests for pytorch models used in `llc_training/grokking`.
### tracr
A modified version of the original tracr codebase.
Code has been added to convert compiled transformers from JAX to PyTorch. This can be found in `tracr/tracr/haiku_to_pytorch.py`.
The conversion is tested in `tests/test_haiku_to_pytorch.py`.
## ablations.ipynb
Contains pipeline to systematically ablate layers from the model and study their affects on LLC.