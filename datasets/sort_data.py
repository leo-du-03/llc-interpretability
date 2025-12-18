import random
from rasp_models.sort import get_sort_model
import numpy as np
import torch


def generate_random_sequences(n, min_length=3, max_length=9, vocab=None):
    """Generate random sequences from vocabulary"""
    if vocab is None:
        vocab = list("abcde")
    
    sequences = []
    for _ in range(n):
        length = random.randint(min_length, max_length)
        sequence = [random.choice(vocab) for _ in range(length)]
        sequences.append(sequence)
    return sequences


def generate_sorted_sequences(n, min_length=3, max_length=9, vocab=None):
    """Generate already sorted sequences"""
    if vocab is None:
        vocab = list("abcde")
    
    sequences = []
    for _ in range(n):
        length = random.randint(min_length, max_length)
        sequence = sorted([random.choice(vocab) for _ in range(length)])
        sequences.append(sequence)
    return sequences


def generate_reverse_sorted_sequences(n, min_length=3, max_length=9, vocab=None):
    """Generate reverse sorted sequences"""
    if vocab is None:
        vocab = list("abcde")
    
    sequences = []
    for _ in range(n):
        length = random.randint(min_length, max_length)
        sequence = sorted([random.choice(vocab) for _ in range(length)], reverse=True)
        sequences.append(sequence)
    return sequences


def generate_constant_sequences(n, min_length=3, max_length=9, vocab=None):
    """Generate sequences with all same tokens"""
    if vocab is None:
        vocab = list("abcde")
    
    sequences = []
    for _ in range(n):
        length = random.randint(min_length, max_length)
        token = random.choice(vocab)
        sequence = [token] * length
        sequences.append(sequence)
    return sequences


def generate_all_sort_testcases(vocab=None, max_seq_len=10):
    """
    Generate test cases for sort algorithm.
    
    Returns:
        List of tuples (input_sequence, expected_output_tensor)
    """
    if vocab is None:
        vocab = list("abcde")
    
    all_cases = []
    bos = "BOS"
    model = get_sort_model(vocab=set(vocab), max_seq_len=max_seq_len)
    
    # Generate different types of sequences
    random_seqs = generate_random_sequences(20, vocab=vocab)
    sorted_seqs = generate_sorted_sequences(10, vocab=vocab)
    reverse_seqs = generate_reverse_sorted_sequences(10, vocab=vocab)
    constant_seqs = generate_constant_sequences(10, vocab=vocab)
    
    all_sequences = random_seqs + sorted_seqs + reverse_seqs + constant_seqs
    
    for seq in all_sequences:
        word = [bos] + seq
        out = model.apply(word)
        all_cases.append((word, torch.tensor(np.array(out.transformer_output), dtype=torch.float64)))
    
    return all_cases


def get_sort_test_cases(vocab=None, max_seq_len=10):
    """
    Get sort test cases.
    
    Args:
        vocab: Vocabulary to use. Defaults to [0, 1, 2, 3, 4].
        max_seq_len: Maximum sequence length.
    
    Returns:
        List of (input, output) tuples.
    """
    return generate_all_sort_testcases(vocab=vocab, max_seq_len=max_seq_len)


def compute_sort_output(seq):
    """
    Compute the sorted output for a sequence.
    
    Args:
        seq: List of tokens (without BOS).
    
    Returns:
        Sorted list.
    """
    return sorted(seq)


# Testing
# if __name__ == "__main__":
#     test_cases = get_sort_test_cases()
#     print(f"Generated {len(test_cases)} test cases")
#     for i, (inp, out) in enumerate(test_cases[:5]):
#         print(f"Input:  {inp}")
#         print(f"Output: {out}")
#         print()
