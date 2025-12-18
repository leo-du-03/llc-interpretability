import random
from rasp_models.triplets import get_triplets_model
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


def generate_increasing_sequences(n, min_length=3, max_length=9, vocab=None):
    """Generate increasing sequences (many valid triplets)"""
    if vocab is None:
        vocab = list("abcde")
    
    sequences = []
    for _ in range(n):
        length = min(random.randint(min_length, max_length), len(vocab))
        sequence = sorted(random.sample(vocab, length))
        sequences.append(sequence)
    return sequences


def generate_decreasing_sequences(n, min_length=3, max_length=9, vocab=None):
    """Generate decreasing sequences (no valid increasing triplets)"""
    if vocab is None:
        vocab = list("abcde")
    
    sequences = []
    for _ in range(n):
        length = min(random.randint(min_length, max_length), len(vocab))
        sequence = sorted(random.sample(vocab, length), reverse=True)
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


def generate_alternating_sequences(n, min_length=3, max_length=9, vocab=None):
    """Generate alternating sequences"""
    if vocab is None:
        vocab = list("abcde")
    
    sequences = []
    for _ in range(n):
        length = random.randint(min_length, max_length)
        tokens = random.sample(vocab, min(2, len(vocab)))
        sequence = [tokens[i % len(tokens)] for i in range(length)]
        sequences.append(sequence)
    return sequences


def generate_all_triplets_testcases(vocab=None, max_seq_len=10):
    """
    Generate test cases for triplets algorithm.
    
    Returns:
        List of tuples (input_sequence, expected_output_tensor)
    """
    if vocab is None:
        vocab = list("abcde")
    
    all_cases = []
    bos = "BOS"
    model = get_triplets_model(vocab=set(vocab), max_seq_len=max_seq_len)
    
    # Generate different types of sequences
    random_seqs = generate_random_sequences(20, vocab=vocab)
    increasing_seqs = generate_increasing_sequences(10, vocab=vocab)
    decreasing_seqs = generate_decreasing_sequences(10, vocab=vocab)
    constant_seqs = generate_constant_sequences(5, vocab=vocab)
    alternating_seqs = generate_alternating_sequences(5, vocab=vocab)
    
    all_sequences = random_seqs + increasing_seqs + decreasing_seqs + constant_seqs + alternating_seqs
    
    for seq in all_sequences:
        word = [bos] + seq
        out = model.apply(word)
        all_cases.append((word, torch.tensor(np.array(out.transformer_output), dtype=torch.float64)))
    
    return all_cases


def get_triplets_test_cases(vocab=None, max_seq_len=10):
    """
    Get triplets test cases.
    
    Args:
        vocab: Vocabulary to use. Defaults to ['a', 'b', 'c', 'd', 'e'].
        max_seq_len: Maximum sequence length.
    
    Returns:
        List of (input, output) tuples.
    """
    return generate_all_triplets_testcases(vocab=vocab, max_seq_len=max_seq_len)


# Testing
# if __name__ == "__main__":
#     test_cases = get_triplets_test_cases()
#     print(f"Generated {len(test_cases)} test cases")
#     for i, (inp, out) in enumerate(test_cases[:5]):
#         print(f"Input:  {inp}")
#         print(f"Output: {out}")
#         print()
