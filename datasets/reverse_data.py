import random
from rasp_models.reverse import get_reverse_model
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


def generate_palindrome_sequences(n, min_length=3, max_length=9, vocab=None):
    """Generate palindrome sequences (reverse equals original)"""
    if vocab is None:
        vocab = list("abcde")
    
    sequences = []
    for _ in range(n):
        half_length = random.randint(min_length // 2, max_length // 2)
        half = [random.choice(vocab) for _ in range(half_length)]
        # Create palindrome
        sequence = half + half[::-1]
        sequences.append(sequence)
    return sequences


def generate_repeated_sequences(n, min_length=3, max_length=9, vocab=None):
    """Generate sequences with repeated tokens"""
    if vocab is None:
        vocab = list("abcde")
    
    sequences = []
    for _ in range(n):
        length = random.randint(min_length, max_length)
        # Use a smaller subset of vocab to ensure repetitions
        subset = random.sample(vocab, min(2, len(vocab)))
        sequence = [random.choice(subset) for _ in range(length)]
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


def generate_all_reverse_testcases(vocab=None, max_seq_len=10):
    """
    Generate test cases for reverse algorithm.
    
    Returns:
        List of tuples (input_sequence, expected_output_tensor)
    """
    if vocab is None:
        vocab = list("abcde")
    
    all_cases = []
    bos = "BOS"
    model = get_reverse_model(vocab=set(vocab), max_seq_len=max_seq_len)
    
    # Generate different types of sequences
    random_seqs = generate_random_sequences(20, vocab=vocab)
    palindrome_seqs = generate_palindrome_sequences(10, vocab=vocab)
    repeated_seqs = generate_repeated_sequences(10, vocab=vocab)
    constant_seqs = generate_constant_sequences(10, vocab=vocab)
    
    all_sequences = random_seqs + palindrome_seqs + repeated_seqs + constant_seqs
    
    for seq in all_sequences:
        word = [bos] + seq
        out = model.apply(word)
        all_cases.append((word, torch.tensor(np.array(out.transformer_output), dtype=torch.float64)))
    
    return all_cases


def get_reverse_test_cases(vocab=None, max_seq_len=10):
    """
    Get reverse test cases.
    
    Args:
        vocab: Vocabulary to use. Defaults to ['a', 'b', 'c', 'd', 'e'].
        max_seq_len: Maximum sequence length.
    
    Returns:
        List of (input, output) tuples.
    """
    return generate_all_reverse_testcases(vocab=vocab, max_seq_len=max_seq_len)


def compute_reverse_output(seq):
    """
    Compute the reversed output for a sequence.
    
    Args:
        seq: List of tokens (without BOS).
    
    Returns:
        Reversed list.
    """
    return seq[::-1]


# Testing
# if __name__ == "__main__":
#     test_cases = get_reverse_test_cases()
#     print(f"Generated {len(test_cases)} test cases")
#     for i, (inp, out) in enumerate(test_cases[:5]):
#         print(f"Input:  {inp}")
#         print(f"Output: {out}")
#         print()
