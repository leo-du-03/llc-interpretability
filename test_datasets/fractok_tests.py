import random
from fractok import check_fractok
import numpy as np
import torch

def generate_random_sequences_with_x(n, max_len, vocab_size):
    """Generate random sequences with guaranteed x's sprinkled in"""
    if vocab_size == 'small':
        max_char = ord('e')  # a-e
    elif vocab_size == 'medium':
        max_char = ord('m')  # a-m
    elif vocab_size == 'large':
        max_char = ord('z')  # a-z

    sequences = []
    for _ in range(n):
        length = random.randint(2, min(max_len - 1, 10))  # At least 2 chars
        
        # Decide how many x's (at least 0, at most length)
        num_x = random.randint(0, length)
        
        # Create sequence with num_x x's and (length - num_x) other letters
        seq = ['x'] * num_x
        for _ in range(length - num_x):
            seq.append(chr(random.randint(ord('a'), max_char)))
        
        # Shuffle to distribute x's randomly throughout
        random.shuffle(seq)
        
        sequences.append(''.join(seq))
    return sequences

def generate_random_sequences(n, max_len, vocab_size):
    """Generate random sequences with random letters (may or may not contain 'x')"""
    if vocab_size == 'small':
        max_char = ord('e')  # a-e
    elif vocab_size == 'medium':
        max_char = ord('m')  # a-m
    elif vocab_size == 'large':
        max_char = ord('z')  # a-z

    sequences = []
    for _ in range(n):
        seq = []
        length = random.randint(1, min(max_len - 1, 10))
        
        # Generate random tokens (including 'x' naturally in the range)
        for _ in range(length):
            seq.append(chr(random.randint(ord('a'), max_char)))
        
        sequences.append(''.join(seq))
    return sequences

def generate_all_prev_fraction_tokens_x_testcases(max_seq_len=10, vocab_size='medium'):
    all_cases = []
    bos = "BOS"
    model = check_fractok(max_seq_len=max_seq_len, vocab_size=vocab_size)

    # Generate random sequences (they'll naturally have varying amounts of 'x')
    random_sequences = generate_random_sequences(100, max_seq_len, vocab_size)
    
    for line in random_sequences:
        word = [bos] + list(line)
        out = model.apply(word)
        all_cases.append((word, torch.tensor(np.array(out.transformer_output), dtype=torch.float64)))
    
    return all_cases

def generate_fractok_training_data(n=1000, max_seq_len=10, vocab_size='medium'):
    """Generate training data with ground truth fractok values (not full transformer output)"""
    all_cases = []
    bos = "BOS"
    
    # Use the new function that guarantees x's
    random_sequences = generate_random_sequences_with_x(n, max_seq_len, vocab_size)
    
    for line in random_sequences:
        word = [bos] + list(line)
        
        # Compute ground truth directly
        fractok_values = []
        for i in range(len(word)):
            if i == 0:  # BOS
                fractok_values.append(0.0)
            else:
                num_x = sum(1 for t in word[1:i+1] if t == 'x')
                fractok_values.append(num_x / i)
        
        all_cases.append((word, torch.tensor(fractok_values, dtype=torch.float32)))
    
    return all_cases