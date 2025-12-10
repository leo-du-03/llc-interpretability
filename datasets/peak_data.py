import random
from rasp_models.peak import get_peak_model  
from rasp_models.dominantpeak import get_dompeak_model
import numpy as np
import torch

def generate_peak_sequences(n, min_length=3, max_length=9, min_val=0, max_val=4):
    """Generate sequences with various peak patterns"""
    sequences = []
    for _ in range(n):
        length = random.randint(min_length, max_length)
        sequence = [random.randint(min_val, max_val) for _ in range(length)]
        sequences.append(sequence)
    return sequences

def generate_alternating_peak_sequences(n, min_length=3, max_length=9, min_val=0, max_val=4):
    """Generate sequences that alternate between peaks and valleys"""
    sequences = []
    for _ in range(n):
        length = random.randint(min_length, max_length)
        sequence = []
        for i in range(length):
            if i % 2 == 0:
                sequence.append(random.randint(min_val + 3, max_val))
            else:
                sequence.append(random.randint(min_val, max_val - 3))
        sequences.append(sequence)
    return sequences

def generate_monotonic_sequences(n, min_length=3, max_length=9, min_val=0, max_val=4):
    """Generate increasing or decreasing sequences (no peaks)"""
    sequences = []
    for _ in range(n):
        length = random.randint(min_length, max_length)
        sequence = sorted([random.randint(min_val, max_val) for _ in range(length)])
        if random.random() < 0.5:
            sequence.reverse()  
        sequences.append(sequence)
    return sequences

def generate_constant_sequences(n, min_length=3, max_length=9, min_val=0, max_val=4):
    """Generate sequences with all same values (no peaks)"""
    sequences = []
    for _ in range(n):
        length = random.randint(min_length, max_length)
        value = random.randint(min_val, max_val)
        sequence = [value] * length
        sequences.append(sequence)
    return sequences

def generate_all_peak_testcases(n):
    all_cases = []
    bos = "BOS"
    if n == 0:
        model = get_peak_model()  
    else:
        model = get_dompeak_model()
    
    random_seqs = generate_peak_sequences(20)
    alternating_seqs = generate_alternating_peak_sequences(15)
    monotonic_seqs = generate_monotonic_sequences(10)
    constant_seqs = generate_constant_sequences(5)
    
    all_sequences = random_seqs + alternating_seqs + monotonic_seqs + constant_seqs
    
    for seq in all_sequences:
        word = [bos] + seq
        out = model.apply(word)
        all_cases.append((word, torch.tensor(np.array(out.transformer_output), dtype=torch.float64)))
    
    return all_cases

def get_peak_test_cases(n):
    if n == 0:
        return generate_all_peak_testcases(0) 
    else:
        return generate_all_peak_testcases(1) #domPeak
    


# test_cases = [
#     (["BOS", 1, 2, 3],       ["BOS", False, False, True]),
#     (["BOS", 0, 0, 0],       ["BOS", False, False, False]),
#     (["BOS", 4, 3, 2, 1],    ["BOS", True, False, False, False]),
#     (["BOS", 1, 1, 1, 1],    ["BOS", False, False, False, False]),
#     (["BOS", 9, 9, 9],       ["BOS", False, False, False]),
#     (["BOS", 1, 2, 3, 4, 5], ["BOS", False, False, False, False, True]),
#     (["BOS", 2, 2, 3, 2],    ["BOS", False, False, True, False]),
#     (["BOS", 1, 3, 1, 3, 1], ["BOS", False, True, False, True, False]),
#     (["BOS", 0, 1, 0, 1, 0], ["BOS", False, True, False, True, False]),
#     (["BOS", 2, 4, 2, 4, 2], ["BOS", False, True, False, True, False]),
#     (["BOS", 1, 5, 1, 5, 1], ["BOS", False, True, False, True, False]),
#     (["BOS", 9, 8, 7, 6, 5], ["BOS", True, False, False, False, False]),
#     (["BOS", 1, 2, 1, 2, 1], ["BOS", False, True, False, True, False]),
#     (["BOS", 0, 2, 0, 2, 0], ["BOS", False, True, False, True, False]),
#     (["BOS", 2, 3, 4, 3, 2], ["BOS", False, False, True, False, False]),
#     (["BOS", 1, 4, 1, 4, 1], ["BOS", False, True, False, True, False]),
#     (["BOS", 0, 1, 2, 1, 0], ["BOS", False, False, True, False, False]),
#     (["BOS", 1, 3, 5, 3, 1], ["BOS", False, False, True, False, False]),
#     (["BOS", 2, 2, 2, 2, 2], ["BOS", False, False, False, False, False]),
#     (["BOS", 3, 5, 7, 5, 3], ["BOS", False, False, True, False, False]),

#     (["BOS", 0, 1, 0, 1, 0], ["BOS", False, True, False, True, False]),
#     (["BOS", 2, 4, 2, 4, 2], ["BOS", False, True, False, True, False]),
#     (["BOS", 1, 3, 1, 3, 1], ["BOS", False, True, False, True, False]),
#     (["BOS", 0, 3, 0, 3, 0], ["BOS", False, True, False, True, False]),
#     (["BOS", 1, 4, 1, 4, 1], ["BOS", False, True, False, True, False]),
#     (["BOS", 2, 5, 2, 5, 2], ["BOS", False, True, False, True, False]),
#     (["BOS", 3, 6, 3, 6, 3], ["BOS", False, True, False, True, False]),
#     (["BOS", 4, 7, 4, 7, 4], ["BOS", False, True, False, True, False]),
#     (["BOS", 5, 8, 5, 8, 5], ["BOS", False, True, False, True, False]),
#     (["BOS", 6, 9, 6, 9, 6], ["BOS", False, True, False, True, False]),

#     # failing test cases were fixed! 
#     (["BOS", 3, 2, 3, 2, 3], ["BOS", True, False, True, False, True]),
#     (["BOS", 4, 3, 4, 3, 4], ["BOS", True, False, True, False, True]),
#     (["BOS", 5, 4, 5, 4, 5], ["BOS", True, False, True, False, True]),
#     (["BOS", 6, 5, 6, 5, 6], ["BOS", True, False, True, False, True]),
#     (["BOS", 8, 0, 2],       ["BOS", True, False, True]),
#     (["BOS", 3, 5, 2, 4],    ["BOS", False, True, False, True]),
#     (["BOS", 7, 5, 7, 5],    ["BOS", True, False, True, False]),
#     (["BOS", 3, 1, 3, 1, 3], ["BOS", True, False, True, False, True]),
#     (["BOS", 5, 3, 5, 3, 5], ["BOS", True, False, True, False, True]),
#     (["BOS", 1, 0, 1, 0, 1], ["BOS", True, False, True, False, True]),
#     (["BOS", 5, 1, 5, 1, 5], ["BOS", True, False, True, False, True]),
#     (["BOS", 6, 2, 6, 2, 6], ["BOS", True, False, True, False, True]),
#     (["BOS", 7, 0, 7, 0, 7], ["BOS", True, False, True, False, True]),
#     (["BOS", 8, 1, 8, 1, 8], ["BOS", True, False, True, False, True]),
#     (["BOS", 9, 2, 9, 2, 9], ["BOS", True, False, True, False, True]),
#     (["BOS", 1, 0, 1, 0, 1], ["BOS", True, False, True, False, True]),
#     (["BOS", 2, 1, 2, 1, 2], ["BOS", True, False, True, False, True]),

#     # single numbers or boundary elements lack both neighbors -> ambiguous whether they are peaks
#     # due to the missing right neighbor, we get False instead of True 
#     (["BOS", 0],             ["BOS", False]),
#     (["BOS", 5],             ["BOS", False]),
# ]

