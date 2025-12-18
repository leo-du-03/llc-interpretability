from tracr.rasp import rasp
from tracr.compiler import compiling
import random

"""
This file contains the code that defines and initializes a RASP model that solves the palindrome detection task.
is_palindrome: defines the model
check_palindrome: initializes the model
"""

def is_palindrome(tokens):
    all_true_selector = rasp.Select(tokens, tokens, rasp.Comparison.TRUE)
    length = rasp.SelectorWidth(all_true_selector)

    # Calculate the reverse of tokens
    opp_idx = (length - rasp.indices).named("opp_idx")
    opp_idx = (opp_idx - 1).named("opp_idx-1")
    reverse_selector = rasp.Select(rasp.indices, opp_idx, rasp.Comparison.EQ)
    reverse = rasp.Aggregate(reverse_selector, tokens)

    # Compute element-wise equality between tokens and reverse
    check = rasp.SequenceMap(lambda x, y: x == y, tokens, reverse)

    return check

def check_palindrome(vocab={"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", " "}):
    bos = "BOS"
    model = compiling.compile_rasp_to_model(
        program=is_palindrome(rasp.tokens),
        vocab=vocab,
        max_seq_len=10,
        compiler_bos=bos,
    )

    return model
