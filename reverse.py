from tracr.rasp import rasp
from tracr.compiler import compiling
import random

def reverse_sequence(tokens):
    """
    Reverse the input sequence.
    For input [BOS, a, b, c, d], output should be [d, c, b, a].
    """
    # Create a selector that matches all positions
    all_true_selector = rasp.Select(tokens, tokens, rasp.Comparison.TRUE)
    length = rasp.SelectorWidth(all_true_selector)

    # Calculate the opposite index for reversal
    # For index i, opposite index is (length - 1 - i)
    opp_idx = (length - rasp.indices).named("opp_idx")
    opp_idx = (opp_idx - 1).named("opp_idx-1")

    # Select tokens from opposite positions
    reverse_selector = rasp.Select(rasp.indices, opp_idx, rasp.Comparison.EQ)
    reverse = rasp.Aggregate(reverse_selector, tokens)

    return reverse

def check_reverse():
    bos = "BOS"
    model = compiling.compile_rasp_to_model(
        program=reverse_sequence(rasp.tokens),
        vocab={"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", " "},
        max_seq_len=10,
        compiler_bos=bos,
    )

    return model
