from tracr.rasp import rasp
from tracr.compiler import compiling
import random

def is_palindrome(tokens):
    all_true_selector = rasp.Select(tokens, tokens, rasp.Comparison.TRUE)
    length = rasp.SelectorWidth(all_true_selector)

    opp_idx = (length - rasp.indices).named("opp_idx")
    opp_idx = (opp_idx - 1).named("opp_idx-1")
    reverse_selector = rasp.Select(rasp.indices, opp_idx, rasp.Comparison.EQ)
    reverse = rasp.Aggregate(reverse_selector, tokens)

    check = rasp.SequenceMap(lambda x, y: x == y, tokens, reverse)

    return check

def check_palindrome():
    bos = "BOS"
    model = compiling.compile_rasp_to_model(
        program=is_palindrome(rasp.tokens),
        vocab={"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", " "},
        max_seq_len=50,
        compiler_bos=bos,
    )

    return model
