from tracr.rasp import rasp
from tracr.compiler import compiling
import random

def detect_repeated_tokens(tokens):
    """
    Detect if each token appears more than once in the sequence.
    For each position, output True if that token appears elsewhere, False otherwise.

    For example:
    - Input: [BOS, a, b, a, c] -> Output: [False, True, False, True, False]
    - Input: [BOS, a, b, c, d] -> Output: [False, False, False, False, False]
    - Input: [BOS, a, a, a, b] -> Output: [False, True, True, True, False]
    """
    # Create a selector that selects positions with the same token
    same_token_selector = rasp.Select(tokens, tokens, rasp.Comparison.EQ)

    # Count how many positions have the same token (including itself)
    token_count = rasp.SelectorWidth(same_token_selector)

    # Create a constant sequence of 1s to compare against
    ones = rasp.Map(lambda x: 1, rasp.indices).named("ones")

    # If count > 1, the token is repeated
    is_repeated = rasp.SequenceMap(lambda count, one: count > one, token_count, ones).named("is_repeated")

    return is_repeated

def check_repeated_token():
    bos = "BOS"
    model = compiling.compile_rasp_to_model(
        program=detect_repeated_tokens(rasp.tokens),
        vocab={"a", "b", "c", "d", "e"},
        max_seq_len=10,
        compiler_bos=bos,
    )

    return model
