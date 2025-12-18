from tracr.rasp import rasp
from tracr.compiler import compiling


def reverse():
    """
    Reverse algorithm: reverse the sequence.
    
    For each position i, output the token at position (n-1-i) where n is the
    sequence length.
    
    Returns the reversed sequence.
    """
    tokens = rasp.tokens
    indices = rasp.indices

    # Get sequence length by counting all positions
    all_true_selector = rasp.Select(tokens, tokens, rasp.Comparison.TRUE)
    length = rasp.SelectorWidth(all_true_selector)

    # Compute the opposite index: opp_idx[i] = length - 1 - i
    opp_idx = length - indices - 1

    # Select the token at the opposite index
    reverse_selector = rasp.Select(indices, opp_idx, rasp.Comparison.EQ)
    reversed_tokens = rasp.Aggregate(reverse_selector, tokens)

    return reversed_tokens


def get_reverse_model(vocab=None, max_seq_len=10):
    """
    Compile the reverse RASP program to a model.
    
    Args:
        vocab: Set of vocabulary tokens.
               Defaults to {"a", "b", "c", "d", "e"}.
        max_seq_len: Maximum sequence length. Defaults to 10.
    
    Returns:
        Compiled tracr model.
    """
    if vocab is None:
        vocab = {"a", "b", "c", "d", "e"}
    
    bos = "BOS"
    model = compiling.compile_rasp_to_model(
        reverse(),
        vocab=vocab,
        max_seq_len=max_seq_len,
        compiler_bos=bos,
    )
    return model


# Testing this method locally:
# if __name__ == "__main__":
#     model = get_reverse_model()
    
#     test_sequences = [
#         ["BOS", "a", "b", "c", "d", "e"],     # -> [e, d, c, b, a]
#         ["BOS", "a", "b", "c"],               # -> [c, b, a]
#         ["BOS", "a", "a", "b", "b"],          # -> [b, b, a, a]
#         ["BOS", "a"],                         # -> [a]
#         ["BOS", "e", "d", "c", "b", "a"],     # -> [a, b, c, d, e]
#     ]
#     print("Testing Reverse:\n")
#     for seq in test_sequences:
#         result = model.apply(seq)
#         print(f"Input:  {seq}")
#         print(f"Output: {result.decoded}")
#         print()
