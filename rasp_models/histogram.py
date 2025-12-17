from tracr.rasp import rasp
from tracr.compiler import compiling


def histogram():
    """
    Histogram algorithm: for each position i, count how many tokens in the 
    sequence are equal to the token at position i.
    
    Returns a sequence where each position contains the frequency of that token
    in the entire sequence.
    """
    tokens = rasp.tokens  # S-Op: sequence of input tokens

    # Select matrix: sel[i, j] = True iff tokens[j] == tokens[i]
    same_token = rasp.Select(tokens, tokens, rasp.Comparison.EQ)

    # SelectorWidth counts how many True's in each row i
    # This returns a sequence of ints, one per position.
    freq_at_i = rasp.SelectorWidth(same_token)

    return freq_at_i


def get_histogram_model(vocab=None, max_seq_len=10):
    """
    Compile the histogram RASP program to a model.
    
    Args:
        vocab: Set of vocabulary tokens. Defaults to lowercase letters.
        max_seq_len: Maximum sequence length. Defaults to 10.
    
    Returns:
        Compiled tracr model.
    """
    if vocab is None:
        vocab = {"a", "b", "c", "d", "e"}
    
    bos = "BOS"
    model = compiling.compile_rasp_to_model(
        histogram(),
        vocab=vocab,
        max_seq_len=max_seq_len,
        compiler_bos=bos,
    )
    return model


# Testing this method locally:
# if __name__ == "__main__":
#     model = get_histogram_model()
    
#     test_sequences = [
#         ["BOS", "a", "b", "c", "a", "b"],     # a:2, b:2, c:1
#         ["BOS", "a", "a", "a"],               # a:3
#         ["BOS", "a", "b", "c", "d"],          # all 1's
#         ["BOS", "a", "a", "b", "b", "a"],     # a:3, b:2
#         ["BOS", "a"],                         # a:1
#     ]
#     print("Testing Histogram:\n")
#     for seq in test_sequences:
#         result = model.apply(seq)
#         print(f"Input:  {seq}")
#         print(f"Output: {result.decoded}")
#         print()
