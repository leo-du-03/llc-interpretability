from tracr.rasp import rasp
from tracr.compiler import compiling


def sort():
    """
    Sort algorithm: for each position, output the token that should be at that
    position in the sorted sequence.
    
    Uses a simple (non-stable) sort based on counting elements less than current.
    
    Returns a sequence where position k contains the k-th smallest token.
    """
    # Input sequence
    x = rasp.tokens
    i = rasp.indices  # 0..n-1

    # Count how many elements are strictly less than x[i]
    # sel_lt[i, j] = True if x[j] < x[i]
    sel_lt = rasp.Select(x, x, rasp.Comparison.LT)
    
    # rank[i] = number of elements less than x[i]
    rank = rasp.SelectorWidth(sel_lt)

    # Build sorted output: output[k] = x[i] where rank[i] == k
    sel_rank_eq_k = rasp.Select(rank, i, rasp.Comparison.EQ)  # (rank[i] == k)
    sorted_x = rasp.Aggregate(sel_rank_eq_k, x)

    return sorted_x


def get_sort_model(vocab=None, max_seq_len=10):
    """
    Compile the sort RASP program to a model.
    
    Args:
        vocab: Set of vocabulary tokens (should be comparable).
               Defaults to {"a", "b", "c", "d", "e"}.
        max_seq_len: Maximum sequence length. Defaults to 10.
    
    Returns:
        Compiled tracr model.
    """
    if vocab is None:
        vocab = {"a", "b", "c", "d", "e"}
    
    bos = "BOS"
    model = compiling.compile_rasp_to_model(
        sort(),
        vocab=vocab,
        max_seq_len=max_seq_len,
        compiler_bos=bos,
    )
    return model


# Testing this method locally:
# if __name__ == "__main__":
#     model = get_sort_model()
    
#     test_sequences = [
#         ["BOS", 3, 1, 4, 1, 5],     # -> [1, 1, 3, 4, 5]
#         ["BOS", 4, 3, 2, 1, 0],     # -> [0, 1, 2, 3, 4]
#         ["BOS", 0, 1, 2, 3, 4],     # -> [0, 1, 2, 3, 4] (already sorted)
#         ["BOS", 2, 2, 2],           # -> [2, 2, 2] (all same)
#         ["BOS", 1, 0],              # -> [0, 1]
#     ]
#     print("Testing Sort:\n")
#     for seq in test_sequences:
#         result = model.apply(seq)
#         print(f"Input:  {seq}")
#         print(f"Output: {result.decoded}")
#         print()
