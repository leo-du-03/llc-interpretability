from tracr.rasp import rasp
from tracr.compiler import compiling
"""
   This algorithm accepts an array of integers and returns an array of booleans that indicate whether each value is a peak (local maximum).
"""
def peak():
    # Getting indicies of L,R neighbors
    left_idx = rasp.indices - 1
    right_idx = rasp.indices + 1

    # Select neighbors by matching indicies 
    left_selector = rasp.Select(rasp.indices, left_idx, rasp.Comparison.EQ)
    right_selector = rasp.Select(rasp.indices, right_idx, rasp.Comparison.EQ)

    # Aggregating L, R neighbor values 
    left_vals = rasp.Aggregate(left_selector, rasp.tokens).named("left_vals")
    right_vals = rasp.Aggregate(right_selector, rasp.tokens).named("right_vals")
    
    # Checking if neighbors exist 
    has_left = rasp.SelectorWidth(left_selector).named("has_left")
    has_right = rasp.SelectorWidth(right_selector).named("has_right")

    # Replacing BOS with -1 to help with comparisons
    tokens_clean = rasp.Map(lambda x: -1 if x == "BOS" else x, rasp.tokens)
    left_clean = rasp.Map(lambda l: -1 if l == "BOS" else l, left_vals)
    right_clean = rasp.Map(lambda r: -1 if r == "BOS" else r, right_vals)

    # Comparing token to neighbors
    cmp_left = rasp.SequenceMap(lambda x, l: x > l, tokens_clean, left_clean)
    cmp_right = rasp.SequenceMap(lambda x, r: x > r, tokens_clean, right_clean)
    
    # Handling missing neighbors
    gt_left = rasp.SequenceMap(lambda cmp, hl: (hl == 0) or cmp, cmp_left, has_left)
    gt_right = rasp.SequenceMap(lambda cmp, hr: (hr == 0) or cmp, cmp_right, has_right)

    # Peak condition: greater than both neighbors
    peaks_cond = rasp.SequenceMap(lambda a, b: a and b, gt_left, gt_right)

    return peaks_cond


def get_peak_model():
    bos = "BOS"
    model = compiling.compile_rasp_to_model(
        peak(),
        vocab={0, 1, 2, 3, 4}, # changed when testing different vocab size for LLC (0-4, 0-6, 0-9)
        max_seq_len=50, # changed when testing different sequence lengths for LLC (10, 50)
        compiler_bos=bos,
    )
    return model

# # Testing this method locally: 
# if __name__ == "__main__":
#     model = get_peak_model()
    
#     test_sequences = [
#         ["BOS", 0, 1, 2, 1, 0],      
#         ["BOS", 0, 1, 4, 1, 0],      
#         ["BOS", 1, 0, 3, 0, 1],      
#         ["BOS", 0, 2, 4, 2, 0],     
#         ["BOS", 1, 2, 3, 2, 1],    
#         ["BOS", 0],    
#         ["BOS", 2, 4, 2],        
#         ["BOS", 2, 0, 4, 0, 2],      
#         ["BOS", 2, 0, 0, 4, 0, 0, 2],      
#         ["BOS", 1, 0, 0, 2, 0, 0, 0],    
#         ["BOS", 2, 0, 2, 4, 2, 0, 2],
#         ["BOS", 2, 0, 2, 4, 2, 0, 2]
#     ]
#     print("Testing Peak Detection:\n")
#     for seq in test_sequences:
#         result = model.apply(seq)
#         print(f"Input:  {seq}")
#         print(f"Output: {result.decoded}")
#         print()