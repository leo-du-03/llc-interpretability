from tracr.rasp import rasp
from tracr.compiler import compiling
"""
    This algorithm builds on peak (where any number simply must be bigger than the numbers)
    Here, peaks must have valleys on both sides:
    1. Must be a local maximum (as it was prior)
    2. Must have a drop of at least 2 on both sides before rising again
    3. No immediate neighbors can be equal (strict inequality)
"""

def dompeak():
    # 1. This segment of code is all that is code present in peak.py: 
    left_idx = rasp.indices - 1
    right_idx = rasp.indices + 1
    
    left_selector = rasp.Select(rasp.indices, left_idx, rasp.Comparison.EQ)
    right_selector = rasp.Select(rasp.indices, right_idx, rasp.Comparison.EQ)
    left_vals = rasp.Aggregate(left_selector, rasp.tokens).named("left_vals")
    right_vals = rasp.Aggregate(right_selector, rasp.tokens).named("right_vals")
    
    has_left = rasp.SelectorWidth(left_selector)
    has_right = rasp.SelectorWidth(right_selector)
    
    tokens_clean = rasp.Map(lambda x: -1 if x == "BOS" else x, rasp.tokens)
    left_clean = rasp.Map(lambda l: -1 if l == "BOS" else l, left_vals)
    right_clean = rasp.Map(lambda r: -1 if r == "BOS" else r, right_vals)
    
    cmp_left = rasp.SequenceMap(lambda x, l: x > l, tokens_clean, left_clean)
    cmp_right = rasp.SequenceMap(lambda x, r: x > r, tokens_clean, right_clean)
    gt_left = rasp.SequenceMap(lambda cmp, hl: (hl == 0) or cmp, cmp_left, has_left)
    gt_right = rasp.SequenceMap(lambda cmp, hr: (hr == 0) or cmp, cmp_right, has_right)
    
    is_peak = rasp.SequenceMap(lambda a, b: a and b, gt_left, gt_right)

    # Get radius-2 neighbors to find valleys
    left2_idx = rasp.indices - 2
    right2_idx = rasp.indices + 2
    
    left2_selector = rasp.Select(rasp.indices, left2_idx, rasp.Comparison.EQ)
    right2_selector = rasp.Select(rasp.indices, right2_idx, rasp.Comparison.EQ)
    left2_vals = rasp.Aggregate(left2_selector, rasp.tokens)
    right2_vals = rasp.Aggregate(right2_selector, rasp.tokens)
    
    has_left2 = rasp.SelectorWidth(left2_selector)
    has_right2 = rasp.SelectorWidth(right2_selector)
    
    left2_clean = rasp.Map(lambda l: -1 if l == "BOS" else l, left2_vals)
    right2_clean = rasp.Map(lambda r: -1 if r == "BOS" else r, right2_vals)
    
    # Valley check: position-1 should be at least 2 less than current
    # AND position-2 should be less than or equal to position-1
    # This ensures a valley exists (drop then stays low or drops more)
    
    # Calculate drop amount
    left_drop_amt = rasp.SequenceMap(lambda x, l: x - l, tokens_clean, left_clean)
    right_drop_amt = rasp.SequenceMap(lambda x, r: x - r, tokens_clean, right_clean)
    
    # 2. Check if drop is sufficient (>= 2) or no neighbor exists
    left_drop = rasp.SequenceMap(
        lambda drop, hl: (hl == 0) or (drop >= 2),
        left_drop_amt, has_left
    ).named("left_drop")
    
    right_drop = rasp.SequenceMap(
        lambda drop, hr: (hr == 0) or (drop >= 2),
        right_drop_amt, has_right
    ).named("right_drop")
    
    # 3. Check that the valley continues (doesn't immediately rise)
    left_valley_check = rasp.SequenceMap(lambda l2, l1: l2 <= l1, left2_clean, left_clean)
    right_valley_check = rasp.SequenceMap(lambda r2, r1: r2 <= r1, right2_clean, right_clean)
    
    left_valley = rasp.SequenceMap(
        lambda check, hl2: (hl2 == 0) or check,
        left_valley_check, has_left2
    ).named("left_valley")
    
    right_valley = rasp.SequenceMap(
        lambda check, hr2: (hr2 == 0) or check,
        right_valley_check, has_right2
    ).named("right_valley")
    
    # Combine valley conditions
    left_has_valley = rasp.SequenceMap(
        lambda drop, valley: drop and valley,
        left_drop, left_valley
    )
    
    right_has_valley = rasp.SequenceMap(
        lambda drop, valley: drop and valley,
        right_drop, right_valley
    )
    
    # Final: peak with valleys on both sides
    # Chain the three conditions in pairs
    peak_and_left = rasp.SequenceMap(
        lambda peak, lv: peak and lv,
        is_peak, left_has_valley
    )
    
    valley_aware_peak = rasp.SequenceMap(
        lambda pl, rv: pl and rv,
        peak_and_left, right_has_valley
    ).named("valley_aware_peak")
    
    return valley_aware_peak


def get_dompeak_model():
    bos = "BOS"
    model = compiling.compile_rasp_to_model(
        dompeak(),
        vocab={0, 1, 2, 3, 4},
        max_seq_len=10,
        compiler_bos=bos,
    )
    return model

# # testing this method locally: 
# if __name__ == "__main__":
#     model = get_dompeak_model()
    
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
#         ["BOS", 2, 0, 2, 4, 2, 0, 2]  
#     ]
#     print("Testing Valley-Aware Peak Detection:\n")
#     for seq in test_sequences:
#         result = model.apply(seq)
#         print(f"Input:  {seq}")
#         print(f"Output: {result.decoded}")
#         print()