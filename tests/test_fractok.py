"""
Contains tests for the fraction of previous tokens model.
Ensures that the model is accurate.
"""
import numpy as np
from rasp_models.fractok import check_fractok
from datasets.fractok_data import generate_fractok_training_data

def fractokTests():
    '''
    Makes sure the fractok model is outputing the correct answers
    '''
    model = check_fractok()

    test_cases = generate_fractok_training_data()
    for i, (input_seq, target) in enumerate(test_cases, 1):
        out = model.apply(input_seq)
        decoded = out.decoded
        decoded = [0] + decoded[1:]
        if np.allclose(decoded, target.tolist(), 1e-06): # An extremely minor floating point imprecision can occur here
            print(".", end="", flush=True)
        else:
            print(
                f"\n--- Decoded Output Mismatch in Test Case {i} ---\n"
                f"Input:    {input_seq}\n"
                f"Expected: {target}\n"
                f"Got:      {decoded}\n"
            )
    print("\n")

if __name__ == "__main__":
    fractokTests()
