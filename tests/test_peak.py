"""
Contains tests for the peak detection model.
Ensures that the model is accurate.
"""
from rasp_models.peak import get_peak_model
from datasets.peak_data import get_hardcoded_peak_test_cases

def peakTests():
    '''
    Makes sure the peak model is outputing the correct answers
    '''
    model = get_peak_model()

    test_cases = get_hardcoded_peak_test_cases()
    for i, (input_seq, expected_output) in enumerate(test_cases, 1):
        out = model.apply(input_seq)
        decoded = out.decoded
        if decoded == expected_output: 
            print(".", end="", flush=True)
        else:
            print(
                f"\n--- Decoded Output Mismatch in Test Case {i} ---\n"
                f"Input:    {input_seq}\n"
                f"Expected: {expected_output}\n"
                f"Got:      {decoded}\n"
            )