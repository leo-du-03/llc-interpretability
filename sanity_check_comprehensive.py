#!/usr/bin/env python3
"""
Comprehensive sanity check script for TRACR compiled transformers.

This script performs thorough testing of all three models with multiple test cases
to ensure they work correctly before running LLC analysis.
"""

# Configuration: Set which algorithms to test
# Options: "repeated_token", "palindrome", "reverse", or "all"
ALGORITHMS_TO_TEST = "all"  # Change this to test specific algorithms

import torch
import numpy as np
from torchinfo import summary

from tracr.tracr.haiku_to_pytorch import haiku_to_pytorch, apply, outputs_equal

# Import model functions
from palindrome import check_palindrome
from repeated_token import check_repeated_token  
from reverse import check_reverse


def test_model_comprehensive(model_name, check_model_func, test_cases):
    """Test model with multiple test cases."""
    
    print(f"\n{'='*60}")
    print(f"{model_name.upper()} MODEL COMPREHENSIVE TEST")
    print(f"{'='*60}")
    
    # Load models once
    print("Loading Haiku model...")
    haiku_model = check_model_func()
    print("Converting to PyTorch...")
    pytorch_model = haiku_to_pytorch(haiku_model)
    
    # Print model summary
    print("\nPyTorch Model Architecture:")
    summary(pytorch_model)
    
    # Test conversion with first test case
    test_input = test_cases[0]
    haiku_result = haiku_model.apply(test_input)
    pytorch_output = apply(pytorch_model, test_input)
    
    conversion_ok = outputs_equal(haiku_result.transformer_output, pytorch_output)
    print(f"\nModel conversion check: {'✅ PASS' if conversion_ok else '❌ FAIL'}")
    
    if not conversion_ok:
        max_diff = torch.max(torch.abs(
            torch.tensor(np.array(haiku_result.transformer_output), dtype=torch.float64) - pytorch_output
        ))
        print(f"Max difference: {max_diff}")
        return False
    
    # Test algorithm correctness on all test cases
    print(f"\nTesting {len(test_cases)} test cases:")
    print("-" * 40)
    
    passed = 0
    failed = 0
    
    for i, test_input in enumerate(test_cases, 1):
        haiku_result = haiku_model.apply(test_input)
        haiku_decoded = haiku_result.decoded
        
        # Skip BOS token for verification  
        input_tokens = test_input[1:] if test_input[0] == 'BOS' else test_input
        output_values = haiku_decoded[1:] if haiku_decoded[0] == 'BOS' else haiku_decoded
        
        correct = verify_algorithm_output(model_name, input_tokens, output_values)
        
        status = "✅" if correct else "❌"
        print(f"{i:2d}. {test_input} → {haiku_decoded} {status}")
        
        if correct:
            passed += 1
        else:
            failed += 1
            if failed <= 3:  # Show details for first few failures
                print(f"    Expected algorithm behavior not matched")
    
    accuracy = passed / len(test_cases)
    print(f"\nResults: {passed}/{len(test_cases)} passed ({accuracy:.1%})")
    
    success = conversion_ok and accuracy >= 0.9
    print(f"Overall: {'✅ PASS' if success else '❌ FAIL'}")
    
    return success


def verify_algorithm_output(model_name, input_tokens, output_values):
    """Verify algorithm output correctness."""
    
    if model_name == "Repeated Token":
        return verify_repeated_token(input_tokens, output_values)
    elif model_name == "Palindrome":
        return verify_palindrome(input_tokens, output_values)
    elif model_name == "Reverse":
        return verify_reverse(input_tokens, output_values)
    
    return False


def verify_repeated_token(tokens, output):
    """Verify repeated token detection."""
    if len(tokens) != len(output):
        return False
    
    for i, token in enumerate(tokens):
        count = tokens.count(token)
        expected = count > 1
        actual = output[i]
        if expected != actual:
            return False
    
    return True


def verify_palindrome(tokens, output):
    """Verify palindrome detection."""
    if len(tokens) != len(output):
        return False
    
    for i in range(len(tokens)):
        mirror_idx = len(tokens) - 1 - i
        expected = tokens[i] == tokens[mirror_idx]
        actual = output[i]
        if expected != actual:
            return False
    
    return True


def verify_reverse(tokens, output):
    """Verify sequence reversal."""
    if len(tokens) != len(output):
        return False
    
    expected_reverse = tokens[::-1]
    for i in range(len(tokens)):
        if expected_reverse[i] != output[i]:
            return False
    
    return True


def main():
    """Run comprehensive sanity checks."""
    
    print("TRACR TRANSFORMER COMPREHENSIVE SANITY CHECKS")
    print("=" * 60)
    
    # Test cases for each algorithm
    repeated_token_cases = [
        ['BOS', 'a'],           # No repeats
        ['BOS', 'a', 'b'],      # No repeats  
        ['BOS', 'a', 'a'],      # Simple repeat
        ['BOS', 'a', 'b', 'a'], # Repeat with gap
        ['BOS', 'a', 'b', 'b'], # Different repeat
        ['BOS', 'a', 'a', 'a'], # Multiple of same
        ['BOS', 'a', 'b', 'a', 'b'], # Multiple different repeats
        ['BOS', 'a', 'b', 'c'], # No repeats, longer
        ['BOS', 'a', 'b', 'c', 'd', 'a'], # Repeat at end
    ]
    
    palindrome_cases = [
        ['BOS', 'a'],           # Single char (palindrome)
        ['BOS', 'a', 'a'],      # Simple palindrome
        ['BOS', 'a', 'b'],      # Not palindrome
        ['BOS', 'a', 'b', 'a'], # Palindrome 
        ['BOS', 'a', 'b', 'c'], # Not palindrome
        ['BOS', 'a', 'b', 'b', 'a'], # Palindrome
        ['BOS', 'a', 'b', 'c', 'b', 'a'], # Palindrome
        ['BOS', 'a', 'b', 'c', 'd'], # Not palindrome
    ]
    
    reverse_cases = [
        ['BOS', 'a'],           # Single char
        ['BOS', 'a', 'b'],      # Two chars
        ['BOS', 'a', 'b', 'c'], # Three chars
        ['BOS', 'a', 'b', 'c', 'd'], # Four chars
        ['BOS', 'a', 'a'],      # Repeated chars
        ['BOS', 'a', 'b', 'a'], # Mixed
        ['BOS', 'z', 'y', 'x', 'w'], # End of alphabet
        ['BOS', 'h', 'e', 'l', 'l', 'o'], # Real word
        ['BOS', 'a', ' ', 'b'], # With space character
        ['BOS', 'x', 'y', 'z'], # Near end of alphabet
    ]
    
    # All available test configurations  
    all_test_configs = [
        ("Repeated Token", check_repeated_token, repeated_token_cases),
        ("Palindrome", check_palindrome, palindrome_cases),
        ("Reverse", check_reverse, reverse_cases)
    ]
    
    # Filter test configurations based on ALGORITHMS_TO_TEST setting
    algorithm_map = {
        "repeated_token": "Repeated Token",
        "palindrome": "Palindrome", 
        "reverse": "Reverse"
    }
    
    if ALGORITHMS_TO_TEST == "all":
        test_configs = all_test_configs
    elif ALGORITHMS_TO_TEST in algorithm_map:
        target_name = algorithm_map[ALGORITHMS_TO_TEST]
        test_configs = [tc for tc in all_test_configs if tc[0] == target_name]
    else:
        print(f"❌ Invalid ALGORITHMS_TO_TEST value: '{ALGORITHMS_TO_TEST}'")
        print("   Valid options: 'repeated_token', 'palindrome', 'reverse', 'all'")
        return
    
    print(f"Testing {len(test_configs)} algorithm(s): {', '.join([tc[0] for tc in test_configs])}")
    
    results = []
    
    for model_name, check_func, test_cases in test_configs:
        try:
            success = test_model_comprehensive(model_name, check_func, test_cases)
            results.append((model_name, success))
        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((model_name, False))
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    all_passed = True
    for model_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{model_name:20s}: {status}")
        if not success:
            all_passed = False
    
    print(f"\n{'='*60}")
    if all_passed:
        print("🎉 ALL COMPREHENSIVE SANITY CHECKS PASSED!")
        print("   Models are validated and ready for LLC analysis.")
        print("   You can now run the LLC estimation with confidence.")
    else:
        print("⚠️  SOME SANITY CHECKS FAILED!")
        print("   Please investigate the failing models before proceeding.")
        print("   Check the algorithm implementations and model compilation.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
