#!/usr/bin/env python3
"""
Simple sanity check script for TRACR compiled transformers.

This script provides a quick validation of each model's conversion and basic functionality.
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


def test_model_basic(model_name, check_model_func, test_input):
    """Test basic model loading, conversion and a simple prediction."""
    
    print(f"\n{'='*60}")
    print(f"{model_name.upper()} MODEL TEST")
    print(f"{'='*60}")
    
    # Load Haiku model
    print("Loading Haiku model...")
    haiku_model = check_model_func()
    
    # Convert to PyTorch
    print("Converting to PyTorch...")
    pytorch_model = haiku_to_pytorch(haiku_model)
    
    # Print model summary
    print("\nPyTorch Model Architecture:")
    summary(pytorch_model)
    
    # Test with sample input
    print(f"\nTesting with input: {test_input}")
    
    # Get Haiku model result
    haiku_result = haiku_model.apply(test_input)
    haiku_decoded = haiku_result.decoded
    haiku_transformer_output = haiku_result.transformer_output
    
    print(f"Haiku decoded output: {haiku_decoded}")
    print(f"Haiku transformer output shape: {haiku_transformer_output.shape}")
    
    # Get PyTorch model result
    pytorch_output = apply(pytorch_model, test_input)
    print(f"PyTorch transformer output shape: {pytorch_output.shape}")
    
    # Check if outputs match
    outputs_match = outputs_equal(haiku_transformer_output, pytorch_output)
    print(f"Transformer outputs match: {outputs_match}")
    
    if outputs_match:
        print("✅ Model conversion and basic test PASSED!")
    else:
        print("❌ Model conversion test FAILED!")
        max_diff = torch.max(torch.abs(
            torch.tensor(np.array(haiku_transformer_output), dtype=torch.float64) - pytorch_output
        ))
        print(f"Max difference: {max_diff}")
    
    # Manual verification of the algorithm
    manual_correct = verify_algorithm_output(model_name, test_input, haiku_decoded)
    if manual_correct:
        print("✅ Algorithm logic verification PASSED!")
    else:
        print("❌ Algorithm logic verification FAILED!")
    
    return outputs_match and manual_correct


def verify_algorithm_output(model_name, test_input, decoded_output):
    """Manually verify if the algorithm output is correct."""
    
    # Skip BOS token for verification
    if test_input[0] == 'BOS':
        input_tokens = test_input[1:]
        # Skip BOS in decoded output if present
        if decoded_output[0] == 'BOS':
            output_values = decoded_output[1:]
        else:
            output_values = decoded_output
    else:
        input_tokens = test_input
        output_values = decoded_output
    
    print(f"Verifying {model_name} algorithm:")
    print(f"  Input tokens: {input_tokens}")
    print(f"  Output values: {output_values}")
    
    if model_name == "Repeated Token":
        return verify_repeated_token(input_tokens, output_values)
    elif model_name == "Palindrome":
        return verify_palindrome(input_tokens, output_values)
    elif model_name == "Reverse":
        return verify_reverse(input_tokens, output_values)
    
    return False


def verify_repeated_token(tokens, output):
    """Verify repeated token detection logic."""
    if len(tokens) != len(output):
        print(f"  ❌ Length mismatch: {len(tokens)} vs {len(output)}")
        return False
    
    for i, token in enumerate(tokens):
        count = tokens.count(token)
        expected = count > 1
        actual = output[i]
        
        print(f"  Token '{token}' at pos {i}: count={count}, expected={expected}, actual={actual}")
        
        if expected != actual:
            print(f"  ❌ Mismatch at position {i}")
            return False
    
    return True


def verify_palindrome(tokens, output):
    """Verify palindrome detection logic."""
    if len(tokens) != len(output):
        print(f"  ❌ Length mismatch: {len(tokens)} vs {len(output)}")
        return False
    
    for i in range(len(tokens)):
        mirror_idx = len(tokens) - 1 - i
        expected = tokens[i] == tokens[mirror_idx]
        actual = output[i]
        
        print(f"  Pos {i} ('{tokens[i]}') vs pos {mirror_idx} ('{tokens[mirror_idx]}'): expected={expected}, actual={actual}")
        
        if expected != actual:
            print(f"  ❌ Mismatch at position {i}")
            return False
    
    return True


def verify_reverse(tokens, output):
    """Verify sequence reversal logic."""
    if len(tokens) != len(output):
        print(f"  ❌ Length mismatch: {len(tokens)} vs {len(output)}")
        return False
    
    expected_reverse = tokens[::-1]
    
    for i in range(len(tokens)):
        expected = expected_reverse[i]
        actual = output[i]
        
        print(f"  Pos {i}: expected='{expected}', actual='{actual}'")
        
        if expected != actual:
            print(f"  ❌ Mismatch at position {i}")
            return False
    
    return True


def main():
    """Run simplified sanity checks."""
    
    print("TRACR TRANSFORMER SANITY CHECKS")
    print("=" * 60)
    print(f"Testing: {ALGORITHMS_TO_TEST.upper()}")
    print("=" * 60)
    
    # All available test cases
    all_test_cases = [
        ("Repeated Token", check_repeated_token, ['BOS', 'a', 'b', 'a']),
        ("Palindrome", check_palindrome, ['BOS', 'a', 'b', 'a']),
        ("Reverse", check_reverse, ['BOS', 'h', 'e', 'l', 'l', 'o'])
    ]
    
    # Filter test cases based on configuration
    test_cases = []
    algorithm_map = {
        "repeated_token": "Repeated Token",
        "palindrome": "Palindrome", 
        "reverse": "Reverse"
    }
    
    if ALGORITHMS_TO_TEST == "all":
        test_cases = all_test_cases
    elif ALGORITHMS_TO_TEST in algorithm_map:
        target_name = algorithm_map[ALGORITHMS_TO_TEST]
        test_cases = [tc for tc in all_test_cases if tc[0] == target_name]
    else:
        print(f"❌ Invalid ALGORITHMS_TO_TEST value: '{ALGORITHMS_TO_TEST}'")
        print("   Valid options: 'repeated_token', 'palindrome', 'reverse', 'all'")
        return
    
    results = []
    
    for model_name, check_func, test_input in test_cases:
        try:
            success = test_model_basic(model_name, check_func, test_input)
            results.append((model_name, success))
        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")
            results.append((model_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    all_passed = True
    for model_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{model_name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All sanity checks PASSED! Models are ready for LLC analysis.")
    else:
        print("\n⚠️ Some sanity checks FAILED. Please review before proceeding.")


if __name__ == "__main__":
    main()
