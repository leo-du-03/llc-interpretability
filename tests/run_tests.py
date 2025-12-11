from rasp_models.palindrome import check_palindrome
from rasp_models.peak import get_peak_model
from rasp_models.fractok import check_fractok
from tracr.haiku_to_pytorch import haiku_to_pytorch, apply, outputs_equal, haiku_params_to_torch
from datasets.palindrome_data import generate_non_palindromes, generate_palindromes
from datasets.peak_data import get_peak_test_cases
from datasets.fractok_data import generate_all_prev_fraction_tokens_x_testcases, generate_fractok_training_data
import torch
import numpy as np

def palindromesTests():
    bos = "BOS"
    model = check_palindrome()
    torch_model = haiku_to_pytorch(model)

    test_cases = []

    print("Compiling test cases:")
    with open("tests/palindromes.txt") as file:
        lines = [line.rstrip() for line in file]
    for line in lines:
        word = [bos] + list(line)
        test_cases.append(word)

    rand_pals = generate_palindromes(50)
    for line in rand_pals:
        word = [bos] + list(line)
        test_cases.append(word)
    
    with open("tests/non_palindromes.txt") as file:
        lines = [line.rstrip() for line in file]
    for line in lines:
        word = [bos] + list(line)
        test_cases.append(word)

    rand_non_pals = generate_non_palindromes(50)
    for line in rand_non_pals:
        word = [bos] + list(line)
        test_cases.append(word)
    print("done.")
    print("Testing haiku model accuracy:")

    for word in test_cases:
        out = model.apply(word)
        target = ['BOS'] + [a == b for a, b in zip(word[1:], word[1:][::-1])]
        
        if target == out.decoded:
            print(".", end="")
        else:
            (
                f"\n--- Decoded Output Mismatch in Test Case {i} ---\n"
                f"Input:    {word}\n"
                f"Expected: {target}\n"
                f"Got:      {out.decoded}\n"
            )
    print("\n")

    print("Testing haiku and pytorch model equality:")
    for word in test_cases:
        out = model.apply(word)
        pytorch_out = apply(torch_model, word)
        if outputs_equal(out.transformer_output, pytorch_out):
            print(".", end="")
        else:
            (
                f"\n--- Decoded Output Mismatch in Test Case {i} ---\n"
                f"Input:    {word}\n"
                f"Expected: {target}\n"
                f"Got:      {out.decoded}\n"
            )
    print("\n")

def peakTests():
    model = get_peak_model()
    torch_model = haiku_to_pytorch(model)
    # params = model.params
    # print(params)
    # print(haiku_params_to_torch(params))
    # for name, param in torch_model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)
    test_cases = get_peak_test_cases()
    for i, (input_seq, expected_output) in enumerate(test_cases, 1):
        out = model.apply(input_seq)
        pytorch_out = apply(torch_model, input_seq)

        assert outputs_equal(out.transformer_output, pytorch_out), (
            f"\n--- Transformer Mismatch in Test Case {i} ---\n"
            f"Input:      {input_seq}\n"
            f"Expected:   {out.transformer_output}\n"
            f"Got:        {pytorch_out}\n"
            f"Difference: {torch.tensor(np.array(out.transformer_output), dtype=torch.float64) - pytorch_out}\n"
        )
        decoded = out.decoded
        assert decoded == expected_output, (
            f"\n--- Decoded Output Mismatch in Test Case {i} ---\n"
            f"Input:    {input_seq}\n"
            f"Expected: {expected_output}\n"
            f"Got:      {decoded}\n"
        )

def fractokTests():
    model = check_fractok()
    torch_model = haiku_to_pytorch(model)

    print("Testing haiku model accuracy:")
    test_cases = generate_fractok_training_data()
    for i, (input_seq, target) in enumerate(test_cases, 1):
        out = model.apply(input_seq)
        decoded = out.decoded
        decoded = [0] + decoded[1:]
        if decoded == target.tolist():
            print(".", end="")
        else:
            print(
                f"\n--- Decoded Output Mismatch in Test Case {i} ---\n"
                f"Input:    {input_seq}\n"
                f"Expected: {target}\n"
                f"Got:      {decoded}\n"
            )
    print("\n")

    print("Testing haiku and pytorch model equality:")
    test_cases = generate_all_prev_fraction_tokens_x_testcases()
    for i, (input_seq, out) in enumerate(test_cases, 1):
        pytorch_out = apply(torch_model, input_seq)

        if outputs_equal(out.tolist(), pytorch_out):
            print(".", end="")
        else:
            f"\n--- Transformer Mismatch in Test Case {i} ---\n"
            f"Input:      {input_seq}\n"
            f"Expected:   {out.tolist()}\n"
            f"Got:        {pytorch_out}\n"
    print("\n")

if __name__ == "__main__":
    print("Running Palindrome Tests")
    palindromesTests()
    # print("Running Peak Tests")
    # peakTests()
    print("Running Fractok Tests")
    fractokTests()