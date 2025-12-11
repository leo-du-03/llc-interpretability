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

    for i, word in enumerate(test_cases, 1):
        out = model.apply(word)
        target = ['BOS'] + [a == b for a, b in zip(word[1:], word[1:][::-1])]
        
        if target == out.decoded:
            print(".", end="", flush=True)
        else:
            print(
                f"\n--- Decoded Output Mismatch in Test Case {i} ---\n"
                f"Input:    {word}\n"
                f"Expected: {target}\n"
                f"Got:      {out.decoded}\n"
            )
    print("\n")

def peakTests():
    model = get_peak_model()

    test_cases = get_peak_test_cases()
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

def fractokTests():
    model = check_fractok()

    print("Testing haiku model accuracy:")
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

def checkParamEquality(hk_model):
    hk_params = hk_model.params
    pt_params = haiku_params_to_torch(hk_params)

    success = True
    for k in hk_params.keys():
        hk_p = hk_params[k]
        pt_p = pt_params[k]

        for k in hk_p:
            hk_p_k = hk_p[k]
            pt_p_k = pt_p[k]
            if np.array(hk_p_k).tolist() == np.array(pt_p_k).tolist():
                print(".", end="", flush=True)
            else:
                success = False
                print(
                    f"\n--- Parameter Mismatch in Parameter {k} ---\n"
                    f"Expected: {np.array(hk_p_k).tolist()}\n"
                    f"Got:      {np.array(pt_p_k).tolist()}\n"
                )
    return success

def checkSuccessfulParamTransfer(hk_model, pt_model):
    hk_params = hk_model.params
    layer_idx = 0
    while f"transformer/layer_{layer_idx}/attn/query" in hk_params:
        layer = pt_model.layers[layer_idx]
        prefix = f"transformer/layer_{layer_idx}"

        # Attention weights
        attn = layer.attn
        assert attn.query.bias.data.tolist() == hk_params[f"{prefix}/attn/query"]['b'].tolist()
        assert attn.key.weight.data.tolist() == hk_params[f"{prefix}/attn/key"]['w'].T.tolist()
        assert attn.key.bias.data.tolist() == hk_params[f"{prefix}/attn/key"]['b'].tolist()
        assert attn.value.weight.data.tolist() == hk_params[f"{prefix}/attn/value"]['w'].T.tolist()
        assert attn.value.bias.data.tolist() == hk_params[f"{prefix}/attn/value"]['b'].tolist()

        # Attention output projection
        assert attn.output.weight.data.tolist() == hk_params[f"{prefix}/attn/linear"]['w'].T.tolist()
        assert attn.output.bias.data.tolist() == hk_params[f"{prefix}/attn/linear"]['b'].tolist()

        # MLP
        mlp = layer.mlp

        assert mlp.linear1.weight.data.tolist() == hk_params[f"{prefix}/mlp/linear_1"]['w'].T.tolist()
        assert mlp.linear1.bias.data.tolist() == hk_params[f"{prefix}/mlp/linear_1"]['b'].tolist()
        assert mlp.linear2.weight.data.tolist() == hk_params[f"{prefix}/mlp/linear_2"]['w'].T.tolist()
        assert mlp.linear2.bias.data.tolist() == hk_params[f"{prefix}/mlp/linear_2"]['b'].tolist()

        layer_idx += 1

def haikuToPytorchTests():
    print("Testing palindrome:")
    hk_model = check_palindrome()
    torch_model = haiku_to_pytorch(hk_model)
    checkParamEquality(hk_model)
    checkSuccessfulParamTransfer(hk_model, torch_model)
    print("\n")

    print("Testing peak:")
    hk_model = get_peak_model()
    torch_model = haiku_to_pytorch(hk_model)
    checkParamEquality(hk_model)
    checkSuccessfulParamTransfer(hk_model, torch_model)
    print("\n")

    print("Testing Fractok:")
    hk_model = check_fractok()
    torch_model = haiku_to_pytorch(hk_model)
    checkParamEquality(hk_model)
    checkSuccessfulParamTransfer(hk_model, torch_model)

    print("\n")
if __name__ == "__main__":
    print("Running Palindrome Tests")
    palindromesTests()
    print("Running Peak Tests")
    peakTests()
    print("Running Fractok Tests")
    fractokTests()
    print("Testing Haiku to PyTorch conversion")
    haikuToPytorchTests()