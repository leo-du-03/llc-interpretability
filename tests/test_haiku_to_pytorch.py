"""
Tests for the haiku_to_pytorch function
Makes sure parameters are correctly converted from jax arrays to pytorch tensors, then parameters are correctly loaded
"""

import numpy as np
from rasp_models.palindrome import check_palindrome
from rasp_models.peak import get_peak_model
from rasp_models.fractok import check_fractok
from tracr.haiku_to_pytorch import haiku_to_pytorch, haiku_params_to_torch

def checkParamEquality(hk_model):
    '''
    Makes sure the haiku parameters are converted correctly into pytorch parameters.
    
    :param hk_model: haiku model you are turning into a pytorch model
    '''
    hk_params = hk_model.params
    pt_params = haiku_params_to_torch(hk_params)

    success = True
    for k in hk_params.keys():
        hk_p = hk_params[k]
        pt_p = pt_params[k]

        for k in hk_p:
            hk_p_k = hk_p[k]
            pt_p_k = pt_p[k]
            if np.array(hk_p_k, copy=None).tolist() == np.array(pt_p_k, copy=None).tolist():
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
    '''
    Checks to make sure the pytorch model's parameters are successfully loaded

    :param hk_model: haiku model you are pulling parameters from
    :param pt_model: pytorch model you have loaded parameters into
    '''
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
    '''
    Tests the haikuToPytorch function's critical capabilities:
        1. Ensures parameters are correctly converted from haiku to pytorch
        2. Ensures parameters are successfully loaded into pytorch model
    If these 2 criteria are met, the haiku and pytorch models are considered equal
    '''
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
    haikuToPytorchTests()