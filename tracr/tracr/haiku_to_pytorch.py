import torch
import torch.nn as nn
import torch.nn.functional as F
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pprint
import re

from tracr.compiler import compiling
from tracr.compiler import lib
from tracr.rasp import rasp
from tracr.compiler import assemble

"""Attention block for PyTorch transformer"""
class TracrAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_k, d_v):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_v
        self.d_k = d_k

        self.query = nn.Linear(d_model, d_k, bias=True)
        self.key   = nn.Linear(d_model, d_k, bias=True)
        self.value = nn.Linear(d_model, d_v, bias=True)
        self.output = nn.Linear(num_heads * d_v, d_model, bias=True)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.output(out)

"""Multilayer Perceptron block for PyTorch transformer"""
class TracrMLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

"""PyTorch transformer block consisting of attention block followed by MLP block"""
class TracrBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, d_k, d_v):
        super().__init__()
        self.attn = TracrAttention(d_model, num_heads, d_k, d_v)
        self.mlp = TracrMLP(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

"""PyTorch transformer architecture"""
class TracrTransformer(nn.Module):
    def __init__(self, model, d_model, num_layers, num_heads, d_ff, d_k, d_v):
        super().__init__()
        self.model = model
        self.layers = nn.ModuleList([
            TracrBlock(d_model, num_heads, d_ff, d_k, d_v) for _ in range(num_layers)
        ])

    def forward(self, x):
        # x: (batch, seq)
        x = torch.tensor(self.model.apply(x).input_embeddings)
        print(x.shape, x)
        for layer in self.layers:
            x = layer(x)
        return x

"""
Prints out the parameters of a Haiku model and their dimensions for debugging.

Args:
    params: the parameters of the Haiku model (model.params).

Returns nothing.
"""
def summarize_haiku_params(params):
    for mod_name, submods in params.items():
        print(f"Module: {mod_name}")
        for sub_name, tensors in submods.items():
            w_shape = tensors.shape
            if len(w_shape) == 2:
                print(f"  {sub_name}: Linear layer ({w_shape[0]} → {w_shape[1]})")
            elif len(w_shape) == 1:
                print(f"  {sub_name}: Linear layer ({w_shape[0]})")


"""
Converts parameters of a Haiku model to torch tensors.

Args:
    params: the parameters of the Haiku model (model.params).

Returns:
    A nested dictionary of torch tensors.
"""
def haiku_params_to_torch(params):
    return jax.tree_util.tree_map(lambda x: torch.tensor(np.array(x)), params)

"""
Infers the hyperparameters of a Tracr-compiled transformer in Haiku given its parameters as torch tensors.

Args:
    hk_params: the parameters of the Haiku model as torch tensors (e.g. the result of calling
        haiku_params_to_torch with model.params).

Returns:
    A dictionary with keys:
        d_model: the input/output dimension of the model.
        d_k: the output dimension of the query and key matrices.
        d_v: the output dimension of the value matrix.
        d_ff: the size of the hidden layer in each MLP.
        n_layers: the number of layers of the model.
        n_heads: the number of attention heads in each attention block.
"""
def infer_transformer_hparams(hk_params):    
    # Get all layer names 
    layer_nums = set()
        for k in hk_params.keys():
        m = re.match(r"transformer/layer_(\d+)/", k)
        if m:
            layer_nums.add(int(m.group(1)))
    n_layers = max(layer_nums) + 1 if layer_nums else 0

    # Infer embedding dimension 
    # Typically from token_embed or pos_embed
    if "token_embed" in hk_params:
        d_model = hk_params["token_embed"]["embeddings"].shape[1]
    elif "pos_embed" in hk_params:
        d_model = hk_params["pos_embed"]["embeddings"].shape[1]
    else:
        # fallback: check first attention query weight
        first_query_key = next(k for k in hk_params.keys() if "attn/query" in k)
        d_model = hk_params[first_query_key].shape[0]

    # Infer feedforward hidden dimension (d_ff)
    first_ff = next(k for k in hk_params.keys() if "mlp/linear_1" in k)
    d_ff = hk_params[first_ff]['w'].shape[1]

    # Infer d_k
    first_query_key = next(k for k in hk_params.keys() if "attn/query" in k)
    q_weight = hk_params[first_query_key]['w']
    d_k = q_weight.shape[1]

    # Infer d_v
    first_value_key = next(k for k in hk_params.keys() if "attn/value" in k)
    v_weight = hk_params[first_value_key]['w']
    d_v = v_weight.shape[1]

    # Infer number of heads from value (shape: d * d_v) and output (shape: n_heads * d_v * d) matrices
    first_output_key = next(k for k in hk_params.keys() if "attn/linear" in k)
    o_weight = hk_params[first_output_key]['w']
    n_heads = o_weight.shape[0] // d_v

    return dict(
        d_model=d_model,
        d_k=d_k,
        d_v=d_v,
        d_ff=d_ff,
        n_layers=n_layers,
        n_heads=n_heads,
    )

"""
Sets the parameters of a PyTorch transformer with the same architecture as a Tracr-compiled 
transformer in Haiku to have the Haiku model's parameters.

Args:
    model: a TracrTransformer with the same hyperparameters as the Haiku model with parameters hk_params.
    hk_params: the parameters of the Haiku model as torch tensors (e.g. the result of calling
        haiku_params_to_torch with model.params).

Returns nothing.
"""
def load_haiku_to_torch(model, hk_params):
    # Iterate through each transformer layer
    layer_idx = 0
    while f"transformer/layer_{layer_idx}/attn/query" in hk_params:
        layer = model.layers[layer_idx]
        prefix = f"transformer/layer_{layer_idx}"

        # Attention weights
        attn = layer.attn

        attn.query.weight.data = hk_params[f"{prefix}/attn/query"]['w'].T
        attn.query.bias.data = hk_params[f"{prefix}/attn/query"]['b']
        attn.key.weight.data = hk_params[f"{prefix}/attn/key"]['w'].T
        attn.key.bias.data = hk_params[f"{prefix}/attn/key"]['b']
        attn.value.weight.data = hk_params[f"{prefix}/attn/value"]['w'].T
        attn.value.bias.data = hk_params[f"{prefix}/attn/value"]['b']

        # Attention output projection
        attn.output.weight.data = hk_params[f"{prefix}/attn/linear"]['w'].T
        attn.output.bias.data = hk_params[f"{prefix}/attn/linear"]['b']

        # MLP
        mlp = layer.mlp

        mlp.linear1.weight.data = hk_params[f"{prefix}/mlp/linear_1"]['w'].T
        mlp.linear1.bias.data = hk_params[f"{prefix}/mlp/linear_1"]['b']
        mlp.linear2.weight.data = hk_params[f"{prefix}/mlp/linear_2"]['w'].T
        mlp.linear2.bias.data = hk_params[f"{prefix}/mlp/linear_2"]['b']

        layer_idx += 1


"""
Converts a Tracr-compiled transformer in Haiku into an equivalent PyTorch transformer.

Args:
    assembled_model: a Tracr-compiled transformer.

Returns:
    A TracrTransformer that is equivalent to assembled_model.
"""
def haiku_to_pytorch(assembled_model):
    params = assembled_model.params
    hk_params = haiku_params_to_torch(params)
    hparams = infer_transformer_hparams(hk_params)

    model = TracrTransformer(
        model=assembled_model,
        d_model=hparams['d_model'],
        num_layers=hparams['n_layers'],
        num_heads=hparams['n_heads'],
        d_ff=hparams['d_ff'],
        d_k=hparams['d_k'],
        d_v=hparams['d_v']
    )

    # Load weights
    load_haiku_to_torch(model, hk_params)
    return model

"""
Runs a TracrTransformer on a provided input.

Args:
    model: a TracrTransformer.
    x: the input to the transformer.

Returns:
    y: the raw output of the transformer.
"""
def apply(model, x):
    with torch.no_grad():
        y = model(x)
    return y

"""
Checks if the outputs of a Tracr-compiled Haiku transformer and a TracrTransformer are equal.

Args:
    hk_output: the transformer output of the Haiku model (e.g. model.apply(x).transformer_output)
    torch_output: the output of the TracrTransformer

Returns:
    True if their tensor representations are equal, otherwise False.
"""
def outputs_equal(hk_output, torch_output):
    return torch.equal(torch.tensor(np.array(hk_output)), torch_output)