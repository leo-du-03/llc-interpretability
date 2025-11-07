import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import jax
import jax.numpy as jnp

from tracr.haiku_to_pytorch import *

"""Attention block for PyTorch transformer"""
class Attention(nn.Module):
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
class MLP(nn.Module):
    def __init__(self, num_ff_layers, d_model, d_ff):
        super().__init__()
        layers = []
        layers.append(nn.Linear(d_model, d_ff))
        layers.append(nn.ReLU())
        for _ in range(num_ff_layers - 2):
            layers.append(nn.Linear(d_ff, d_ff))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(d_ff, d_model))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

"""PyTorch transformer block consisting of attention block followed by MLP block"""
class Block(nn.Module):
    def __init__(self, d_model, num_heads, num_ff_layers, d_ff, d_k, d_v):
        super().__init__()
        self.attn = Attention(d_model, num_heads, d_k, d_v)
        self.mlp = MLP(num_ff_layers, d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

"""PyTorch transformer architecture"""
class Transformer(nn.Module):
    def __init__(self, model, d_model, num_layers, num_heads, num_ff_layers, d_ff, d_k, d_v):
        super().__init__()
        self.model = model
        self.layers = nn.ModuleList([
            Block(d_model, num_heads, num_ff_layers, d_ff, d_k, d_v) for _ in range(num_layers)
        ])

    def forward(self, x):
        # x: (batch, seq)
        x = torch.tensor(self.model.apply(x).input_embeddings, dtype=torch.float64)
        # print(x.shape, x)
        for layer in self.layers:
            x = layer(x)
        return x

"""
Trains a model with the given hyperparameters.

Args:
    dataloader: the dataloader to train the model.
    assembled_model: the Tracr-compiled Haiku model.
    epochs: the number of epochs.
    num_layers: the number of attention + mlp blocks in the transformer.
    num_heads: the number of attention heads.
    num_ff_layers: the number of layers in each mlp (>= 2).
    d_ff: the size of the hidden layers in each mlp.
    d_k: hidden dimension of key and query matrices.
    d_v: hidden dimension of value matrix.

Returns:
    A Transformer trained using Adam on the provided dataloader.
"""
def train_model(dataloader, assembled_model, epochs, hparam_dict = dict(num_layers = -1, num_heads = -1, num_ff_layers = -1, d_ff = -1, d_k = -1, d_v = -1)):
    print("STARTING")
    hk_params = haiku_params_to_torch(assembled_model.params)
    hparams = infer_transformer_hparams(hk_params)
    print("INFERRED HYPERPARAMETERS")

    num_layers = hparams['n_layers'] if hparam_dict['num_layers'] == -1 else hparam_dict['num_layers']
    num_heads = hparams['n_heads'] if hparam_dict['num_heads'] == -1 else hparam_dict['num_heads']
    num_ff_layers = 2 if hparam_dict['num_ff_layers'] == -1 else hparam_dict['num_ff_layers']
    d_ff = hparams['d_ff'] if hparam_dict['d_ff'] == -1 else hparam_dict['d_ff']
    d_k = hparams['d_k'] if hparam_dict['d_k'] == -1 else hparam_dict['d_k']
    d_v = hparams['d_v'] if hparam_dict['d_v'] == -1 else hparam_dict['d_v']

    print(num_layers, num_heads, num_ff_layers, d_ff, d_k, d_v)

    model = Transformer(model = assembled_model, d_model = hparams['d_model'], num_layers = num_layers, num_heads = num_heads, num_ff_layers = num_ff_layers, d_ff = d_ff, d_k = d_k, d_v = d_v)
    criterion = nn.CrossEntropyLoss()
    for param in model.parameters():
        print(type(param), param.size)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9,0.98), eps=1e-9)

    model.train()  
    for epoch in range(epochs):
        for x, y in dataloader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}: loss = {loss.item():.4f}")
    
    model.eval()
    
    return model