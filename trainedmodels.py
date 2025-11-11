import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import copy

from devinterp.optim import SGLD
from devinterp.slt.sampler import estimate_learning_coeff_with_summary
from devinterp.utils import plot_trace, default_nbeta

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
    def __init__(self, tracr_model, d_model, num_layers, num_heads, num_ff_layers, d_ff, d_k, d_v):
        super().__init__()
        self.tracr_model = tracr_model
        self.layers = nn.ModuleList([
            Block(d_model, num_heads, num_ff_layers, d_ff, d_k, d_v) for _ in range(num_layers)
        ])

    def forward(self, x):
        # x: (batch, seq)
        x = torch.tensor(self.tracr_model.apply(x).input_embeddings, dtype=torch.float32)
        # print(x.shape, x)
        for layer in self.layers:
            x = layer(x)
        return x

def evaluate(model, data):
    inputs, outputs = data[0]

    model_output = model.forward(inputs)

    if torch.isnan(sum(sum(sum(model_output)))):
        print(inputs)
        print(outputs)
        print(model_output)

    loss = F.cross_entropy(model_output, outputs)
    return loss, {
        "logits": model_output
    }

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
    hk_params = haiku_params_to_torch(assembled_model.params)
    hparams = infer_transformer_hparams(hk_params)

    num_layers = hparams['n_layers'] if hparam_dict['num_layers'] == -1 else hparam_dict['num_layers']
    num_heads = hparams['n_heads'] if hparam_dict['num_heads'] == -1 else hparam_dict['num_heads']
    num_ff_layers = 2 if hparam_dict['num_ff_layers'] == -1 else hparam_dict['num_ff_layers']
    d_ff = hparams['d_ff'] if hparam_dict['d_ff'] == -1 else hparam_dict['d_ff']
    d_k = hparams['d_k'] if hparam_dict['d_k'] == -1 else hparam_dict['d_k']
    d_v = hparams['d_v'] if hparam_dict['d_v'] == -1 else hparam_dict['d_v']

    model = Transformer(tracr_model = assembled_model, d_model = hparams['d_model'], num_layers = num_layers, num_heads = num_heads, num_ff_layers = num_ff_layers, d_ff = d_ff, d_k = d_k, d_v = d_v)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5, betas=(0.9,0.98), eps=1e-9)

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for example in dataloader:
            x = example[0][0]
            y = example[0][1]
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        running_loss /= len(dataloader)
        print(f"Epoch {epoch + 1}: loss = {running_loss:.4f}")

        learning_coeff_stats = estimate_learning_coeff_with_summary(
            copy.deepcopy(model),
            loader=dataloader,
            evaluate=evaluate,
            sampling_method=SGLD,
            optimizer_kwargs=dict(lr=1e-5, localization=1.0, nbeta=default_nbeta(dataloader)),
            num_chains=4,  # How many independent chains to run
            num_draws=100,  # How many samples to draw per chain
            num_burnin_steps=0,  # How many samples to discard at the beginning of each chain
            num_steps_bw_draws=1,  # How many steps to take between each sample
            device='cpu',
            online=True,
        )
        trace = learning_coeff_stats["loss/trace"]
        print(round(sum(learning_coeff_stats['llc/means'])/len(learning_coeff_stats['llc/means']), 2))
    
    model.eval()
    
    return model