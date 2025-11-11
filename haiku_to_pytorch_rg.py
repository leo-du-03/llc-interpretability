import torch
import torch.nn as nn
import torch.nn.functional as F
import jax
import numpy as np
import re

# ========== MODULE DEFINITIONS ==========

class TracrAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_k, d_v):
        super().__init__()
        self.query = nn.Linear(d_model, d_k)
        self.key   = nn.Linear(d_model, d_k)
        self.value = nn.Linear(d_model, d_v)
        self.output = nn.Linear(num_heads * d_v, d_model)
        self.num_heads = num_heads
        self.head_dim = d_v

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.output(out)


class TracrMLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


class TracrBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, d_k, d_v):
        super().__init__()
        self.attn = TracrAttention(d_model, num_heads, d_k, d_v)
        self.mlp = TracrMLP(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class TracrTransformer(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, d_ff, d_k, d_v, vocab_size, max_seq_len):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len + 1, d_model))
        self.layers = nn.ModuleList([
            TracrBlock(d_model, num_heads, d_ff, d_k, d_v)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        token_emb = self.token_embed(x)
        pos = self.pos_embed[:, :x.size(1), :]
        x = token_emb + pos
        for layer in self.layers:
            x = layer(x)
        return x

# ========== CONVERSION UTILS ==========

def haiku_params_to_torch(params):
    return jax.tree_util.tree_map(lambda x: torch.tensor(np.array(x), dtype=torch.float64), params)


def infer_transformer_hparams(hk_params):
    layer_nums = set()
    for k in hk_params.keys():
        m = re.match(r"transformer/layer_(\d+)/", k)
        if m:
            layer_nums.add(int(m.group(1)))
    n_layers = max(layer_nums) + 1 if layer_nums else 0

    d_model = hk_params["token_embed"]["embeddings"].shape[1]
    d_ff = hk_params["transformer/layer_0/mlp/linear_1"]["w"].shape[1]
    d_k = hk_params["transformer/layer_0/attn/query"]["w"].shape[1]
    d_v = hk_params["transformer/layer_0/attn/value"]["w"].shape[1]
    vocab_size = hk_params["token_embed"]["embeddings"].shape[0]
    max_seq_len = hk_params["pos_embed"]["embeddings"].shape[0] - 1
    n_heads = hk_params["transformer/layer_0/attn/linear"]["w"].shape[0] // d_v

    return dict(
        d_model=d_model,
        d_k=d_k,
        d_v=d_v,
        d_ff=d_ff,
        n_layers=n_layers,
        n_heads=n_heads,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
    )


def load_haiku_to_torch(model, hk_params):
    model.token_embed.weight.data = hk_params["token_embed"]["embeddings"]
    model.pos_embed.data = hk_params["pos_embed"]["embeddings"].unsqueeze(0)

    for i, layer in enumerate(model.layers):
        prefix = f"transformer/layer_{i}"
        attn = layer.attn
        mlp = layer.mlp

        attn.query.weight.data = hk_params[f"{prefix}/attn/query"]["w"].T
        attn.query.bias.data = hk_params[f"{prefix}/attn/query"]["b"]
        attn.key.weight.data = hk_params[f"{prefix}/attn/key"]["w"].T
        attn.key.bias.data = hk_params[f"{prefix}/attn/key"]["b"]
        attn.value.weight.data = hk_params[f"{prefix}/attn/value"]["w"].T
        attn.value.bias.data = hk_params[f"{prefix}/attn/value"]["b"]
        attn.output.weight.data = hk_params[f"{prefix}/attn/linear"]["w"].T
        attn.output.bias.data = hk_params[f"{prefix}/attn/linear"]["b"]

        mlp.linear1.weight.data = hk_params[f"{prefix}/mlp/linear_1"]["w"].T
        mlp.linear1.bias.data = hk_params[f"{prefix}/mlp/linear_1"]["b"]
        mlp.linear2.weight.data = hk_params[f"{prefix}/mlp/linear_2"]["w"].T
        mlp.linear2.bias.data = hk_params[f"{prefix}/mlp/linear_2"]["b"]


def haiku_to_pytorch(assembled_model):
    hk_params = haiku_params_to_torch(assembled_model.params)
    hparams = infer_transformer_hparams(hk_params)

    model = TracrTransformer(
        d_model=hparams["d_model"],
        num_layers=hparams["n_layers"],
        num_heads=hparams["n_heads"],
        d_ff=hparams["d_ff"],
        d_k=hparams["d_k"],
        d_v=hparams["d_v"],
        vocab_size=hparams["vocab_size"],
        max_seq_len=hparams["max_seq_len"],
    )
    load_haiku_to_torch(model, hk_params)
    return model
