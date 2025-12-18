import torch.nn as nn
import torch

"""
Contains different sized pytorch models that all use transformers and are meant to be trained on the peak detection task
"""

class Nano(nn.Module):
    """
    Smallest model in this file.
    Contains a single transformer, and two linear layers.
    """
    def __init__(self, params):
        super().__init__()
        self.embedding = nn.Embedding(params.v, params.embed_dim)
        self.transformer = nn.TransformerEncoderLayer(d_model=params.embed_dim, nhead=1, dim_feedforward=params.hidden_size, batch_first=True)
        self.linear1 = nn.Linear(params.embed_dim, params.linear_hidden_size, bias=True)
        self.linear2 = nn.Linear(params.linear_hidden_size, params.embed_dim, bias=True)
        self.out = nn.Linear(params.embed_dim, 2, bias=False)
        self.act = nn.GELU()
        self.softmax = nn.Softmax(dim=2)

        self.vocab_size = params.v
        self.length = params.l
        self.blocks = params.blocks
        self.hidden_size = params.hidden_size

    def forward(self, x):
        embs = []
        for i in range(self.length):
            embs.append(self.embedding(x[..., i]).tolist())
            
        x = torch.tensor(embs).transpose(0, 1)
        x = self.transformer(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.act(x)
        x = self.out(x)
        x = self.softmax(x)
        return x
    
class Small(nn.Module):
    """
    Second smallest model in this file.
    Contains l transformers and l + 1 linear layers, where l is the maximum length of an input.
    Each element of the input is fed into its own transformer and linear layer.
    """
    def __init__(self, params):
        super().__init__()
        self.embedding = nn.Embedding(params.v, params.embed_dim)
        self.transformers = nn.ModuleList()
        self.elem_linears = nn.ModuleList()
        for _ in range(params.l):
            self.transformers.append(nn.TransformerEncoderLayer(d_model=params.embed_dim, nhead=1, dim_feedforward=params.hidden_size))
            self.elem_linears.append(nn.Linear(params.embed_dim, params.hidden_size, bias=True))
        self.linear2 = nn.Linear(params.hidden_size, params.l, bias=False)
        self.act = nn.GELU()
        self.vocab_size = params.v
        self.length = params.l
        self.hidden_size = params.hidden_size
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embs = []
        for i in range(self.length):
            embs.append(self.embedding(x[..., i]))
        x = None
        for i in range(len(embs)):
            if x is None:
                temp = self.transformers[i](embs[i])
                x = self.elem_linears[i](temp)
            else:
                temp = self.transformers[i](embs[i])
                x += self.elem_linears[i](temp)
        x = self.act(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x
    
class Medium_Large(nn.Module):
    """
    Largest model in this file.
    Has params.blocks number of transformers, and params.blocks * 2 + 1 linear layers.
    Input is passed sequentially through the blocks.
    Each block contains a transformer, and two linear layers.
    """
    def __init__(self, params):
        super().__init__()
        self.embedding = nn.Embedding(params.v, params.embed_dim)
        self.transformers = nn.ModuleList()
        self.elem_linears1 = nn.ModuleList()
        self.elem_linears2 = nn.ModuleList()
        for _ in range(params.blocks):
            self.transformers.append(nn.TransformerEncoderLayer(d_model=params.embed_dim, nhead=1, dim_feedforward=params.hidden_size, batch_first=True))
            self.elem_linears1.append(nn.Linear(params.embed_dim, params.linear_hidden_size, bias=True))
            self.elem_linears2.append(nn.Linear(params.linear_hidden_size, params.embed_dim, bias=True))
        self.out = nn.Linear(params.embed_dim, 2, bias=False)
        self.act = nn.GELU()
        self.softmax = nn.Softmax(dim=2)

        self.vocab_size = params.v
        self.length = params.l
        self.blocks = params.blocks
        self.hidden_size = params.hidden_size

    def forward(self, x):
        embs = []
        for i in range(self.length):
            embs.append(self.embedding(x[..., i]).tolist())

        x = torch.tensor(embs).transpose(0, 1)
        for i in range(self.blocks):
            x = self.transformers[i](x)
            x = self.elem_linears1[i](x)
            x = self.elem_linears2[i](x)
        x = self.act(x)
        x = self.out(x)
        x = self.softmax(x)
        return x
