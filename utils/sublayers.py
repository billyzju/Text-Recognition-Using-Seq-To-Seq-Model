# --------------------------------------------------------------------------------
# 		Import
# --------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torchvision
import math
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


# --------------------------------------------------------------------------------
#       Funcs
# --------------------------------------------------------------------------------
def attention(q, k, v, d_k, mask=None, dropout=None):
    """ Calculate attention scores
        Input: three matrices are q, k, v, mask for padding and dropout
        Output: scores
    """
    # dot product between query and key
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply mask to reduce values where the input i padding
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    # dot product between scores and value
    output = torch.matmul(scores, v)
    return output


# --------------------------------------------------------------------------------
#       Classes
# --------------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention scores show how much each word will be expressed
    at this position
    """
    def __init__(self, heads, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert d_model % heads == 0
        # Length of each emb
        self.d_model = d_model
        self.d_k = d_model // heads
        # Numbers of heads
        self.heads = heads

        # Wq matrix
        self.q_linear = nn.Linear(d_model, d_model)
        # Wv matrix
        self.v_linear = nn.Linear(d_model, d_model)
        # Wk matrix
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        # Perform linear operation and split into h heads
        k = self.k_linear(k).view(batch_size, -1, self.heads, self.d_k)
        q = self.q_linear(q).view(batch_size, -1, self.heads, self.d_k)
        v = self.v_linear(v).view(batch_size, -1, self.heads, self.d_k)

        # Transpose to batch_size * heads * sequence_len * d_k
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # Calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # Concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(batch_size,
                                                          -1, self.d_model)
        output = self.out(concat)
        return output


class FeedForward(nn.Module):
    """
    After Attention, FeedForward layer is used, it just a simple architecture
    """
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(FeedForward, self).__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Norm(nn.Module):
    """
    Normalization for result between each layer in both encoder and decoder
    """
    def __init__(self, d_model, eps=1e-6):
        super(Norm, self).__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalization
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True))\
                            / (x.std(dim=-1, keepdim=True) + self.eps)\
                            + self.bias
        return norm


class PositionalEncoder(nn.Module):
    """
    Add position encoding into embeddings
    """
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoder, self).__init__()

        self.d_model = d_model
        # create constant 'pe' matrix with values dependant on pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, 0:seq_len, :], requires_grad=False).cuda()
        return x


class Embedder(nn.Module):
    """
    Convert from word to embedding for target
    """
    def __init__(self, vocab_size, d_model):
        super(Embedder, self).__init__()

        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        x = x.long()
        return self.embed(x) * math.sqrt(self.d_model)
