# --------------------------------------------------------------------------------
# 		Import
# --------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torchvision
from utils.sublayers import MultiHeadAttention
from utils.sublayers import FeedForward
from utils.sublayers import Norm
from utils.sublayers import PositionalEncoder
from utils.sublayers import Embedder
import copy


# --------------------------------------------------------------------------------
#       Funcs
# --------------------------------------------------------------------------------
def get_clones(module, N):
    """ Copy layer to generate multi layer for Transformer
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# --------------------------------------------------------------------------------
#       Classes
# --------------------------------------------------------------------------------
class EncoderLayer(nn.Module):
    """ Encoder layer is one element of Transformer Encoder
        It contains two main sublayers which are attention and feedforward
    """
    def __init__(self, d_model, heads, dropout=0.1):
        """ Input:
           d_model is length of vector of embeddings
           heads is number of heads for MultiHeadAttetion
        """
        super(EncoderLayer, self).__init__()

        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)

        # Two main sub-layers
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)

        # Residual dropout
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        """ Input:
            x is the embs
            mask is the mask of sequence after adding padding
        """
        x_norm = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x_norm, x_norm, x_norm, mask))
        x_norm = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x_norm))
        return x


class DecoderLayer(nn.Module):
    """ Decoder layer is one element of Transformer Decoder
    """
    def __init__(self, d_model, heads, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.attn_1 = MultiHeadAttention(heads, d_model)
        self.attn_2 = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model).cuda()

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        """ Input:
            x is target embedding
            e_outputs is output of Encoder
            src_mask is the mask from source
            trg_mask is target mask
        """
        x_norm = self.norm_1(x)
        # Self attention
        x = x + self.dropout_1(self.attn_1(x_norm, x_norm, x_norm, trg_mask))
        x_norm = self.norm_2(x)

        # Source attention
        x = x + self.dropout_2(self.attn_2(x_norm, e_outputs, e_outputs,
                                           src_mask))
        x_norm = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x_norm))
        return x


class TransformerEncoder(nn.Module):
    """ Encoder of Transformer
        Embeddings is obtained by CNN and feedforward to Encoder row by row
        Moreover, Positional encoding is add into embs
    """
    def __init__(self, d_model, N, heads, max_seq_len):
        super(TransformerEncoder, self).__init__()

        self.N = N
        self.position = PositionalEncoder(d_model, max_seq_len)
        # Generate N Encoder layers
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        """ Input:
            src is embedings from CNN
            mask is the mask of sequence after adding padding
        """
        x = src
        x = self.position(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, max_seq_len):
        super(TransformerDecoder, self).__init__()

        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.position = PositionalEncoder(d_model, max_seq_len)
        # Generate N Decoder layers
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        """ Input:
            trg is output embedding in Transformer paper, it also is the
            previous outputs
        """
        x = self.embed(trg)
        x = self.position(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


class Transformer(nn.Module):
    """ The final model for Transformer
    """
    def __init__(self, trg_vocab, d_model, N, heads, max_seq_len):
        super(Transformer, self).__init__()

        self.encoder = TransformerEncoder(d_model, N, heads, max_seq_len)
        self.decoder = TransformerDecoder(trg_vocab, d_model, N, heads,
                                          max_seq_len)

        # This operation convert output of Decoder to a vector which have the
        # same size with one-hot vector
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        # Output from Encoder
        e_outputs = self.encoder(src, src_mask)

        # Decoder
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output
