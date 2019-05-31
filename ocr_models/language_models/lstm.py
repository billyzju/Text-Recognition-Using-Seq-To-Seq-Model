# --------------------------------------------------------------------------------
# 		Import
# --------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from ocr_models.language_models.sublayers import Embedder


# --------------------------------------------------------------------------------
# 		Class
# --------------------------------------------------------------------------------
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer,
                 bidirectional):
        super(LSTMEncoder, self).__init__()

        # Define LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layer,
                            bidirectional=bidirectional)

    def forward(self, src):
        """
        :param src: Output of CNN encoder (batch_size, seq_len, chanel=emb_dim)
        :return:
        output: ()
        hidden_state: ()
        hidden_cell: ()
        """
        all_outputs = []
        for row in range(src.size(2)):
            # Covert row size into [seq_length, batch_size, embed_length]
            inp = src[:, :, row, :].transpose(1, 2).transpose(0, 1)
            output, (hidden_state, hidden_cell) = self.lstm(inp)
            all_outputs.append(output)

        output = torch.cat(all_outputs, 0)
        return output, hidden_state, hidden_cell


class GlobalAttention(nn.Module):
    def __init__(self, dim):
        super(GlobalAttention, self).__init__()
        # Size of encoder and decoder hidden states may be different
        self.dim = dim

        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_out = nn.Linear(2 * dim, dim)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def score(self, h_t, h_s):
        """
        :param h_t: (batch, tgt_len, dim)
        :param h_s: (batch, src_len, dim)
        :return: Attention score
        """
        tgt_batch, tgt_len, dim = h_t.size()
        # h_t^T W_a
        h_t = h_t.view(tgt_batch * tgt_len, dim)
        h_t = self.linear_in(h_t)
        h_t = h_t.view(tgt_batch, tgt_len, dim)
        # (batch, d, s_len)
        h_s = h_s.transpose(1, 2)
        # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
        return torch.bmm(h_t, h_s)

    def forward(self, inputs, context, context_lengths):
        """
        :param inputs: (batch, tgt_len, dim)
        :param context: (batch, src_len, dim)
        :param context_lengths: The source context length
        :return:
        """
        # (batch, tgt_len, src_len)
        align = self.score(inputs, context)
        batch, tgt_len, src_len = align.size()
        align_vectors = self.softmax(align.view(batch * tgt_len, src_len))
        align_vectors = align_vectors.view(batch, tgt_len, src_len)
        # (batch, tgt_len, src_len) * (batch, src_len, enc_hidden)
        # -> (batch, tgt_len, enc_hidden)
        c = torch.bmm(align_vectors, context)

        # \hat{h_t} = tanh(W [c_t, h_t])
        concat_c = torch.cat([c, inputs], 2).view(
                    batch * tgt_len,
                    self.dim * 2)
        attn_h = self.tanh(self.linear_out(concat_c).view(
                    batch,
                    tgt_len,
                    self.dim))

        # transpose will make it non-contiguous
        attn_h = attn_h.transpose(0, 1).contiguous()
        align_vectors = align_vectors.transpose(0, 1).contiguous()
        # (tgt_len, batch, dim)
        return attn_h, align_vectors


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, input_dim, hidden_dim,
                 num_layer, bidirectional, dropout=0.2):
        super(LSTMDecoder, self).__init__()

        self.embed = Embedder(vocab_size, input_dim)
        # Define LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layer,
                            bidirectional=bidirectional)
        if bidirectional:
            hidden_dim = 2 * hidden_dim
        self.linear_out = nn.Linear(hidden_dim, vocab_size)
        self.attn = GlobalAttention(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs, init_hidden_state, init_hidden_cell,
                context, context_lengths):
        """
        inputs: (batch_size, tgt_len) --> (tgt_len, batch_size, d)
        hidden: last hidden state from encoder
        context: (src_len, batch_size, hidden_size), outputs of encoder
        """
        emb = self.embed(inputs)
        emb = self.drop(emb)
        emb = emb.transpose(0, 1)
        decoder_unpacked, decoder_hidden = self.lstm(
                                            emb)
                                            # (init_hidden_state,
                                            #  init_hidden_cell))
        attn_outputs, attn_scores = self.attn(
            # (len, batch, d) -> (batch, len, d)
            decoder_unpacked.transpose(0, 1).contiguous(),
            # (len, batch, d) -> (batch, len, d)
            context.transpose(0, 1).contiguous(),
            context_lengths=context_lengths
        )
        outputs = self.linear_out(attn_outputs)
        return outputs, decoder_hidden


class LSTM(nn.Module):
    def __init__(self, input_dim, enc_hidden_dim,
                 enc_bidirectional, dec_hidden_dim,
                 dec_bidirectional, num_layer, vocab_size):
        super(LSTM, self).__init__()

        self.encoder = LSTMEncoder(input_dim, enc_hidden_dim,
                                   num_layer, enc_bidirectional)
        self.decoder = LSTMDecoder(vocab_size, input_dim, dec_hidden_dim,
                                   num_layer, dec_bidirectional)

    def forward(self, src, trg):
        context, hidden_state, hidden_cell = self.encoder(src)
        output, _ = self.decoder(trg, hidden_state, hidden_cell, context,
                                 context.size(1))

        return output
