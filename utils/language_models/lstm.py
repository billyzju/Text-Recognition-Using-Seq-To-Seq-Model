# --------------------------------------------------------------------------------
# 		Import
# --------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torchvision
from utils.language_models.sublayers import Embedder


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
        # Src size  = [seq_length, batch_size, embed_length]
        out, hidden = self.lstm(src)
        return out, hidden


class GlobalAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim):
        """
        encoder_dim: dim hidden states Encoder
        decoder_dim: dim of hidden state Decoder
        """
        super(GlobalAttention, self).__init__()

        dim = encoder_dim
        self.linear_in = nn.Linear(encoder_dim, decoder_dim, bias=False)
        self.linear_context = nn.Linear(dim, dim, bias=False)
        self.linear_query = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, 1, bias=False)
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()
        self.dim = dim

    def score(self, h_t, h_s, type_score):
        """
        h_t (FloatTensor): batch x tgt_len x dim, inputs
        h_s (FloatTensor): batch x src_len x dim, context
        There are three different alternatives
        """
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        src_batch, src_len, src_dim = h_s.size()
        dim = self.dim

        if type_score == "dot":
            h_s = h_s.transpose(1, 2)
            score = torch.bmm(h_t, h_s)

        elif type_score == "general":
            h_t = h_t.view(tgt_batch * tgt_len, tgt_dim)
            h_t = self.linear_in(h_t)
            h_t = h_t.view(tgt_batch, tgt_len, tgt_dim)
            h_s = h_s.transpose(1, 2)
            score = torch.bmm(h_t, h_s)

        elif type_score == "concat":
            wq = self.linear_query(h_t.view(-1, dim))
            wq = wq.view(tgt_batch, tgt_len, 1, dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, dim)

            uh = self.linear_context(h_s.contiguous().view(-1, dim))
            uh = uh.view(src_batch, 1, src_len, dim)
            uh = uh.expand(src_batch, tgt_len, src_len, dim)

            # (batch, t_len, s_len, d)
            wquh = torch.tanh(wq + uh)

            score = self.v(wquh.view(-1, dim)).view(
                        tgt_batch,
                        tgt_len,
                        src_len)

        return score

    def forward(self, inputs, context, context_lengths):
        """
        input (FloatTensor): batch x tgt_len x dim: decoder's rnn's output. (h_t)
        context (FloatTensor): batch x src_len x dim: src hidden states
        context_lengths (LongTensor): the source context lengths.
        """
        dim = self.dim

        # (batch, tgt_len, src_len)
        align = self.score(inputs, context)
        batch, tgt_len, src_len = align.size()

        align_vectors = self.softmax(align.view(batch*tgt_len, src_len))
        align_vectors = align_vectors.view(batch, tgt_len, src_len)

        # (batch, tgt_len, src_len) * (batch, src_len, enc_hidden)
        # -> (batch, tgt_len, enc_hidden)
        c = torch.bmm(align_vectors, context)

        # \hat{h_t} = tanh(W [c_t, h_t])
        concat_c = torch.cat([c, inputs], 2).view(
                    batch * tgt_len,
                    2 * dim)
        attn_h = self.tanh(self.linear_out(concat_c).view(
                    batch,
                    tgt_len,
                    dim))

        # Transpose will make it non-contiguous
        attn_h = attn_h.transpose(0, 1).contiguous()
        align_vectors = align_vectors.transpose(0, 1).contiguous()
        # (tgt_len, batch, dim)
        return attn_h, align_vectors


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, input_dim, tgt_hidden_dim, src_hidden_dim,
                 num_layer, bidirectional, dropout=0.2):
        super(LSTMDecoder, self).__init__()

        self.embed = Embedder(vocab_size, input_dim)
        # Define LSTM layer
        self.lstm = nn.LSTM(input_dim, tgt_hidden_dim, num_layer,
                            bidirectional=bidirectional)
        self.linear_out = nn.Linear(tgt_hidden_dim, vocab_size)
        self.attn = GlobalAttention(src_hidden_dim, tgt_hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs, init_hidden, context, context_lengths):
        """
        inputs: (tgt_len, batch_size, d)
        hidden: last hidden state from encoder
        context: (src_len, batch_size, hidden_size), outputs of encoder
        """
        emb = self.embed(inputs)
        emb = self.drop(emb)
        decoder_unpacked, decoder_hidden = self.lstm(inputs, init_hidden)
        attn_outputs, attn_scores = self.attn(
            decoder_unpacked.transpose(0, 1).contiguous(), # (len, batch, d) -> (batch, len, d)
            context.transpose(0, 1).contiguous(), # (len, batch, d) -> (batch, len, d)
            context_lengths=context_lengths
        )
        outputs = self.linear_out(attn_outputs)
        return outputs, decoder_hidden


class LSTM(nn.Module):
    def __init__(self, input_dim, src_hidden_dim, tgt_hidden_dim,
                 num_layer, bidirectional, vocab_size):
        super(LSTM, self).__init__()

        self.encoder = LSTMEncoder(input_dim, hidden_dim,
                                   num_layer, bidirectional)
        self.decoder = LSTMDecoder(vocab_size, input_dim, tgt_hidden_dim,
                                   src_hidden_dim, num_layer,
                                   bidirectional)

    def forward(self, src, trg, init_hidden, context_lengths):
        context, hidden_encoder = self.encoder(src)
        self.src_hidden_dim = hidden_encoder.size()
        output, _ = self.decoder(trg, hidden_encoder, context, context_lengths)

        return output
