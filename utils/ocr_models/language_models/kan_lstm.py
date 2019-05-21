import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import *

class Model(nn.Module):
    def __init__(self, n_vocab):
        super(Model, self).__init__()

        # Encoder
        self.Encoder = Encoder(dropout_rate=0.2)
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((1, None))

        # MultiLayer LSTM
        channel_out = 288
        self.MultiLSTM = MultiLSTM(in_dim=channel_out, hid_dim=512, bidirectional=True, n_stack=1, dropout=0.2)

        # Attention
        self.prediction = Attention(input_size=512, hidden_size=256, num_classes=n_vocab, embed_dim=128)

    def process_label(self, labels, labels_len, EOS_TOKEN, SOS_TOKEN, batch_max_length):
        """
        :param labels: list of text
        :param labels_len: list of int
        :return:
        """
        labels = torch.LongTensor(labels)
        batch_max_length += 1

        batch_labels = torch.LongTensor(len(labels), batch_max_length + 1).fill_(SOS_TOKEN)
        for i, (lbl, lbl_len) in enumerate(zip(labels, labels_len)):
            batch_labels[i][1:1 + lbl_len] = lbl[:lbl_len]
            batch_labels[i][1 + lbl_len] = EOS_TOKEN

        return batch_labels

    def forward(self, src, text, is_train, batch_max_length = 25):
        # (w, b, c)
        memory = self._encode(src).transpose(1,0)

        # (b, batch_max_length + 1, n_vocab)
        probs = self._decode(text, memory, is_train, batch_max_length)

        return probs

    def _encode(self, src):
        # (b, c, h, w)
        output = self.Encoder(src)

        # force h to 1, <=> (b, c, 1, w)
        output = self.AdaptiveAvgPool(output)

        # resize <=> (b, w, c)
        output = output.squeeze(2).transpose(2,1)

        return output

    def _decode(self, text, memory, is_train, batch_max_length):
        # (w, b, 512)
        memory = self.MultiLSTM(memory)

        # (b, w, 512)
        memory = memory.transpose(1, 0)

        # (b, batch_max_length + 1, n_vocab)
        probs = self.prediction(memory, text, is_train, batch_max_length)

        return probs

class Encoder(nn.Module):
    def __init__(self, dropout_rate = 0.2):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=(2,2), padding=2, bias=False)

        self.dense_block1 = _DenseBlock(nb_layers=4, nb_filter=64, growth_rate=16, dropout_rate=dropout_rate)
        self.trans_block1 = _TransitionBlock(nb_in_filter=64 + 16 * 4, nb_out_filter=128)

        self.dense_block2 = _DenseBlock(nb_layers=6, nb_filter=128, growth_rate=16, dropout_rate=dropout_rate)
        self.trans_block2 = _TransitionBlock(nb_in_filter=128 + 16*6, nb_out_filter=224)

        self.dense_block3 = _DenseBlock(nb_layers=4, nb_filter=224, growth_rate=16, dropout_rate=dropout_rate)

        self.batch_norm4 = nn.BatchNorm2d(288)

    def forward(self, src): # (b, c, h, w)
        batch_size = src.size(0)

        out = self.conv1(src[:, :, :, :] - 0.5)

        out = self.dense_block1(out)
        out = self.trans_block1(out)

        out = self.dense_block2(out)
        out = self.trans_block2(out)

        out = self.dense_block3(out)

        src = F.relu(self.batch_norm4(out), inplace=True)

        return src

class _ConvBlock(nn.Sequential):
    def __init__(self, input_channel, growth_rate, dropout_rate=0.2):
        super(_ConvBlock, self).__init__()

        self.add_module('norm1_1', nn.BatchNorm2d(input_channel)),
        self.add_module('relu2_1', nn.ReLU(inplace=True)),
        self.add_module('conv2_1', nn.Conv2d(input_channel, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False))

        self.dropout_rate = dropout_rate

    def forward(self, x):
        new_features = super(_ConvBlock, self).forward(x)
        if self.dropout_rate > 0:
            new_features = F.dropout(new_features, p=self.dropout_rate, training=self.training)

        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Sequential):
    def __init__(self, nb_layers, nb_filter, growth_rate, dropout_rate=0.2, weight_decay=1e-4):
        super(_DenseBlock, self).__init__()

        for i in range(nb_layers):
            layer = _ConvBlock(nb_filter + i * growth_rate, growth_rate, dropout_rate)
            self.add_module('conv_block%d' % (i + 1), layer)

class _TransitionBlock(nn.Sequential):
    def __init__(self, nb_in_filter, nb_out_filter, dropout_rate=None):
        super(_TransitionBlock, self).__init__()

        self.add_module('norm', nn.BatchNorm2d(nb_in_filter))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(nb_in_filter, nb_out_filter, kernel_size=1, stride=1, bias=False))

        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

import copy
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SingleLSTM(nn.Module):
    def __init__(self, hid_dim, bidirectional,dropout=0.1):
        super(SingleLSTM, self).__init__()

        n_direction = 1 if bidirectional is False else 2

        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(int(hid_dim), int(hid_dim // n_direction), num_layers=1, bidirectional=bidirectional)
        self.layer_norm = LayerNorm(hid_dim)

    def forward(self, input):
        rnn_output, _ = self.lstm(input)

        return self.layer_norm(input + self.dropout(rnn_output))

class MultiLSTM(nn.Module):
    def __init__(self, in_dim, hid_dim, bidirectional, n_stack=3, dropout=0.1):
        super(MultiLSTM, self).__init__()

        lstm = SingleLSTM(hid_dim,bidirectional,dropout)

        self.embedding = nn.Linear(in_dim, hid_dim)
        self.embedding_norm = LayerNorm(hid_dim)

        self.lstm_layers = clones(lstm, n_stack)

    def forward(self, input):
        output = self.embedding(input)
        output = self.embedding_norm(output)

        for layer in self.lstm_layers:
            output = layer(output)

        return output


class Attention(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes, embed_dim):
        super(Attention, self).__init__()
        self.attention_cell = AttentionCell(input_size, hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.generator = nn.Linear(hidden_size, num_classes)

    def _char_to_onehot(self, input_char, onehot_dim=38):
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        one_hot = torch.cuda.FloatTensor(batch_size, onehot_dim).zero_()
        one_hot = one_hot.scatter_(1, input_char, 1)
        return one_hot

    def forward(self, batch_H, text, is_train=True, batch_max_length=25):
        """
        input:
            batch_H : contextual_feature H = hidden state of encoder. [batch_size x num_steps x num_classes]
            text : the text-index of each image. [batch_size x (max_length+1)]. +1 for [GO] token. text[:, 0] = [GO].
        output: probability distribution at each step [batch_size x num_steps x num_classes]
        """
        batch_size = batch_H.size(0)
        num_steps = batch_max_length + 1  # +1 for [s] at end of sentence.

        output_hiddens = torch.cuda.FloatTensor(batch_size, num_steps, self.hidden_size).fill_(0)
        hidden = (torch.cuda.FloatTensor(batch_size, self.hidden_size).fill_(0),
                  torch.cuda.FloatTensor(batch_size, self.hidden_size).fill_(0))

        if is_train:
            for i in range(num_steps):
                # one-hot vectors for a i-th char. in a batch
                char_onehots = self._char_to_onehot(text[:, i], self.num_classes)
                # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_onehots : one-hot(y_{t-1})
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                output_hiddens[:, i, :] = hidden[0]  # LSTM hidden index (0: hidden, 1: Cell)
            probs = self.generator(output_hiddens)

        else:
            targets = torch.cuda.LongTensor(batch_size).fill_(SOS_token)  # [GO] token
            probs = torch.cuda.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0)

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, self.num_classes)
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                probs_step = self.generator(hidden[0])
                probs[:, i, :] = probs_step
                _, next_input = probs_step.max(1)
                targets = next_input

        return probs  # batch_size x num_steps x num_classes


class AttentionCell(nn.Module):

    def __init__(self, input_size, hidden_size, num_embeddings):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)  # either i2i or h2h should have bias
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
        e = self.score(torch.tanh(batch_H_proj + prev_hidden_proj))  # batch_size x num_encoder_step * 1

        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(1)  # batch_size x num_channel
        concat_context = torch.cat([context, char_onehots], 1)  # batch_size x (num_channel + num_embedding)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha

