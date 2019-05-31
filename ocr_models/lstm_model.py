# --------------------------------------------------------------------------------
# 		Import
# --------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from ocr_models.language_models.lstm import LSTM
from ocr_models.backbones.vgg16 import vgg16
from ocr_models.backbones.onmt_cnn import ONMT


# --------------------------------------------------------------------------------
#       Classes
# --------------------------------------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, enc_hidden_dim,
                 enc_bidirectional, dec_hidden_dim,
                 dec_bidirectional, num_layer, vocab_size):
        super(LSTMModel, self).__init__()

        self.cnn_model = vgg16()
        # self.cnn_model = ONMT()
        self.lstm = LSTM(input_dim, enc_hidden_dim,
                         enc_bidirectional, dec_hidden_dim,
                         dec_bidirectional, num_layer, vocab_size)

    def forward(self, src, trg):
        """
        src: (batch_size, 1, height, width)
        trg: (batch_size, decoder_seq_len)
        """
        src = self.cnn_model(src)
        # Output of CNN model have size of (batch_size, col, emb_dim)
        output = self.lstm(src, trg)
        return output
