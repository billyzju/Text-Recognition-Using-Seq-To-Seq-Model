# --------------------------------------------------------------------------------
# 		Import
# --------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from utils.ocr_models.language_models.lstm import LSTM
from utils.ocr_models.backbones.vgg16 import vgg16


# --------------------------------------------------------------------------------
#       Classes
# --------------------------------------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 num_layer, bidirectional, vocab_size):
        super(LSTMModel, self).__init__()

        self.cnn_model = vgg16()
        self.lstm = LSTM(input_dim, hidden_dim,
                         num_layer, bidirectional, vocab_size)

    def forward(self, src, trg):
        src = self.cnn_model(src)
        output = self.lstm(src, trg)
        return output
