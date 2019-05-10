# --------------------------------------------------------------------------------
# 		Import
# --------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from utils.language_models.transformer import Transformer
from utils.data_processing import subsequent_mask
from utils.backbones.vgg16 import vgg16


# --------------------------------------------------------------------------------
#       Classes
# --------------------------------------------------------------------------------
class MainModel(nn.Module):
    def __init__(self, trg_vocab, d_model, N, heads, dropout=0.1,
                 max_seq_len=100):
        super(MainModel, self).__init__()

        self.cnn_model = vgg16()
        self.transformer = Transformer(trg_vocab, d_model, N, heads,
                                       max_seq_len)

    def forward(self, img, trg, trg_mask):
        src = self.cnn_model(img)
        src_mask = Variable(subsequent_mask(src.size(1)).
                            type_as(src))
        output = self.transformer(src, trg, src_mask, trg_mask)
        return output
