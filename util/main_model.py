# --------------------------------------------------------------------------------
# 		Import
# --------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torchvision
from util.transformer import Transformer
from util.cnn_models import vgg16


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

    def forward(self, img, trg, src_mask, trg_mask):
        src = self.cnn_model(img)
        output = self.transformer(src, trg, src_mask, trg_mask)
        return output
