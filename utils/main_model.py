# --------------------------------------------------------------------------------
# 		Import
# --------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torchvision
from utils.transformer import Transformer
from utils.cnn_models import vgg16


# --------------------------------------------------------------------------------
#       Classes
# --------------------------------------------------------------------------------
class MainModel(nn.Module):
    def __init__(self, trg_vocab, d_model, N, heads, dropout=0.1):
        super(MainModel, self).__init__()

        self.cnn_model = vgg16()
        self.transformer = Transformer(trg_vocab, d_model, N, heads)

    def forward(self, img, trg, src_mask, trg_mask):
        """ Input:
        img is origin image
        """
        src = self.cnn_model(img)
        output = self.transformer(src, trg, src_mask, trg_mask)
        return output
