# --------------------------------------------------------------------------------
# 		Import
# --------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torchvision
import argparse
import json
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
from utils.data_processing import create_mask, subsequent_mask
from utils.metrics import translate, accuracy_char_1, accuracy_char_2
from utils.metrics import accuracy_word
from utils.logger import Logger
from torchsummary import summary


# --------------------------------------------------------------------------------
# 		Class of base for trainer
# --------------------------------------------------------------------------------
