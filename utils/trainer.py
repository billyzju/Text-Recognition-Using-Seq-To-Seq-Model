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
import torch.nn.functional as F
from utils.main_model import MainModel
from utils.data_loader import IAMDataLoader
import matplotlib.pyplot as plt
from utils.data_processing import create_mask
from utils.metrics import translate, accuracy_char
from utils.logger import Logger


# --------------------------------------------------------------------------------
# 		Class
# --------------------------------------------------------------------------------
class Trainer:
    def __init__(self, *args, **kwargs):
        super(Trainer).__init__()