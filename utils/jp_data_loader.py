# --------------------------------------------------------------------------------
# 		Import
# --------------------------------------------------------------------------------
import cv2
from PIL import Image
import os
import numpy as np
from random import shuffle
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor
from utils.data_processing import get_emb

# --------------------------------------------------------------------------------
#       Classes
# --------------------------------------------------------------------------------
class JPDataset(Dataset):
    def __init__(self, path_p_lines, full_path_images, path_dict_char,
                 max_seq_len, expected_size=None):