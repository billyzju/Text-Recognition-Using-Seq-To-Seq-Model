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
from utils.data_processing import pad, get_emb


# --------------------------------------------------------------------------------
#       Classes
# --------------------------------------------------------------------------------
class IAMDataset(Dataset):
    def __init__(self, path_p_lines, full_path_images, path_dict_char,
                 max_seq_len, expected_size=None):

        # Get information of images and labels
        with open(path_p_lines, "r") as f:
            lines = f.readlines()

        # Full path of all images
        with open(full_path_images, "r") as f:
            path_images = f.readlines()

        # Get characters in dictionary
        with open(path_dict_char, "r") as f:
            dict_char = f.readlines()
            dict_p_char = []
            for i in dict_char:
                dict_p_char.append(i[0:-1])

        self.images = []
        self.labels = []

        self.expected_size = expected_size
        num_lines = np.shape(lines)[0]

        for i in range(num_lines):
            if lines[i].split(" ")[0] ==\
               os.path.split(path_images[i])[1].split(".")[0]:
                # Corresponding label line
                line = lines[i].split(" ")[-1][0:-1]
                if len(line) < 25:
                    # Get image
                    self.images.append(path_images[i][0:-1])

                    # Get embedding of label
                    label = get_emb(dict_p_char, line, max_seq_len)
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """ Read images and labels
        """
        label = self.labels[idx]
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize image
        image = image / 255

        # Preprocessing image
        image = pad(image, (500, 32))

        # Transforms
        image = torch.tensor(image, dtype=torch.float32)

        # Preprocessing label
        label = torch.tensor(label, dtype=torch.float32)

        return image, label


class IAMDataLoader(object):
    def __init__(self, batch_size, shuffle, path_p_lines, full_path_images,
                 path_dict_char, max_seq_len, expected_size):
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.path_p_lines = path_p_lines
        self.full_path_images = full_path_images
        self.path_dict_char = path_dict_char
        self.max_seq_len = max_seq_len
        self.expected_size = expected_size
        self.dataset = IAMDataset(self.path_p_lines, self.full_path_images,
                                  self.path_dict_char, self.max_seq_len,
                                  self.expected_size)

    def loader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size,
                          shuffle=self.shuffle)
