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
from utils.data_processing import get_emb, pa


# --------------------------------------------------------------------------------
#       Classes
# --------------------------------------------------------------------------------
class JPDataset(Dataset):
    def __init__(self, path_labels, full_path_images, path_dict_word,
                 max_seq_len, expected_size=None):

        # Get information of labels
        with open(path_labels, "r", encoding="utf8") as f:
            lines = f.readlines()
            labels = []
            for i in lines:
                labels.append(i[0:-1])

        # Full path of all images
        with open(full_path_images, "r", encoding="utf8") as f:
            path_images = f.readlines()
            path_p_images = []
            for i in path_images:
                path_p_images.append(i[0:-1])

        # Get characters in dictionary
        with open(path_dict_word, "r", encoding="utf8") as f:
            dict_word = f.readlines()
            dict_p_word = []
            for i in dict_word:
                dict_p_word.append(i[0:-1])

        self.images = path_p_images
        self.labels = []
        self.expected_size = expected_size
        num_lines = len(self.images)

        for i in range(num_lines):
            label = get_emb(dict_p_word, labels[i], max_seq_len)
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


class JPDataLoader(object):
    def __init__(self, batch_size, shuffle, path_labels, full_path_images,
                 path_dict_word, max_seq_len, expected_size):
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.path_labels = path_labels
        self.full_path_images = full_path_images
        self.path_dict_word = path_dict_word
        self.max_seq_len = max_seq_len
        self.expected_size = expected_size
        self.dataset = JPDataset(self.path_labels, self.full_path_images,
                                 self.path_dict_word, self.max_seq_len,
                                 self.expected_size)

    def loader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size,
                          shuffle=self.shuffle)
