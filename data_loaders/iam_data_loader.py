# --------------------------------------------------------------------------------
# 		Import
# --------------------------------------------------------------------------------
import cv2
import random
from PIL import Image
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor
from utils.data_processing import pad, get_emb


# --------------------------------------------------------------------------------
#       Classes
# --------------------------------------------------------------------------------
class IAMDataset(Dataset):
    def __init__(self, config, labels, path_images, dictionary, max_len):

        # Get information of labels
        with open(labels, "r") as f:
            labels = f.readlines()

        # Path of all images
        with open(path_images, "r") as f:
            path_images = f.readlines()

        # Get characters in dictionary
        with open(dictionary, "r") as f:
            dict_char = f.readlines()
            dict_p_char = []
            for i in dict_char:
                dict_p_char.append(i.strip())

        self.images = []
        self.labels = []
        self.path_to_data = "E:/data/OCR/iam/"
        num_images = np.shape(labels)[0]

        # Pre-processing target labels before data loader
        for i in range(num_images):
            if labels[i].split(" ")[0] ==\
               os.path.split(path_images[i])[1].split(".")[0]:
                # Corresponding label line
                label = labels[i].split(" ")[-1].strip()
                if len(label) < max_len:
                    # Get image path
                    self.images.append(path_images[i].strip())
                    # Get index embedding of label
                    label = get_emb(dict_p_char, label, max_len)
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        :param idx: index of item
        :return:
        image:
        label:
        """
        label = self.labels[idx]
        # Full path image
        path_image = os.path.join(self.path_to_data, self.images[idx])
        image = cv2.imread(path_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255
        # Convert to tensor
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return image, label


class IAMDataLoader(object):
    def __init__(self, config, shuffle, labels, path_images,
                 dictionary, max_len):
        self.batch_size = config["trainer"]["batch_size"]
        self.shuffle = shuffle
        self.dataset = IAMDataset(config, labels, path_images,
                                  dictionary, max_len)

    def loader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size,
                          shuffle=self.shuffle, collate_fn=collate_fnc)


class OCR_collate:
    def __init__(self, enable_augment=False):
        self.enable_augment = enable_augment

    def __call__(self, batchs):
        images, labels = zip(*batchs)
        images = pad_batch_image_tensor(images)
        images = images.permute(0, 3, 1, 2)
        labels = [list(label) for _, label in enumerate(labels)]
        labels = torch.tensor(labels, dtype=torch.float32)
        return images, labels


def pad_batch_image_tensor(tensor_images, enable_augment=True):
    """
    :param tensor_images: list(h, w, c)
    :return:
    tensor(n_batch, max_height, max_width, n_channel)
    """
    c = tensor_images[0].size(2)
    h = max([e.size(0) for e in tensor_images])
    w = max([e.size(1) for e in tensor_images])
    batch_images = torch.zeros(len(tensor_images), h, w, c).fill_(1)
    for i, image in enumerate(tensor_images):
        started_h = max(0, random.randint(0, h - image.size(0)))
        started_w = max(0, random.randint(0, w - image.size(1)))

        if enable_augment is False:
            started_h = 0
            started_w = 0
        batch_images[i, started_h:started_h + image.size(0),
                     started_w:started_w + image.size(1), :] = image
    return batch_images

collate_fnc = OCR_collate()