# --------------------------------------------------------------------------------
# 		Import
# --------------------------------------------------------------------------------
import cv2
import torch
import numpy as np
from torch.autograd import Variable


# --------------------------------------------------------------------------------
#       Funcs
# --------------------------------------------------------------------------------
def get_emb(dict_char, line, max_seq_len):
    """ Crate embdding for target from dictionary
    """
    start_char = len(dict_char)
    stop_char = start_char + 1

    label = []
    label.append(start_char)

    for i in range(len(line)):
        index = dict_char.index(line[i])
        label.append(index)

    label.append(stop_char)

    # Padding label
    pad_char = len(dict_char) - 1
    line_pad = [pad_char] * max_seq_len
    pad = (max_seq_len - len(label)) // 2
    line_pad[pad:(pad + len(label))] = label[:]

    return line_pad


def subsequent_mask(size):
    """ Mask out subsequent positions
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def create_mask(target, pad=79):
    """ Get mask for target
    """
    target_mask = (target != pad).unsqueeze(1)
    target_mask = target_mask & Variable(
                    subsequent_mask(target.size(-1)).type_as(target_mask.data))
    return target_mask


def binarize(img, threshold):
    return cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)[1]


def scale(img, max_h):
    w = np.shape(img)[1]
    img = cv2.resize(img, (w, max_h))
    return img


def pad(img, expected_size):
    w = np.shape(img)[1]
    padding = np.ones((np.shape(img)[0], 
                       expected_size,
                       np.shape(img)[2])) * 255
    pad_size = (expected_size - w) // 2
    padding[:, pad_size:(pad_size + w), :] = img[:, :, :]
    return padding
