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
def get_emb(dictionary, label, max_len):
    """
    Get index for target from dictionary
    """
    # The index of start character and stop one
    start_char = len(dictionary)
    stop_char = start_char + 1
    label_index= []
    label_index.append(start_char)

    for i in range(len(label)):
        index = dictionary.index(label[i])
        label_index.append(index)

    label_index.append(stop_char)

    # Padding label
    pad_char = len(dictionary) - 1
    line_pad = [pad_char] * max_len
    line_pad[0:len(label)] = label_index[:]

    return line_pad


def subsequent_mask(size):
    """
    Mask out subsequent positions
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def create_mask(target, pad=79):
    """
    Get mask for target
    """
    target_mask = (target != pad).unsqueeze(1)
    target_mask = target_mask & Variable(
                    subsequent_mask(target.size(-1)).type_as(target_mask.data))
    return target_mask


def pad(img, expected_size):
    # Scale
    h_img = np.shape(img)[0]
    w_img = np.shape(img)[1]

    h_expected = expected_size[1]
    w_expected = expected_size[0]

    # Scale width
    scale_rate = w_img / w_expected
    if scale_rate > 1:
        h_img = round(h_img / scale_rate)
        w_img = w_expected
        img = cv2.resize(img, (w_expected, h_img))

    # Scale height
    scale_rate = h_img / h_expected
    w_offset = 0
    if scale_rate > 1:
        w_img_s = round(w_img / scale_rate)
        img_s = cv2.resize(img, (w_img_s, h_expected))[:, :, 0]
        padding = np.ones((h_expected, w_expected))
        padding[:, w_offset:(w_offset + w_img_s)] = img_s[:, :]
    else:
        img = img[:, :, 0]
        padding = np.ones((h_expected, w_expected))
        h_offset = (h_expected - h_img) // 2
        padding[h_offset:(h_offset + h_img), w_offset:(w_offset + w_img)] = img[:, :]

    return [padding]
