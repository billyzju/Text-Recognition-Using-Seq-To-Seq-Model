# --------------------------------------------------------------------------------
# 		Import
# --------------------------------------------------------------------------------
import numpy as np
import torch.nn.functional as F


# --------------------------------------------------------------------------------
#       Funcs
# --------------------------------------------------------------------------------
def translate(output, predict_target, path_dict_char):
    """ Translate one-hot vector to character
    """
    with open(path_dict_char, "r") as f:
        dict_char = f.readlines()
        dict_p_char = []
        for i in dict_char:
            dict_p_char.append(i[0:-1])
    # Softmax
    output = F.softmax(output, dim=-1)
    index_char = output.max(1)[1]
    index_char = index_char[:99]
    s = "The line is   "
    for i in index_char:
        if i >= len(dict_p_char):
            s = s + '|'
        else:
            s = s + dict_p_char[int(i)]
    print(s)

    predict_target = predict_target[:99]
    s = "The target is "
    for j in predict_target:
        if j >= len(dict_p_char):
            s = s + '|'
        else:
            s = s + dict_p_char[int(j)]
    print(s)


def accuracy_char(output, predict_target):
    output = F.softmax(output, dim=-1)
    index_char = output.max(1)[1]
    acc = 0
    for i in range(index_char.size(0)):

        if index_char[i] == predict_target[i]:
            acc = acc + 1

    return acc / index_char.size(0)


def accuracy_seq():
    pass
