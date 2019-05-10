# --------------------------------------------------------------------------------
# 		Import
# --------------------------------------------------------------------------------
import numpy as np
import torch.nn.functional as F


# --------------------------------------------------------------------------------
#       Funcs
# --------------------------------------------------------------------------------
def translate(output, predict_target, path_dict_char):
    """
    Translate one-hot vector to character
    """
    with open(path_dict_char, "r", encoding="utf8") as f:
        dict_char = f.readlines()
        dict_p_char = []
        for i in dict_char:
            dict_p_char.append(i[0:-1])
    # Softmax
    output = F.softmax(output, dim=-1)
    index_char = output.max(1)[1]
    index_char = index_char[:57]
    s = "The line is   "
    predict = ""
    for i in index_char:
        if i >= len(dict_p_char):
            s = s + '*'
        else:
            s = s + dict_p_char[int(i)]
            predict = predict + dict_p_char[int(i)]
    print(s)

    predict_target = predict_target[:57]
    s = "The target is "
    target= ""
    for j in predict_target:
        if j >= len(dict_p_char):
            s = s + '*'
        else:
            s = s + dict_p_char[int(j)]
            target = target + dict_p_char[int(j)]
    print(s)
    return predict, target


def accuracy_char_1(output, predict_target):
    output = F.softmax(output, dim=-1)
    index_char = output.max(1)[1]
    acc = 0
    for i in range(index_char.size(0)):

        if index_char[i] == predict_target[i]:
            acc = acc + 1

    return acc / index_char.size(0)


def accuracy_char_2(output, predict_target):
    """
    Calculating accuracy on characters of origin label
    """
    output = F.softmax(output, dim=-1)
    index_char = output.max(2)[1]
    acc = 0
    n_char = 0
    for i in range(index_char.size(0)):
        for j in range(index_char.size(1)):
            if predict_target[i, j] == 79:
                break
            n_char += 1
            if index_char[i, j] == predict_target[i, j]:
                acc = acc + 1

    return acc / n_char


def accuracy_word(output, predict_target):
    """ Calculating accuracy on words
    """
    output = F.softmax(output, dim=-1)
    index_char = output.max(2)[1]
    acc = 0
    # Get 1 element in batch size
    for i in range(index_char.size(0)):
        n_char = 0
        for j in range(index_char.size(1)):
            if index_char[i, j] == predict_target[i, j]:
                n_char += 1
        # If number of correct characters is equal with
        # max sequence length
        if n_char == index_char.size(1):
            acc += 1

    return acc / index_char.size(0)
