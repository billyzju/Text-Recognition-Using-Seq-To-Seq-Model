# --------------------------------------------------------------------------------
# 		Import
# --------------------------------------------------------------------------------
import numpy as np


# --------------------------------------------------------------------------------
#       Funcs
# --------------------------------------------------------------------------------
def translate(matrix, predict_target, path_dict_char):
    """ Translate one-hot vector to character
    """
    print(predict_target.size())
    with open(path_dict_char, "r") as f:
        dict_char = f.readlines()
        dict_p_char = []
        for i in dict_char:
            dict_p_char.append(i[0:-1])

    for i in range(1):
        index_char = matrix[i, :].max(1)[1]
        s = "The line is   "
        for j in index_char:
            s = s + dict_p_char[j]
    print(s)

    predict_target = predict_target[:100]
    s = "The target is "

    for j in predict_target:
        s = s + dict_p_char[int(j)]
    print(s)

def accuracy_char():
    pass

def accuracy_seq():
    pass
