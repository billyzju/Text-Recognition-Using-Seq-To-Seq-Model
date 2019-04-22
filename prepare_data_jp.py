# --------------------------------------------------------------------------------
# 		Import
# --------------------------------------------------------------------------------
import os
import numpy as np
import argparse
import json
import codecs
import sys


# --------------------------------------------------------------------------------
#       Config
# --------------------------------------------------------------------------------
with open("config.json") as json_file:
    config = json.load(json_file)

    # Data
    data = config["data_jp"]
    path_data = data["path_data"]
    path_preprocessing_data = data["path_preprocessing_data"]
    path_data_images = data["path_data_images"]
    path_labels = data["path_labels"]


# --------------------------------------------------------------------------------
#       Main
# --------------------------------------------------------------------------------
path_p_dictionary = os.path.join(path_preprocessing_data, "dict_word.txt")
path_data_images = os.path.join(path_data, path_data_images)
path_labels_file = os.path.join(path_data, path_labels)

with open(path_labels_file, "r", encoding="utf8") as f:
    infor = json.load(f)
    path_images = []
    labels = []
    for i in infor:
        path_img = os.path.join(path_data_images, i)
        path_images.append(path_img)
        labels.append(infor[i])

    num_images = len(path_images)
    num_images_train = round(num_images * 0.8)
    num_images_valid = round(num_images * 0.9)

    index = np.arange(num_images)
    np.random.seed(0)
    np.random.shuffle(index)

    index_train = index[:num_images_train]
    index_valid = index[num_images_train:num_images_valid]
    index_test = index[num_images_valid:]

    path_images_train = []
    labels_train = []

    path_images_valid = []
    labels_valid = []

    path_images_test = []
    labels_test = []

    for i in index_train:
        path_images_train.append(path_images[i])
        labels_train.append(labels[i])

    for i in index_valid:
        path_images_valid.append(path_images[i])
        labels_valid.append(labels[i])

    for i in index_test:
        path_images_test.append(path_images[i])
        labels_test.append(labels[i])

# Write path of images to file text
with open(os.path.join(path_preprocessing_data, "images_train.txt"), "w") as f:
    for i in path_images_train:
        f.write(i + "\n")

with open(os.path.join(path_preprocessing_data, "images_valid.txt"), "w") as f:
    for i in path_images_valid:
        f.write(i + "\n")

with open(os.path.join(path_preprocessing_data, "images_test.txt"), "w") as f:
    for i in path_images_test:
        f.write(i + "\n")

# Write labels target to file text
with open(os.path.join(path_preprocessing_data, "lines_train.txt"),
          "w", encoding="utf8") as f:
    for i in labels_train:
        f.write(i + "\n")

with open(os.path.join(path_preprocessing_data, "lines_valid.txt"),
          "w", encoding="utf8") as f:
    for i in labels_valid:
        f.write(i + "\n")

with open(os.path.join(path_preprocessing_data, "lines_test.txt"),
          "w", encoding="utf8") as f:
    for i in labels_test:
        f.write(i + "\n")

dictionary = []
with open(path_p_dictionary, "w", encoding="utf8") as f:
    for line in labels:
        for j in range(len(line)):
            if line[j] not in dictionary:
                dictionary.append(line[j])
                f.write(line[j] + "\n")
    # Padding character
    f.write('|' + '\n')
