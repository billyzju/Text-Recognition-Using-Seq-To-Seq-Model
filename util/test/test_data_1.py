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
    path_preprocessing_data = data["path_preprocessing_data"]

    lines_train = os.path.join(path_preprocessing_data, "lines_train.txt")

    lines_valid = os.path.join(path_preprocessing_data, "lines_valid.txt")
    images_valid = os.path.join(path_preprocessing_data, "images_valid.txt")

    lines_test = os.path.join(path_preprocessing_data, "lines_test.txt")
    images_test = os.path.join(path_preprocessing_data, "images_test.txt")


# --------------------------------------------------------------------------------
#       Main
# --------------------------------------------------------------------------------
# Get training dictionary
with open(lines_train, "r", encoding="utf8") as f:
    dict_train = []
    lines = f.readlines()
    for line in lines:
        for i in range(len(line) - 1):
            if line[i] not in dict_train:
                dict_train.append(line[i])

# Get the words are out of training dictionary and correspoding images
with open(images_valid, "r") as f:
    images = f.readlines()

with open(lines_valid, "r", encoding="utf8") as f:
    words_out = []
    images_out = []
    lines = f.readlines()
    count = 0
    for line in lines:
        for i in range(len(line) - 1):
            if line[i] not in dict_train:
                words_out.append(line[i])
                images_out.append(images[count])
        count += 1

print("The words are out of training dictionary", words_out)
print("The images have above words", images_out)

# Get the words are out of training dictionary and correspoding images
with open(images_test, "r") as f:
    images = f.readlines()

with open(lines_test, "r", encoding="utf8") as f:
    words_out = []
    images_out = []
    lines = f.readlines()
    count = 0
    for line in lines:
        for i in range(len(line) - 1):
            if line[i] not in dict_train:
                words_out.append(line[i])
                images_out.append(images[count])
        count += 1

print("The words are out of training dictionary", words_out)
print("The images have above words", images_out)
