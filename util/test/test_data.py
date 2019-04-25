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

# # Get the words are out of training dictionary and correspoding images
# with open(images_valid, "r") as f:
#     images = f.readlines()

# with open(lines_valid, "r", encoding="utf8") as f:
#     lines_in = []
#     images_in = []
#     lines = f.readlines()
#     count = 0
#     for line in lines:
#         count_out = 0
#         for i in range(len(line) - 1):
#             if line[i] not in dict_train:
#                 count_out += 1

#         if count_out == 0:
#             lines_in.append(line)
#             images_in.append(images[count])
#         count += 1

# with open(lines_valid, "w", encoding="utf8") as f:
#     for line in lines_in:
#         f.write(line)

# with open(images_valid, "w", encoding="utf8") as f:
#     for image in images_in:
#         image = image.split('/')[-1]
#         f.write("E:/data/OCR/synthesis04/data/" + image)

# # Get the words are out of training dictionary and correspoding images
# with open(images_test, "r") as f:
#     images = f.readlines()

# with open(lines_test, "r", encoding="utf8") as f:
#     lines_in = []
#     images_in = []
#     lines = f.readlines()
#     count = 0
#     for line in lines:
#         count_out = 0
#         for i in range(len(line) - 1):
#             if line[i] not in dict_train:
#                 count_out += 1

#         if count_out == 0:
#             lines_in.append(line)
#             images_in.append(images[count])
#         count += 1

# with open(lines_test, "w", encoding="utf8") as f:
#     for line in lines_in:
#         f.write(line)

# with open(images_test, "w", encoding="utf8") as f:
#     for image in images_in:
#         image = image.split('/')[-1]
#         f.write("E:/data/OCR/synthesis04/data/" + image)

with open("E:\data\OCR\synthesis04/test/Form123_bank_branch_ocr_Val_label_new100Fuku.txt", "r", encoding="utf8") as f:
    images = []
    labels = []
    lines = f.readlines()
    for line in lines:
        image = line.split('|')[0]
        label = line.split('|')[1]
        count_out = 0
        for i in range(len(label) - 1):
            if label[i] not in dict_train:
                count_out += 1
            if len(label) > 18:
                count_out += 1

        if count_out == 0:
            images.append(image)
            labels.append(label)

with open("preprocessing_data/jp/cinnamon_test_labels.txt", "w", encoding="utf8") as f:
    for label in labels:
        f.write(label)

with open("preprocessing_data/jp/cinnamon_test_images.txt", "w", encoding="utf8") as f:
    for image in images:
        f.write("E:/data/OCR/synthesis04/test/" + image + '\n')
