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
    data = config["data_eng"]
    path_preprocessing_data = data["path_preprocessing_data"]

    lines_train = os.path.join(path_preprocessing_data, "lines_train.txt")
    images_train = os.path.join(path_preprocessing_data, "images_train.txt")

    lines_valid = os.path.join(path_preprocessing_data, "lines_valid.txt")
    images_valid = os.path.join(path_preprocessing_data, "images_valid.txt")

    lines_test = os.path.join(path_preprocessing_data, "lines_test.txt")
    images_test = os.path.join(path_preprocessing_data, "images_test.txt")


# --------------------------------------------------------------------------------
#       Main
# --------------------------------------------------------------------------------
with open(images_test, "r") as f:
    lines = f.readlines()

with open(images_test, "w") as f:
    for i in lines:
        line = i.replace("E:/data/OCR", "../input/iam")
        f.write(line)
