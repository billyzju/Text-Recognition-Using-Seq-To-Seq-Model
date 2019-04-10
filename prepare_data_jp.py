# --------------------------------------------------------------------------------
# 		Import
# --------------------------------------------------------------------------------
import os
import numpy as np
import argparse
import json

# --------------------------------------------------------------------------------
#       Parser
# --------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='preprocessing data')

parser.add_argument('--preprocess_label', type=str, default='True',
                    help='Create files labels for lines, words and dictionary')

parser.add_argument('--preprocess_image', type=str, default='False',
                    help='Create files images path for lines, words and\
                          dictionary')

args = parser.parse_args()


# --------------------------------------------------------------------------------
#       Config
# --------------------------------------------------------------------------------
with open("config.json") as json_file:
    config = json.load(json_file)

    # Data
    data = config["data_jp"]
    path_data = data["path_data"]
    path_preprocessing_data = data["path_preprocessing_data"]
    path_labels = data["labels"]


# --------------------------------------------------------------------------------
#       Main
# --------------------------------------------------------------------------------
path_p_dictionary = os.path.join(path_preprocessing_data, "dict_word.txt")

