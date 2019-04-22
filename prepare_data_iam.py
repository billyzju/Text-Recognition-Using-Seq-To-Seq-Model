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
    data = config["data"]
    path_data = data["path_data"]
    path_preprocessing_data = data["path_preprocessing_data"]

    # Label
    path_label_forms = data["path_label_forms"]
    path_label_lines = data["path_label_lines"]
    path_label_sentences = data["path_label_sentences"]
    path_label_words = data["path_label_words"]

    # Images
    path_forms = data["path_forms"]
    path_lines = data["path_lines"]
    path_sentences = data["path_sentences"]
    path_words = data["path_words"]


# --------------------------------------------------------------------------------
#       Main
# --------------------------------------------------------------------------------
path_p_lines = os.path.join(path_preprocessing_data, path_label_lines)
path_p_words = os.path.join(path_preprocessing_data, path_label_words)
path_p_dictionary = os.path.join(path_preprocessing_data, "dict_word.txt")
path_p_dict_char = os.path.join(path_preprocessing_data, "dict_char.txt")

if args.preprocess_label == "True":

    # Get raw label for lines
    with open(os.path.join(path_data, path_label_lines), "r") as f:
        lines = f.readlines()

    # Remove headlines in lines file
    with open(path_p_lines, "w") as f:
        for i in lines:
            if i[0] != "#":
                f.write(i)

    # Get raw label for words
    with open(os.path.join(path_data, path_label_words), "r") as f:
        words = f.readlines()

    # Remove headlines in words file
    with open(path_p_words, "w") as f:
        preprocessing_words = []
        for i in words:
            if i[0] != "#":
                f.write(i.split()[-1] + "\n")
                preprocessing_words.append(i.split()[-1])

    # Get dictionary of words
    with open(path_p_dictionary, "w") as f:
        dictionary = []
        for word in preprocessing_words:
            if word not in dictionary:
                dictionary.append(word)
                f.write(word + "\n")

    # Get dictionary of characters
    with open(path_p_dict_char, "w") as f:
        dict_char = []
        for word in dictionary:
            for i in range(len(word)):
                if word[i] not in dict_char:
                    dict_char.append(word[i])
                    f.write(word[i] + '\n')
        # Add character for space
        f.write('|' + '\n')
        # Add character for padding
        f.write(' ' + '\n')


path_data_lines = os.path.join(path_data, path_lines)
path_images = os.path.join(path_preprocessing_data, "path_images.txt")

if args.preprocess_image == "True":

    full_path_images_3 = []
    # Get dir of all images
    for f in os.listdir(path_data_lines):
        full_path_images_1 = os.path.join(path_data_lines, f)
        for i in os.listdir(full_path_images_1):
            full_path_images_2 = os.path.join(full_path_images_1, i)
            for j in os.listdir(full_path_images_2):
                full_path_images_3.append(os.path.join(full_path_images_2, j))

    # Write all path to file
    with open(path_images, "w") as f:
        for i in full_path_images_3:
            f.write(i + "\n")


# Split data for training and validating
with open(path_p_lines) as f:
    lines = f.readlines()

with open(path_images) as f:
    path_imgs = f.readlines()

num_images = len(lines)
num_images_train = round(num_images * 0.8)

index = np.arange(num_images)
np.random.seed(0)
np.random.shuffle(index)
index_train = index[:num_images_train]
index_valid = index[num_images_train:]

path_images_train = []
lines_train = []
path_images_valid = []
lines_valid = []

for i in index_train:
    path_images_train.append(path_imgs[i])
    lines_train.append(lines[i])

for i in index_valid:
    path_images_valid.append(path_imgs[i])
    lines_valid.append(lines[i])

with open(os.path.join(path_preprocessing_data, "images_train.txt"), "w") as f:
    for i in path_images_train:
        f.write(i)

with open(os.path.join(path_preprocessing_data, "images_valid.txt"), "w") as f:
    for i in path_images_valid:
        f.write(i)

with open(os.path.join(path_preprocessing_data, "lines_train.txt"), "w") as f:
    for i in lines_train:
        f.write(i)

with open(os.path.join(path_preprocessing_data, "lines_valid.txt"), "w") as f:
    for i in lines_valid:
        f.write(i)
