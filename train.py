# --------------------------------------------------------------------------------
# 		Import
# --------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torchvision
import argparse
import json
import numpy as np
import time
import os
import torch.nn.functional as F
from utils.main_model import MainModel
from utils.iam_data_loader import IAMDataLoader
from utils.jp_data_loader import JPDataLoader
import matplotlib.pyplot as plt
from utils.data_processing import create_mask
from utils.logger import Logger
from utils.trainer import Trainer


# --------------------------------------------------------------------------------
#       Parser
# --------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Training model')
parser.add_argument('--train_japanese', type=str, default='True',
                    help='Train model for japanese')

parser.add_argument('--train_english', type=str, default='False',
                    help='Train model for english')

parser.add_argument('--train_scratch', type=str, default='False',
                    help='Train model from scratch')

parser.add_argument('--pre_train', type=str, default='True',
                    help='Train model from previous checkpoints')

args = parser.parse_args()


# --------------------------------------------------------------------------------
#       Config for model
# --------------------------------------------------------------------------------
if args.train_english == "True":
    with open("config.json") as json_file:
        config = json.load(json_file)

        # Data
        data = config["data_eng"]
        path_data = data["path_data"]
        path_preprocessing_data = data["path_preprocessing_data"]

        # Label
        path_label_lines = data["path_label_lines"]
        path_label_words = data["path_label_words"]

        # Images
        path_lines = data["path_lines"]
        path_words = data["path_words"]

        # Model
        models = config["models"]
        batch_size = models["batch_size"]
        d_model = models["d_model"]
        heads = models["heads"]
        N = models["N"]
        path_checkpoints = models["checkpoints"]
        path_logs = models["logs"]
        lr = models["learning_rate"]

    # Path to processing data
    path_p_lines = os.path.join(path_preprocessing_data, path_label_lines)
    full_path_images = os.path.join(path_preprocessing_data, "path_images.txt")
    path_dict_char = os.path.join(path_preprocessing_data, "dict_char.txt")

    lines_train = os.path.join(path_preprocessing_data,
                               "lines_train.txt")
    path_images_train = os.path.join(path_preprocessing_data,
                                     "images_train.txt")
    lines_valid = os.path.join(path_preprocessing_data,
                               "lines_valid.txt")
    path_images_valid = os.path.join(path_preprocessing_data,
                                     "images_valid.txt")

    # Target vocab
    with open(os.path.join(path_preprocessing_data, "dict_char.txt")) as f:
        trg_vocab = np.shape(f.readlines())[0] + 2

    # Define model
    if args.train_scratch == "True":
        model = MainModel(trg_vocab, d_model, N, heads)
        model = model.cuda()

        # Init Xavier
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    if args.pre_train == "True":
        model = MainModel(trg_vocab, d_model, N, heads)
        model = model.cuda()
        model.load_state_dict(
            torch.load("checkpoints/eng/17/model_checkpoint_17.pth"))

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr, betas=(0.9, 0.98), eps=1e-9)

    # Logger
    train_logger = Logger("logs")

    # Trainer
    trainer = Trainer(
        model=model, data_loader=IAMDataLoader, optimizer=optimizer,
        train_logger=train_logger, valid_logger=None, max_seq_len=100,
        lines_train=lines_train, path_images_train=path_images_train,
        lines_valid=lines_valid, path_images_valid=path_images_valid,
        path_dict_char=path_dict_char)

if args.train_japanese == "True":
    with open("config.json") as json_file:
        config = json.load(json_file)

        # Data
        data = config["data_jp"]
        path_data = data["path_data"]
        path_preprocessing_data = data["path_preprocessing_data"]
        path_data_images = data["path_data_images"]
        path_labels = data["path_labels"]

        # Model
        models = config["models"]
        batch_size = models["batch_size"]
        d_model = models["d_model"]
        heads = models["heads"]
        N = models["N"]
        path_checkpoints = models["checkpoints"]
        path_logs = models["logs"]
        lr = models["learning_rate"]

    # Path to processing data
    full_path_images = os.path.join(path_preprocessing_data, "path_images.txt")
    path_dict_word = os.path.join(path_preprocessing_data, "dict_word.txt")

    lines_train = os.path.join(path_preprocessing_data,
                               "lines_train.txt")
    path_images_train = os.path.join(path_preprocessing_data,
                                     "images_train.txt")
    lines_valid = os.path.join(path_preprocessing_data,
                               "lines_valid.txt")
    path_images_valid = os.path.join(path_preprocessing_data,
                                     "images_valid.txt")

    # Target vocab
    with open(os.path.join(path_preprocessing_data, "dict_word.txt"),
              encoding="utf8") as f:
        trg_vocab = np.shape(f.readlines())[0] + 2

    # Define model
    if args.train_scratch == "True":
        model = MainModel(trg_vocab, d_model, N, heads)
        model = model.cuda()

        # Init Xavier
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    if args.pre_train == "True":
        model = MainModel(trg_vocab, d_model, N, heads)
        model = model.cuda()
        model.load_state_dict(
            torch.load("checkpoints/jp/12042019/model_checkpoint_12042019.pth"))

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr, betas=(0.9, 0.98), eps=1e-9)

    # Logger
    train_logger = Logger("logs/jp")

    # Trainer
    trainer = Trainer(
        model=model, data_loader=JPDataLoader, optimizer=optimizer,
        train_logger=train_logger, valid_logger=None, max_seq_len=20,
        lines_train=lines_train, path_images_train=path_images_train,
        lines_valid=lines_valid, path_images_valid=path_images_valid,
        path_dict_char=path_dict_word)

# --------------------------------------------------------------------------------
#       Main
# --------------------------------------------------------------------------------
if args.train_english == "True":
    trainer.setup_data(batch_size)
    trainer.train_teacher_forcing(10, path_checkpoints)
elif args.train_japanese == "True":
    trainer.setup_data(batch_size)
    trainer.train_teacher_forcing(10, path_checkpoints)
