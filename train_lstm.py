# --------------------------------------------------------------------------------
# 		Import
# --------------------------------------------------------------------------------
import torch
import torch.nn as nn
import argparse
import json
import numpy as np
import os
from torchsummary import summary
from ocr_models.lstm_model import LSTMModel
from data_loaders.iam_data_loader import IAMDataLoader
from data_loaders.jp_data_loader import JPDataLoader
from utils.logger import Logger
from trainers.lstm_trainer import LSTMTrainer


# --------------------------------------------------------------------------------
#       Parser
# --------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Training model')
parser.add_argument('--language', type=str, default='eng',
                    help='Train model for japanese or english')

parser.add_argument('--resume_path', type=str, default=None,
                    help='Path to checkpoint file to resume')

args = parser.parse_args()


# --------------------------------------------------------------------------------
#       Configuration
# --------------------------------------------------------------------------------
with open("config/config_lstm.json") as json_file:
    config = json.load(json_file)
    trainer_config = config["trainer"]
    model_config = config["model"]
    if args.language == "eng":
        # English data
        data_config = config["data_eng"]
    elif args.language == "jp":
        # Japanese Data
        data_config = config["data_jp"]

# Setup files for dataloader
data_files_dir = data_config["path_preprocessing_files"]
images_train = os.path.join(data_files_dir, data_config["file_images_train"])
images_valid = os.path.join(data_files_dir, data_config["file_images_valid"])
images_test = os.path.join(data_files_dir, data_config["file_images_test"])

labels_train = os.path.join(data_files_dir, data_config["file_labels_train"])
labels_valid = os.path.join(data_files_dir, data_config["file_labels_valid"])
labels_test = os.path.join(data_files_dir, data_config["file_labels_test"])

dictionary = os.path.join(data_files_dir, data_config["file_dict"])

# Target vocab size
with open(dictionary) as f:
    trg_vocab = np.shape(f.readlines())[0] + 2

# Define model
input_dim = model_config["input_dim"]
enc_hidden_dim = model_config["enc_hidden_dim"]
enc_bidirectional = model_config["enc_bidirectional"]
dec_hidden_dim = model_config["dec_hidden_dim"]
dec_bidirectional = model_config["dec_bidirectional"]
num_layer = model_config["num_layer"]
max_len = model_config["max_len"]

model = LSTMModel(input_dim, enc_hidden_dim, enc_bidirectional,
                  dec_hidden_dim, dec_bidirectional, num_layer,
                  trg_vocab)
# # Init Xavier
# for p in model.parameters():
#     if p.dim() > 1:
#         nn.init.xavier_uniform_(p)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(),
                             lr=trainer_config["lr"],
                             betas=(0.9, 0.98), eps=1e-9)

# Logger
train_logger = Logger("logs/eng/test/train/")
valid_logger = Logger("logs/eng/test/valid/")

# Trainer
trainer = LSTMTrainer(
    model=model, optimizer=optimizer, data_loader=IAMDataLoader,
    config=config, resume_path=args.resume_path,
    train_logger=train_logger, valid_logger=valid_logger,
    labels_train=labels_train, path_images_train=images_train,
    labels_valid=labels_valid, path_images_valid=images_valid,
    path_dictionary=dictionary, max_len=max_len, trg_vocab=trg_vocab)

# summary(model, [(1, 224, 224),()])
# --------------------------------------------------------------------------------
#       Main
# --------------------------------------------------------------------------------
trainer.train()
