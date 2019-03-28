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
from utils.data_loader import IAMDataLoader
import matplotlib.pyplot as plt


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

    # Model
    models = config["models"]
    batch_size = models["batch_size"]
    d_model = models["d_model"]
    heads = models["heads"]
    N = models["N"]
    path_checkpoints = models["checkpoints"]
    path_logs = models["logs"]
    lr = models["learning_rate"]


# --------------------------------------------------------------------------------
#       Configuration for trainer
# --------------------------------------------------------------------------------
# Target vocab
with open(os.path.join(path_preprocessing_data, "dict_char.txt")) as f:
    trg_vocab = np.shape(f.readlines())[0]

# Define model
model = MainModel(trg_vocab, d_model, N, heads)
model = model.cuda()

# Init Xavier
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(),
                             lr=lr, betas=(0.9, 0.98), eps=1e-9)

# Path to data train
path_p_lines = os.path.join(path_preprocessing_data, path_label_lines)
full_path_images = os.path.join(path_preprocessing_data, "path_images.txt")
path_dict_char = os.path.join(path_preprocessing_data, "dict_char.txt")

lines_train = os.path.join(path_preprocessing_data, "lines_train.txt")
path_images_train = os.path.join(path_preprocessing_data, "images_train.txt")
lines_valid = os.path.join(path_preprocessing_data, "lines_valid.txt")
path_images_valid = os.path.join(path_preprocessing_data, "images_valid.txt")


# --------------------------------------------------------------------------------
#       Funcs
# --------------------------------------------------------------------------------
def train_model(epochs):

    # Setup data loader
    print("Dataloader for train ----------------------------------")
    train_dataloader = IAMDataLoader(batch_size, True, lines_train,
                                     path_images_train, path_dict_char,
                                     100, 3000)

    print("Dataloader for valid ----------------------------------")
    valid_dataloader = IAMDataLoader(batch_size, True, lines_valid,
                                     path_images_valid, path_dict_char,
                                     100, 3000)

    if train_dataloader is not None:
        train_dataloader = train_dataloader.loader()
    if valid_dataloader is not None:
        valid_dataloader = valid_dataloader.loader()

    # Train
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_dataloader):

            data = data.cuda()

            # img = data[0].cpu()
            # img = torchvision.transforms.functional.to_pil_image(img)
            # plt.imshow(img)
            # plt.show()

            index_target = target.cuda()

            # The words used to train model
            input_target = index_target[:, :-1]

            # The words we want model try to predict
            predict_target = index_target[:, 1:].contiguous().view(-1)

            # Clear gradients
            optimizer.zero_grad()

            # Output
            output = model(data, input_target, src_mask=None, trg_mask=None)
            # translate(output, predict_target, path_dict_char)

            # Loss
            predict_target = predict_target.long()
            loss = F.cross_entropy(output.view(-1, output.size(-1)),
                                   predict_target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(loss.item())

        # Save model checkpoints
        idx_checkpoint = epoch

        if os.path.exists(path_checkpoints + '/' + str(idx_checkpoint)):
            torch.save(model.state_dict(), path_checkpoints + '/' +
                       str(idx_checkpoint) + '/' + "model_checkpoint_" +
                       str(idx_checkpoint) + '.pth')
        else:
            os.mkdir(path_checkpoints + '/' + str(idx_checkpoint))
            torch.save(model.state_dict(), path_checkpoints + '/' +
                       str(idx_checkpoint) + '/' + "model_checkpoint_" +
                       str(idx_checkpoint) + '.pth')

        # Logger
        logs = {'train_loss': total_loss/len(train_dataloader)}

        print("epoch = %d ,batch_idx = %d, loss = %.3f " % (epoch + 1,
              batch_idx + 1, total_loss/len(train_dataloader)))


# --------------------------------------------------------------------------------
#       Main
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    train_model(20)
