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
from utils.data_loaders.jp_data_loader import JPDataLoader
from utils.data_loaders.iam_data_loader import IAMDataLoader
import matplotlib.pyplot as plt
from utils.data_processing import subsequent_mask
from utils.metrics import translate, accuracy_char_1, accuracy_char_2
from utils.metrics import accuracy_word
from utils.logger import Logger
from torch.autograd import Variable

with open("preprocessing_data/jp/dict_word.txt", encoding="utf8") as f:
    dic = f.readlines()
    dic_n = []
    for i in dic:
        dic_n.append(i.split('\\')[0])

with open("E:\data\OCR\Form12_Field8_Lines_Grayscale/Form12_field8_ocr_Val_label_new100Fuku.txt",
            encoding="utf8") as f:
    lines = f.readlines()
    images = []
    labels = []
    for i in lines:
        image = i.split('|')[0]
        label = i.split('|')[1]
        if len(label) <= 18:
            n = 0
            for j in range(len(label)):
                if label[j] in dic_n:
                    n += 1
            if n == len(label):
                labels.append(label)
                images.append(image)

with open("preprocessing_data/jp/test8_images.txt", "w") as f:
    for i in images:
        f.write("E:\data\OCR\Form12_Field8_Lines_Grayscale\Form12_Field8_Lines_Grayscale/" +
                i + '\n')

with open("preprocessing_data/jp/test8_labels.txt", "w", encoding="utf8") as f:
    for i in labels:
        f.write(i)

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
path_dict_word = os.path.join(path_preprocessing_data, "dict_word.txt")

lines_test = os.path.join(path_preprocessing_data,
                            "test8_labels.txt")
path_images_test = os.path.join(path_preprocessing_data,
                                    "test8_images.txt")


# Target vocab
with open(os.path.join(path_preprocessing_data, "dict_word.txt"),
            encoding="utf8") as f:
    trg_vocab = np.shape(f.readlines())[0] + 2

# Define model
model = MainModel(trg_vocab, d_model, N, heads)
model = model.cuda()
model.load_state_dict(
    torch.load("checkpoints/jp/12042019/model_checkpoint_12042019.pth"))

test_dataloader = JPDataLoader(batch_size, False, lines_test, path_images_test,
                                path_dict_word, 20, None)
if test_dataloader is not None:
    test_dataloader = test_dataloader.loader()

test_acc_char_2 = 0
test_acc_seq = 0

for batch_idx, (data, target) in enumerate(test_dataloader):
    data = data.cuda()
    index_target = target.cuda()
    predict_target = index_target[:, 1:].contiguous().view(-1)

    embs = model.cnn_model(data)
    memory = model.transformer.encoder(embs, mask=None)
    input_target = (torch.ones(batch_size, 1).fill_(trg_vocab - 2).
                    type_as(index_target))
    output_seq = (torch.ones(batch_size, 1, trg_vocab).
                  type_as(index_target))

    for i in range(20 - 1):
        # Output of Decoder
        output = model.transformer.decoder(
            input_target,
            memory,
            src_mask=None,
            trg_mask=Variable(subsequent_mask(input_target.size(1)).
                              type_as(data)))

        # Output of Linear layer
        output = model.transformer.out(output[:, i:i+1, :])
        prob = F.log_softmax(output, dim=-1)

        # Get index of next word
        _, next_word = torch.max(prob, dim=-1)
        next_word = next_word.type_as(index_target)
        input_target = torch.cat((input_target, next_word), dim=1)
        output_seq = torch.cat((output_seq, output), dim=1)

    output = output_seq[:, 1:, :]
    predict_target = predict_target.long()
    acc_char_2 = accuracy_char_2(output, index_target[:, 1:].
                                 long())
    acc_seq = accuracy_word(output, index_target[:, 1:].long())
    test_acc_char_2 += acc_char_2
    test_acc_seq += acc_seq
    print(test_acc_char_2 / (batch_idx + 1), test_acc_seq / (batch_idx + 1))
