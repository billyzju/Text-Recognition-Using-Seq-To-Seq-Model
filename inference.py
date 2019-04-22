import torch
import torch.nn as nn
import torchvision
import argparse
import json
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
from utils.data_processing import subsequent_mask
from utils.main_model import MainModel
from utils.iam_data_loader import IAMDataLoader
from utils.jp_data_loader import JPDataLoader
from utils.metrics import translate, accuracy_char_1, accuracy_char_2
from utils.metrics import accuracy_word

test_english = "False"
test_japanese = "True"

if test_english == "True":
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

if test_japanese == "True":
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

if test_english == "True":
    # Path to data
    path_p_lines = os.path.join(path_preprocessing_data, path_label_lines)
    full_path_images = os.path.join(path_preprocessing_data, "path_images.txt")
    path_dict_char = os.path.join(path_preprocessing_data, "dict_char.txt")

    lines_test = os.path.join(path_preprocessing_data, "lines_test.txt")
    path_images_test = os.path.join(path_preprocessing_data, "images_test.txt")

    # Target vocab
    with open(os.path.join(path_preprocessing_data, "dict_char.txt")) as f:
        trg_vocab = np.shape(f.readlines())[0] + 2

    # Define model
    model = MainModel(trg_vocab, d_model, N, heads)
    model = model.cuda()
    model.load_state_dict(torch.load("checkpoints/12042019/model_checkpoint_12042019.pth"))

# Test japanese
if test_japanese == "True":
    # Path to data
    path_dict_word = os.path.join(path_preprocessing_data, "dict_word.txt")

    lines_test = os.path.join(path_preprocessing_data, "cinnamon_test_labels.txt")
    path_images_test = os.path.join(path_preprocessing_data, "cinnamon_test_images.txt")

    # Target vocab
    with open(os.path.join(path_preprocessing_data, "dict_word.txt"),
              encoding="utf8") as f:
        trg_vocab = np.shape(f.readlines())[0] + 2

    # Define model
    model = MainModel(trg_vocab, d_model, N, heads, 20)
    model = model.cuda()
    model.load_state_dict(torch.load("checkpoints/12042019/model_checkpoint_12042019.pth"))

test_dataloader = JPDataLoader(batch_size, True, lines_test, path_images_test,
                               path_dict_word, 20, None)

if test_dataloader is not None:
    test_dataloader = test_dataloader.loader()

# Metrics
test_loss = 0
test_acc_char_1 = 0
test_acc_char_2 = 0
test_acc_seq = 0

# Testing
for batch_idx, (data, target) in enumerate(test_dataloader):
    data = data.cuda()
    index_target = target.cuda()

    # The words we want model try to predict
    predict_target = index_target[:, 1:].contiguous().view(-1)

    # Output of CNN
    embs = model.cnn_model(data)

    # Output of Encoder
    memory = model.transformer.encoder(embs, mask=None)

    # Feed character for star of sequence
    input_target = torch.ones(batch_size, 1).fill_(trg_vocab - 2).\
        type_as(index_target)

    output_seq = torch.ones(batch_size, 1, 2628).type_as(index_target)

    for i in range(20 - 1):
        # Output of Decoder
        output = model.transformer.decoder(input_target,
                                           memory,
                                           src_mask=None,
                                           trg_mask=Variable(subsequent_mask(input_target.size(1))
                                                                             .type_as(data)))

        # output = model.transformer.out(output)
        output = model.transformer.out(output[:, i:i+1, :])

        # Probability layer
        prob = F.log_softmax(output, dim=-1)

        # Get index of next word
        _, next_word = torch.max(prob, dim=-1)
        next_word = next_word.type_as(index_target)
        # input_target = next_word
        input_target = torch.cat((input_target, next_word), dim=1)
        output_seq = torch.cat((output_seq, output), dim=1)

    # Accuracy
    output = output_seq[:, 1:, :]
    acc_char_1 = accuracy_char_1(output.contiguous().view(-1, output.size(-1)),
                                 predict_target.long())
    acc_char_2 = accuracy_char_2(output, index_target[:, 1:].long())
    acc_seq = accuracy_word(output, index_target[:, 1:].long())
    test_acc_char_1 += acc_char_1
    test_acc_char_2 += acc_char_2
    test_acc_seq += acc_seq
    if acc_seq == 1:
        translate(output.contiguous().view(-1, output.size(-1)), predict_target,
                path_dict_word)

        # Show image input
        img = data[0].cpu()
        img = torchvision.transforms.functional.to_pil_image(img)
        plt.imshow(img)
        plt.show()


print("accuracy test by char = %.3f, accuracy test by word = %.3f" % (test_acc_char_2 / (batch_idx + 1),
                                                                      test_acc_seq / (batch_idx + 1)))
