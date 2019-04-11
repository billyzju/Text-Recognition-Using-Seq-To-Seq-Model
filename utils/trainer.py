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
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils.data_processing import create_mask
from utils.metrics import translate, accuracy_char_1, accuracy_char_2
from utils.metrics import accuracy_word
from utils.logger import Logger


# --------------------------------------------------------------------------------
# 		Funcs
# --------------------------------------------------------------------------------
def save_checkpoints(path_checkpoints, idx_checkpoint, model):
    if os.path.exists(path_checkpoints + '/' + str(idx_checkpoint)):
        torch.save(model.state_dict(), path_checkpoints + '/' +
                   str(idx_checkpoint) + '/' + "model_checkpoint_" +
                   str(idx_checkpoint) + '.pth')
    else:
        os.mkdir(path_checkpoints + '/' + str(idx_checkpoint))
        torch.save(model.state_dict(), path_checkpoints + '/' +
                   str(idx_checkpoint) + '/' + "model_checkpoint_" +
                   str(idx_checkpoint) + '.pth')


# --------------------------------------------------------------------------------
# 		Class
# --------------------------------------------------------------------------------
class Trainer:
    def __init__(self, model, data_loader, optimizer, train_logger,
                 valid_logger, max_seq_len, lines_train, path_images_train,
                 lines_valid, path_images_valid, path_dict_char):
        super(Trainer).__init__()
        # Model
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.train_logger = train_logger
        self.valid_logger = valid_logger
        self.max_seq_len = max_seq_len
        # Path
        self.lines_train = lines_train
        self.path_images_train = path_images_train
        self.lines_valid = lines_valid
        self.path_images_valid = path_images_valid
        self.path_dict_char = path_dict_char

    def setup_data(self, batch_size):
        """ Setup data loader
        """
        print("Load data for train ----------------------------------")
        train_dataloader = self.data_loader(
            batch_size, True, self.lines_train, self.path_images_train,
            self.path_dict_char, self.max_seq_len, None)

        print("Load data for valid ----------------------------------")
        valid_dataloader = self.data_loader(
            batch_size, True, self.lines_valid, self.path_images_valid,
            self.path_dict_char, self.max_seq_len, None)

        if train_dataloader is not None:
            self.train_dataloader = train_dataloader.loader()
        if valid_dataloader is not None:
            self.valid_dataloader = valid_dataloader.loader()

    def validate_model(self):
        """ Validate model on validate subset
        """
        valid_acc_char = 0
        for batch_idx, (data, target) in enumerate(self.valid_dataloader):
            data = data.cuda()
            index_target = target.cuda()

            input_target = index_target[:, :-1]
            target_mask = create_mask(input_target)

            predict_target = index_target[:, 1:].contiguous().view(-1)

            output = self.model(data, input_target, src_mask=None,
                                trg_mask=target_mask)

            acc_char = accuracy_char_1(output.contiguous().
                                       view(-1, output.size(-1)),
                                       predict_target.long())

            valid_acc_char += acc_char
        return valid_acc_char / (batch_idx + 1)

    def train_teacher_forcing(self, epochs, path_checkpoints):
        """ Train model with teacher forcing
        """
        step = 0
        for epoch in range(epochs):
            # Metrics
            train_loss = 0
            train_acc_char_1 = 0
            train_acc_char_2 = 0
            train_acc_seq = 0
            old_acc_seq = 0

            # Train
            for batch_idx, (data, target) in enumerate(self.train_dataloader):
                step += 1
                data = data.cuda()
                index_target = target.cuda()

                # Show image input
                # img = data[0].cpu()
                # img = torchvision.transforms.functional.to_pil_image(img)
                # plt.imshow(img)
                # plt.show()

                # The words used to train model
                input_target = index_target[:, :-1]
                target_mask = create_mask(input_target)

                # The words we want model try to predict
                predict_target = index_target[:, 1:].contiguous().view(-1).long()

                # Clear gradients
                self.optimizer.zero_grad()

                # Output
                output = self.model(data, input_target, src_mask=None,
                                    trg_mask=target_mask)

                # Cross entropy loss
                loss = F.cross_entropy(output.view(-1, output.size(-1)),
                                       predict_target.long())

                # Accuracy with char-level and seq-level
                acc_char_1 = accuracy_char_1(output.view(-1, output.size(-1)),
                                             predict_target)
                acc_char_2 = accuracy_char_2(output, index_target[:, 1:].
                                             long())
                acc_seq = accuracy_word(output, index_target[:, 1:].long())

                train_loss += loss.item()
                train_acc_char_1 += acc_char_1
                train_acc_char_2 += acc_char_2
                train_acc_seq += acc_seq

                loss.backward()
                self.optimizer.step()

                # Logger
                if (batch_idx + 1) % 500 == 0:
                    info = {'train_loss': train_loss / 500,
                            'train_acc_char_2': train_acc_char_2 / 500,
                            'train_acc_seq': train_acc_seq / 500}
                    for tag, value in info.items():
                        self.train_logger.scalar_summary(tag, value, step)

                    print("epoch = %d, train_acc_char_2 = %.3f, train_acc_seq = %.3f, train_loss = %.3f " %
                          (epoch + 1, train_acc_char_2 / 500,
                           train_acc_seq / 500, train_loss / 500))
                    # Translate and print output
                    translate(output.view(-1, output.size(-1)), predict_target,
                              self.path_dict_char)

                    if (train_acc_seq / 500) > old_acc_seq:
                        old_acc_seq = train_acc_seq / 500
                        save_checkpoints(path_checkpoints, "12042019", self.model)

                    # Reset metrics
                    train_loss = 0
                    train_acc_char_1 = 0
                    train_acc_char_2 = 0
                    train_acc_seq = 0

    def train_greedy(self, batch_size, epochs, trg_vocab, path_checkpoints):
        """ Train model with greedy and greedy is also use for inference
        """
        for epoch in range(epochs):
            # Metrics
            train_loss = 0
            train_acc_char_1 = 0
            train_acc_char_2 = 0
            train_acc_seq = 0
            old_acc_seq = 0

            # Train
            for batch_idx, (data, target) in enumerate(self.train_dataloader):
                data = data.cuda()
                index_target = target.cuda()

                # The words we want model try to predict
                predict_target = index_target[:, 1:].contiguous().view(-1)

                # Clear gradients
                self.optimizer.zero_grad()

                # Output of CNN
                embs = self.model.cnn_model(data)
                # Output of Encoder
                memory = self.model.transformer.encoder(embs, mask=None)

                # Feed character for star of sequence
                input_target = torch.ones(batch_size, 1).fill_(trg_vocab - 2).\
                    type_as(index_target)

                output_seq = torch.ones(8, 1, 82).type_as(index_target)

                for i in range(self.max_seq_len - 1):
                    # Output of Decoder
                    output = self.model.transformer.decoder(input_target,
                                                            memory,
                                                            src_mask=None,
                                                            trg_mask=None)

                    output = self.model.transformer.out(output[:, i:i+1, :])

                    # Probability layer
                    prob = F.log_softmax(output, dim=-1)

                    # Get index of next word
                    _, next_word = torch.max(prob, dim=-1)
                    next_word = next_word.type_as(index_target)
                    input_target = torch.cat((input_target, next_word), dim=1)
                    output_seq = torch.cat((output_seq, output), dim=1)

                # Loss
                output = output_seq[:, 1:, :]
                predict_target = predict_target.long()
                loss = F.cross_entropy(output.contiguous().view(-1, output.size(-1)),
                                       predict_target)

                acc_char_1 = accuracy_char_1(output.contiguous().view(-1, output.size(-1)),
                                             predict_target)
                acc_char_2 = accuracy_char_2(output, index_target[:, 1:].long())
                acc_seq = accuracy_word(output, index_target[:, 1:].long())

                train_loss += loss.item()
                train_acc_char_1 += acc_char_1
                train_acc_char_2 += acc_char_2
                train_acc_seq += acc_seq
                loss.backward()
                self.optimizer.step()

                # Logger
                if (batch_idx + 1) % 500 == 0:
                    info = {'train_loss': train_loss / 500,
                            'train_acc_char_2': train_acc_char_2 / 500,
                            'train_acc_seq': train_acc_seq / 500}
                    for tag, value in info.items():
                        self.train_logger.scalar_summary(tag, value,
                                                         (batch_idx + 1) // 500)

                    print("epoch = %d, train_acc_char_2 = %.3f, train_acc_seq = %.3f, train_loss = %.3f " %
                          (epoch + 1, train_acc_char_2 / 500,
                           train_acc_seq / 500, train_loss / 500))

                    # Reset metrics
                    train_loss = 0
                    train_acc_char_1 = 0
                    train_acc_char_2 = 0
                    train_acc_seq = 0

                if (batch_idx + 1) % 1000 == 0:
                    if (train_acc_seq / 1000) > old_acc_seq:
                        old_acc_seq = train_acc_seq
                        save_checkpoints(path_checkpoints, "09042019", self.model)
