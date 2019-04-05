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
from utils.data_processing import create_mask
from utils.metrics import translate, accuracy_char
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

            acc_char = accuracy_char(output.view(-1, output.size(-1)),
                                     predict_target.long())

            valid_acc_char += acc_char
        return valid_acc_char / (batch_idx + 1)

    def train_teacher_forcing(self, epochs, path_checkpoints):
        """ Train model with teacher forcing
        """
        for epoch in range(epochs):
            # Metrics
            train_loss = 0
            train_acc_char = 0

            # Train
            for batch_idx, (data, target) in enumerate(self.train_dataloader):
                data = data.cuda()
                index_target = target.cuda()

                # The words used to train model
                input_target = index_target[:, :-1]
                target_mask = create_mask(input_target)

                # The words we want model try to predict
                predict_target = index_target[:, 1:].contiguous().view(-1)

                # Clear gradients
                self.optimizer.zero_grad()

                # Output
                output = self.model(data, input_target, src_mask=None,
                                    trg_mask=target_mask)
                # Translate and print output
                translate(output.view(-1, output.size(-1)), predict_target,
                          self.path_dict_char)

                # Cross entropy loss
                predict_target = predict_target.long()
                loss = F.cross_entropy(output.view(-1, output.size(-1)),
                                       predict_target)

                # Accuracy with char-level
                acc_char = accuracy_char(output.view(-1, output.size(-1)),
                                         predict_target)

                loss.backward()
                self.optimizer.step()

                # Logger
                if (batch_idx + 1) % 50 == 0:
                    info = {'train_loss': loss.item(),
                            'train acc char': acc_char}
                    for tag, value in info.items():
                        self.train_logger.scalar_summary(tag, value,
                                                         (batch_idx + 1) // 50)

                train_loss += loss.item()
                train_acc_char += acc_char

            print("epoch = %d ,train_acc_char = %.3f, train_loss = %.3f " %
                  (epoch + 1,
                   train_acc_char / len(self.train_dataloader),
                   train_loss / len(self.train_dataloader)))

            save_checkpoints(path_checkpoints, epoch, self.model)

    def train_greedy(self, batch_size, epochs, trg_vocab, path_checkpoints):
        """ Train model with greedy and greedy is also use for inference
        """
        for epoch in range(epochs):
            # Metrics
            train_loss = 0
            train_acc_char = 0

            # Train
            for batch_idx, (data, target) in enumerate(self.train_dataloader):
                data = data.cuda()
                index_target = target.cuda()

                # The words we want model try to predict
                predict_target = index_target[:, 1:].long()

                # Output of CNN
                embs = self.model.cnn_model(data)
                # Output of Encoder
                memory = self.model.transformer.encoder(embs, mask=None)

                # Clear gradients
                self.optimizer.zero_grad()

                # Feed character for star of sequence
                input_target = torch.ones(batch_size, 1).fill_(trg_vocab - 2).\
                    type_as(index_target)

                # Loss for backprop
                loss = 0

                for i in range(self.max_seq_len - 1):
                    print("pass")
                    # Output of Decoder
                    output = self.model.transformer.decoder(input_target,
                                                            memory,
                                                            src_mask=None,
                                                            trg_mask=None)

                    output = self.model.transformer.out(output[:, i:i+1, :])
                    # Loss for character ith
                    loss_tmp = F.cross_entropy(
                        output.view(-1, output.size(-1)),
                        predict_target[:, i:i+1].contiguous().view(-1))
                    loss += loss_tmp
                    # Probability layer
                    prob = F.log_softmax(output, dim=-1)
                    # Get index of next word
                    _, next_word = torch.max(prob, dim=-1)
                    next_word = next_word.type_as(index_target)
                    input_target = torch.cat((input_target, next_word), dim=1)

                # # Translate and print output
                # translate(input_target[:, 1:, :].view(-1, output.size(-1)),
                #           predict_target, self.path_dict_char)

                # # Loss of greedy is also cross entropy
                # output = output_seq[:, 1:, :]
                # loss = F.cross_entropy(output.view(-1, output.size(-1)),
                #                        predict_target)

                # acc_char = accuracy_char(output.view(-1, output.size(-1)),
                #                          predict_target)

                loss = loss / (self.max_seq_len - 1)
                # loss.backward()
                # self.optimizer.step()

                # Logger
                if (batch_idx + 1) % 50 == 0:
                    info = {'train_loss': loss.item(),
                            'train acc char': acc_char}
                    for tag, value in info.items():
                        self.train_logger.scalar_summary(tag, value,
                                                         (batch_idx + 1) // 50)

                train_loss += loss.item()
                train_acc_char += acc_char

            print("epoch = %d ,train_acc_char = %.3f, train_loss = %.3f " %
                  (epoch + 1,
                   train_acc_char / len(self.train_dataloader),
                   train_loss / len(self.train_dataloader)))

            save_checkpoints(path_checkpoints, epoch, self.model)
