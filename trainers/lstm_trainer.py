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
import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
from utils.data_processing import create_mask, subsequent_mask
from utils.metrics import*
from base.trainer_base import TrainerBase


# --------------------------------------------------------------------------------
# 		Funcs
# --------------------------------------------------------------------------------
class LSTMTrainer(TrainerBase):
    def __init__(self, model, optimizer, data_loader, config,
                 resume_path, train_logger, valid_logger, labels_train,
                 path_images_train, labels_valid, path_images_valid,
                 path_dictionary, max_len, trg_vocab):
        super(LSTMTrainer, self).__init__(model, optimizer, config,
                                          resume_path, train_logger,
                                          valid_logger)
        self.config = config
        self.path_dict = path_dictionary
        self.batch_size = config['trainer']['batch_size']
        self.max_len = max_len
        self.trg_vocab = trg_vocab
        # Setup data loader for training
        train_data_loader = data_loader(
            config=config, shuffle=True, labels=labels_train,
            path_images=path_images_train, dictionary=path_dictionary, max_len=max_len)
        # Setup data loader for validating
        valid_data_loader = data_loader(
            config=config, shuffle=False, labels=labels_valid,
            path_images=path_images_valid, dictionary=path_dictionary, max_len=max_len)

        if train_data_loader is not None:
            print("Load data for train ...")
            self.train_data_loader = train_data_loader.loader()
        if valid_data_loader is not None:
            print("Load data for valid ...")
            self.valid_data_loader = valid_data_loader.loader()

    def _train_one_epoch(self, epoch):
        # Train command from TrainerBase class
        self.model.train()
        total_loss = 0
        total_acc_char = 0
        total_acc_field = 0

        n_iter = len(self.train_data_loader)
        print("Training model")
        train_pbar = tqdm.tqdm(enumerate(self.train_data_loader), total=n_iter)
        for batch_idx, (data, target) in train_pbar:
            data = data.to(self.device)
            # image = data[0].cpu()
            # import torchvision.transforms as transforms
            # import matplotlib.pyplot as plt
            # show = transforms.ToPILImage()
            # image = show(image)
            # plt.imshow(image)
            # plt.show()
            index_target = target.to(self.device)
            # The words we feed to force
            input_target = index_target[:, :-1]
            # The words we want model try to predict
            predict_target = (index_target[:, 1:].contiguous().view(-1).
                              long())
            # Clear gradients
            self.optimizer.zero_grad()
            # Output
            output = self.model(data, input_target)
            output = output.transpose(0, 1)
            # Cross entropy loss
            loss = F.cross_entropy(output.contiguous().view(-1, output.size(-1)),
                                   predict_target.long())
            loss.backward()
            self.optimizer.step()

            # Metrics
            acc_char = accuracy_char_2(output, index_target[:, 1:].long())
            acc_field = accuracy_word(output, index_target[:, 1:].long())
            # translate(output.contiguous().view(-1, output.size(-1)),
            #           predict_target, self.path_dict)

            total_loss += loss.item()
            total_acc_char += acc_char
            total_acc_field += acc_field
            break
        total_loss /= len(self.train_data_loader)
        total_acc_char /= len(self.train_data_loader)
        total_acc_field /= len(self.train_data_loader)

        train_log = {'loss': total_loss,
                     'acc_char': total_acc_char,
                     'acc_field': total_acc_field}
        log = {'train_metrics': train_log}
        if self.valid_logger is not None:
            print("Validating model")
            valid_log = self._eval_one_epoch_greedy()
            log['valid_metrics'] = valid_log

        return log

    def _eval_one_epoch(self):
        """
        Validating model with teacher forcing
        """
        total_loss = 0
        total_acc_char = 0
        total_acc_field = 0

        n_iter = len(self.valid_data_loader)
        valid_pbar = tqdm.tqdm(enumerate(self.valid_data_loader), total=n_iter)
        with torch.no_grad():
            for batch_idx, (data, target) in valid_pbar:
                data = data.to(self.device)
                index_target = target.to(self.device)
                # The words we feed to force
                input_target = index_target[:, :-1]
                # The words we want model try to predict
                predict_target = (index_target[:, 1:].contiguous().view(-1).
                                  long())
                # Output
                output = self.model(data, input_target)
                output = output.transpose(0, 1)
                # Cross entropy loss
                loss = F.cross_entropy(output.contiguous().view(-1, output.size(-1)),
                                       predict_target.long())
                # Metrics
                acc_char = accuracy_char_2(output, index_target[:, 1:].long())
                acc_field = accuracy_word(output, index_target[:, 1:].long())

                total_loss += loss.item()
                total_acc_char += acc_char
                total_acc_field += acc_field

            total_loss /= len(self.valid_data_loader)
            total_acc_char /= len(self.valid_data_loader)
            total_acc_field /= len(self.valid_data_loader)

            valid_log = {'loss': total_loss,
                         'acc_char': total_acc_char,
                         'acc_field': total_acc_field}
        return valid_log

    def _eval_one_epoch_greedy(self):
        """
        Validating model with greedy
        """
        total_loss = 0
        total_acc_char = 0
        total_acc_field = 0

        n_iter = len(self.valid_data_loader)
        valid_pbar = tqdm.tqdm(enumerate(self.valid_data_loader), total=n_iter)
        with torch.no_grad():
            for batch_idx, (data, target) in valid_pbar:
                data = data.to(self.device)
                index_target = target.to(self.device)
                predict_target = index_target[:, 1:].contiguous().view(-1).long()
                embs = self.model.cnn_model(data)
                context, hidden_state, hidden_cell = self.model.lstm.encoder(embs)
                # The character for start of sequence
                input_target = (torch.ones(self.batch_size, 1).fill_(self.trg_vocab - 2).
                                type_as(index_target))
                output_seq = torch.ones(self.batch_size, 1, self.trg_vocab).type_as(index_target)
                for i in range(self.max_len + 1):
                    # Output of Decoder
                    output, (hidden_state, hidden_cell) = self.model.lstm.decoder(
                                                            input_target, hidden_state,
                                                            hidden_cell, context,
                                                            context.size(1))
                    output = output.transpose(0, 1)
                    # Probability of output
                    prob = F.log_softmax(output, dim=-1)
                    # Get index of next word
                    _, next_word = torch.max(prob, dim=-1)
                    next_word = next_word.type_as(index_target)
                    # input_target = torch.cat((input_target, next_word), dim=1)
                    input_target = next_word
                    output_seq = torch.cat((output_seq, output), dim=1)

                output = output_seq[:, 1:, :]
                # Cross entropy loss
                loss = F.cross_entropy(output.contiguous().view(-1, output.size(-1)),
                                       predict_target.long())
                # Metrics
                acc_char = accuracy_char_2(output, index_target[:, 1:].long())
                acc_field = accuracy_word(output, index_target[:, 1:].long())

                total_loss += loss.item()
                total_acc_char += acc_char
                total_acc_field += acc_field

            total_loss /= len(self.valid_data_loader)
            total_acc_char /= len(self.valid_data_loader)
            total_acc_field /= len(self.valid_data_loader)

            valid_log = {'loss': total_loss,
                         'acc_char': total_acc_char,
                         'acc_field': total_acc_field}
        return valid_log
