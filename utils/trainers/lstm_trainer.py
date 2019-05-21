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
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
from utils.data_processing import create_mask, subsequent_mask
from utils.metrics import*
from utils.logger import Logger
from torchsummary import summary
from utils.base.trainer_base import TrainerBase


# --------------------------------------------------------------------------------
# 		Funcs
# --------------------------------------------------------------------------------
class LSTMTrainer(TrainerBase):
    def __init__(self, model, optimizer, data_loader, config,
                 resume, resume_path, train_logger, valid_logger, labels_train,
                 path_images_train, labels_valid, path_images_valid,
                 path_dictionary):
        super(LSTMTrainer, self).__init__(model, optimizer, config, resume,
                                          resume_path, train_logger,
                                          valid_logger)
        self.config = config
        batch_size = config['trainer']['batch_size']
        # Setup dataloader for training
        train_data_loader = data_loader(
            batch_size, True, labels_train, path_images_train,
            path_dictionary, 25, None)
        # Setup dataloader for validating
        valid_data_loader = data_loader(
            batch_size, False, labels_valid, path_images_valid,
            path_dictionary, 25, None)

        if train_data_loader is not None:
            print("Load data for train ----------------------------------")
            self.train_data_loader = train_data_loader.loader()
        if valid_data_loader is not None:
            print("Load data for valid ----------------------------------")
            self.valid_data_loader = valid_data_loader.loader()

    def _train_one_epoch(self, epoch):
        # Train command from TrainerBase class
        self.model.train()
        total_loss = 0
        total_acc_char = 0
        total_acc_field = 0

        n_iter = len(self.train_data_loader)
        train_pbar = tqdm.tqdm(enumerate(self.train_data_loader), total=n_iter)

        for batch_idx, (data, target) in train_pbar:
            data = data.to(self.device)
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

            total_loss += loss.item()
            total_acc_char += acc_char
            total_acc_field += acc_field

        total_loss /= len(self.train_data_loader)
        total_acc_char /= len(self.train_data_loader)
        total_acc_field /= len(self.train_data_loader)

        train_log = {'loss': total_loss,
                     'acc_char': total_acc_char,
                     'acc_field': total_acc_field}
        log = {'train_metrics': train_log}
        if self.valid_logger is not None:
            valid_log = self._eval_one_epoch()
            log['valid_metrics'] = valid_log

        return log

    def _eval_one_epoch(self):
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
