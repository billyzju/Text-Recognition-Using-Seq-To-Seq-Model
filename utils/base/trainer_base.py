# --------------------------------------------------------------------------------
# 		Import
# --------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torchvision
import argparse
import json
import numpy as np
import datetime
import time
import os
from utils.data_processing import create_mask, subsequent_mask
from utils.metrics import translate, accuracy_char_1, accuracy_char_2
from utils.metrics import accuracy_word
from utils.logger import Logger
from torchsummary import summary


# --------------------------------------------------------------------------------
# 		Funcs
# --------------------------------------------------------------------------------
def logging(train_logger, result, step):
    for tag, value in result.items():
        train_logger.scalar_summary(tag, value, step)


# --------------------------------------------------------------------------------
# 		Class of base for trainer
# --------------------------------------------------------------------------------
class TrainerBase:
    def __init__(self, model, optimizer, config, resume, train_logger=None,
                 valid_logger=None):
        # Setup directory for checkpoint saving
        self.start_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
        self.checkpoint_dir = os.path.join(config['trainer']['checkpoint'],
                                           self.start_time)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.model = model
        self.optimizer = optimizer
        self.train_logger = train_logger
        self.valid_logger = valid_logger

        self.epochs = config['trainer']['epochs']
        self.save_freq = config['trainer']['save_freq']

    def train(self):
        for epoch in range(self.star_epoch, self.epochs + self.star_epoch):
            print("\n--------------------------------------------------------")
            start_time = time()
            result = self._train_one_epoch(epoch)
            if (self.train_logger is not None) and\
               (self.valid_logger is not None):
                for key, value in result.item():
                    if key == "train_metrics":
                        logging(self.train_logger, result, epoch)
                    elif key == "valid_metrics":
                        logging(self.valid_logger, result, epoch)

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            "epoch": epoch,
            "train_logger": self.train_logger,
            "valid_logger": self.valid_logger,
            "state_dict":  self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config
        }
        # Save checkpoint
        if self.save_freq is not None:
            if epoch % self.save_freq == 0:
                file_name = os.path.join(self.checkpoint_dir,
                                         'epoch{}.pth'.format(epoch))

        elif save_best:
            best_file = os.path.join(self.checkpoint_dir,
                                     'best_model.pth')


    def _resume_checkpoint(self, )



		
        

