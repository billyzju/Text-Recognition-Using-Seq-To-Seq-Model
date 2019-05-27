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
import tqdm


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
    def __init__(self, model, optimizer, config,
                 resume_path=None, train_logger=None, valid_logger=None):
        # Setup directory for checkpoint saving
        self.start_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
        self.checkpoint_dir = os.path.join(config['trainer']['checkpoint'],
                                           self.start_time)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # Setup device
        self.device, device_ids = self._prepare_device(config['trainer']['n_gpu'])
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.optimizer = optimizer
        self.train_logger = train_logger
        self.valid_logger = valid_logger
        self.config = config
        self.epochs = config['trainer']['epochs']
        self.save_freq = config['trainer']['save_freq']
        self.star_epoch = 1
        if resume_path is not None:
            self._resume_checkpoint(resume_path)

    def train(self):
        for epoch in range(self.star_epoch, self.epochs + self.star_epoch):
            print("\n--------------------------------------------------------")
            start_time = time.time()
            result = self._train_one_epoch(epoch)
            end_time = time.time()
            print("Time to train the epoch", end_time - start_time)

            # Log metrics
            if (self.train_logger is not None) or\
               (self.valid_logger is not None):
                for key, value in result.items():
                    if key == "train_metrics":
                        print("The metrics of training ", value)
                        logging(self.train_logger, value, epoch)
                    elif key == "valid_metrics":
                        print("The metrics of validating ", value)
                        logging(self.valid_logger, value, epoch)

            # Save checkpoints
            self._save_checkpoint(epoch, save_best=True)

    def _prepare_device(self, n_gpu_use):
        """
        Setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine, "
                  "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print("Warning: The number of GPU\'s configured to use is {}, "
                  "but only {} are available on this machine."
                  .format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _train_one_epoch(self, epoch):
        raise NotImplementedError

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            "epoch": epoch,
            "state_dict":  self.model.state_dict(),
            "config": self.config
        }
        # Save checkpoint
        if self.save_freq is not None:
            if epoch % self.save_freq == 0:
                file_name = os.path.join(self.checkpoint_dir,
                                         'epoch{}.pth'.format(epoch))
                torch.save(state, file_name)
        elif save_best:
            # Save the best checkpoints
            best_file = os.path.join(self.checkpoint_dir,
                                     'best_model.pth')
            torch.save(state, best_file)

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        """
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1

        # Load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            print("Warning: Architecture configuration given in config file "
                  "is different from that of checkpoint. This may yield an "
                  "exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        print("Checkpoint loaded. Resume training from epoch {}"
              .format(self.start_epoch))
