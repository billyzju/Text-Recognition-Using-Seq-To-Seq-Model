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
from torch.autograd import Variable
from util.data_processing import create_mask, subsequent_mask
from util.metrics import translate, accuracy_char_1, accuracy_char_2
from util.metrics import accuracy_word
from util.logger import Logger
from torchsummary import summary


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
        train_dataloader = self.data_loader(
            batch_size, True, self.lines_train, self.path_images_train,
            self.path_dict_char, self.max_seq_len, None)

        valid_dataloader = self.data_loader(
            batch_size, True, self.lines_valid, self.path_images_valid,
            self.path_dict_char, self.max_seq_len, None)

        if train_dataloader is not None:
            print("Load data for train ----------------------------------")
            self.train_dataloader = train_dataloader.loader()
        if valid_dataloader is not None:
            print("Load data for valid ----------------------------------")
            self.valid_dataloader = valid_dataloader.loader()

    def validate_model(self):
        """
        Validate model on validate subset. In training stage, validating model with
        validating subset for saving checkpoints
        """
        valid_acc_char = 0
        valid_acc_seq = 0

        for batch_idx, (data, target) in enumerate(self.valid_dataloader):
            data = data.cuda()
            index_target = target.cuda()

            input_target = index_target[:, :-1]
            target_mask = create_mask(input_target)

            output = self.model(data, input_target,
                                trg_mask=target_mask)
            acc_char_2 = accuracy_char_2(output, index_target[:, 1:].
                                         long())
            acc_seq = accuracy_word(output, index_target[:, 1:].long())

            valid_acc_seq += acc_seq
            valid_acc_char += acc_char_2

        return (valid_acc_char / (batch_idx + 1),
                valid_acc_seq / (batch_idx + 1))

    def train_teacher_forcing(self, epochs, path_checkpoints):
        """
        Train model with teacher forcing. If training is enough good,
        model will achieve the same accuracy with greedy decoding
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
                predict_target = (index_target[:, 1:].contiguous().view(-1).
                                  long())

                # Clear gradients
                self.optimizer.zero_grad()

                # Output
                output = self.model(data, input_target,
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
                    # Reset metrics
                    train_loss = 0
                    train_acc_char_1 = 0
                    train_acc_char_2 = 0
                    train_acc_seq = 0

                if (batch_idx + 1) % 5000 == 0:
                    valid_acc = self.validate_model()[1]
                    print("valid_acc_seq = ", valid_acc)
                    if valid_acc > old_acc_seq:
                        old_acc_seq = valid_acc
                        save_checkpoints(path_checkpoints, "12042019",
                                         self.model)

    def greedy_decoding(self, batch_size, trg_vocab):
        """
        Decoding with greedy which is use for inference
        """
        valid_acc_char = 0
        valid_acc_seq = 0
        for batch_idx, (data, target) in enumerate(self.valid_dataloader):
            data = data.cuda()
            index_target = target.cuda()
            predict_target = index_target[:, 1:].contiguous().view(-1)

            embs = self.model.cnn_model(data)
            src_mask = Variable(subsequent_mask(embs.size(1)).
                                type_as(data))
            memory = self.model.transformer.encoder(embs, mask=src_mask)
            input_target = (torch.ones(batch_size, 1).fill_(trg_vocab - 2).
                            type_as(index_target))
            output_seq = (torch.ones(batch_size, 1, trg_vocab).
                          type_as(index_target))

            for i in range(self.max_seq_len - 1):
                # Output of Decoder
                output = self.model.transformer.decoder(
                    input_target,
                    memory,
                    src_mask=src_mask,
                    trg_mask=Variable(subsequent_mask(input_target.size(1)).
                                      type_as(data)))

                # Output of Linear layer
                output = self.model.transformer.out(output[:, i:i+1, :])
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
            valid_acc_seq += acc_seq
            valid_acc_char += acc_char_2

        return (valid_acc_char / (batch_idx + 1),
                valid_acc_seq / (batch_idx + 1))
