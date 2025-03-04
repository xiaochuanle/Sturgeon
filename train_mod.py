import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from sklearn import metrics
import torchmetrics
import numpy as np
import argparse
import os
import sys
import time
import re
from tqdm import tqdm
import copy
import logging
from queue import Queue
from threading import Thread
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import multiprocessing
from multiprocessing import Process, Manager
import random

from models.model import BiLSTM_attn
from dataloader import *

device = torch.device("cuda:1")

train_dir = "/data1/YHC/QiTan_data/Mod/train/"

kmer_size = 21

model_save = "/tmp/model/"
model_type = "lstm_attn_"
signal_len = 15
f_accuracy = torchmetrics.Accuracy('binary', num_classes=2).to(device)
f_precision = torchmetrics.Precision("binary", num_classes=2).to(device)
f_recall = torchmetrics.Recall('binary', num_classes=2).to(device)
f_F1_score = torchmetrics.F1Score('binary', num_classes=2).to(device)

train_batch_size = 1024
valid_batch_size = 4096
max_epoch = 25
SEED = 71
model = BiLSTM_attn(kmer=kmer_size,
                    hidden_size=256,
                    embed_size=[16, 4],
                    dropout_rate=0.3,
                    num_layer1=2,
                    num_layer2=3,
                    num_classes=2)

model = model.to(device)

model.load_state_dict(torch.load("/tmp/model/lstm_attn_b21_s15_epoch7_accuracy:0.9567.pt", map_location=device))

train_files = [train_dir + x for x in os.listdir(train_dir) if x.endswith("npy")]

random.seed(71)

step_interval = 4
counter = 0
step = 0
weight_rank = torch.from_numpy(np.array([1, 1])).float()
weight_rank = weight_rank.to(device)
criterion = nn.CrossEntropyLoss(weight=weight_rank)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=1, gamma=0.4)
scalar = torch.cuda.amp.GradScaler()
curr_best_accuracy = 0
model.train()
epoch = 0
global_best_accuracy = 0
for epoch in range(max_epoch):
    random.shuffle(train_files)
    curr_epoch_accuracy = 0
    no_best_model = True
    tlosses = torch.tensor([]).to(device)
    start = time.time()
    # k-fold training
    random.shuffle(train_files)
    split = int(0.7 * len(train_files))
    train_list = train_files[:split]
    valid_list = train_files[split:]
    for train_file in train_list[:]:
        print("Reading k-fold train_data from-{}".format(train_file))
        dataset_ = Dataset_npy(train_file, kmer=kmer_size)
        data_loader = torch.utils.data.DataLoader(dataset=dataset_,
                                                  batch_size=train_batch_size,
                                                  num_workers=16,
                                                  pin_memory=True,
                                                  shuffle=True)
        # loss = torch.tensor([np.inf]).cuda()
        pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc="Training")
        for i, sfeatures in pbar:
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda'): # enable amp to accelerate training
                kmer, signals, labels = sfeatures
                kmer, signals, labels = kmer.to(device), signals.float().to(device), labels.long().to(device)
                outputs, _ = model(kmer, signals)
                loss = criterion(outputs, labels)
            if torch.isnan(loss):
                print("NaN loss detected! Skipping this batch.")
                continue  # Skip this batch
            scalar.scale(loss).backward()
            scalar.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
            # optimizer.step()
            scalar.step(optimizer)
            scalar.update()
            tlosses = torch.concatenate((tlosses, loss.detach().reshape(1)))
            pbar.set_description("Loss: {:.4f}".format(loss.item()))

    model.eval()
    with torch.no_grad():
        vlosses, vaccus, vprecs, vrecas, vf1score = torch.tensor([]).to(device), \
            torch.tensor([]).to(device), torch.tensor([]).to(device), \
            torch.tensor([]).to(device), torch.tensor([]).to(device)
        for valid_file in valid_list[:]:
            print("Read k-fold valid_data from-{}".format(valid_file))
            dataset_ = Dataset_npy(valid_file, kmer=kmer_size)
            data_loader = torch.utils.data.DataLoader(dataset=dataset_,
                                                      batch_size=valid_batch_size,
                                                      num_workers=16,
                                                      pin_memory=True,
                                                      shuffle=True)
            for vi, vsfeatures in tqdm(enumerate(data_loader)):
                kmer, signals, vlabels = vsfeatures
                kmer = kmer.to(device)
                signals = signals.float().to(device)
                vlabels = vlabels.long().to(device)
                voutputs, vlogits = model(kmer, signals)
                vloss = criterion(voutputs, vlabels)

                _, vpredicted = torch.max(vlogits.data, 1)
                vpredicted = vpredicted.to(device)

                vaccus = torch.concatenate((vaccus, f_accuracy(vpredicted, vlabels).reshape(1)))
                vprecs = torch.concatenate((vprecs, f_precision(vpredicted, vlabels).reshape(1)))
                vrecas = torch.concatenate((vrecas, f_recall(vpredicted, vlabels).reshape(1)))
                vf1score = torch.concatenate((vf1score, f_F1_score(vpredicted, vlabels).reshape(1)))
                vlosses = torch.concatenate((vlosses, vloss.detach().reshape(1)))

        curr_epoch_accuracy = torch.mean(vaccus)
        if curr_epoch_accuracy > global_best_accuracy - 0.0002:
            traced_script_module = torch.jit.trace(model, (kmer, signals))
            torch.save(model.state_dict(),
                       model_save + model_type + 'b{}_s{}_epoch{}_accuracy:{:.4f}.pt'.format(kmer_size,
                                                                                              signal_len,
                                                                                              epoch + 1,
                                                                                              curr_epoch_accuracy))
            traced_script_module.save(
                model_save + model_type + 'script_b{}_s{}_epoch{}_accuracy:{:.4f}.pt'.format(kmer_size,
                                                                                             signal_len,
                                                                                             epoch + 1,
                                                                                             curr_epoch_accuracy))
            global_best_accuracy = curr_epoch_accuracy

    time_cost = time.time() - start
    print('Epoch [{}/{}], TrainLoss: {:.4f}; '
          'ValidLoss: {:.4f}, '
          'Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1-score: {:.4f}, '
          'curr_epoch_best_accuracy: {:.4f}; Time: {:.2f}s'
          .format(epoch + 1, max_epoch, torch.mean(tlosses),
                  torch.mean(vlosses), torch.mean(vaccus), torch.mean(vprecs),
                  torch.mean(vrecas), torch.mean(vf1score), curr_epoch_accuracy, time_cost))
    f_accuracy.reset()
    f_precision.reset()
    f_recall.reset()
    tlosses = torch.tensor([]).to(device)
    start = time.time()
    sys.stdout.flush()
    scheduler.step()
    model.train()