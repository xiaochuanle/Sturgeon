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



def compute_valid(y_true, y_pred):
    """
    compute accuracy, precision, recall, f1_score
    """
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, zero_division=np.nan)
    recall = metrics.recall_score(y_true, y_pred, zero_division=np.nan)
    f1_score = metrics.f1_score(y_true, y_pred, zero_division=np.nan)

    return accuracy, precision, recall, f1_score

def R(T, t=0.3, Tk=3):
    """
    get selected rate during co-teaching training
    """
    return 1 - t * min(T / Tk, 1)

def select_instance(model, kmer, signal, label, select_num):
    outputs, logits = model(kmer, signal)
    index_label = label.type(torch.int64).unsqueeze(1)
    probability = logits.gather(1, index_label)
    probability = probability.squeeze(1)
    one = torch.ones(probability.shape).cuda()
    instance_loss = one - probability
    _, instance_index = instance_loss.topk(select_num, 0, False, False)
    selected_kmer = torch.index_select(kmer, 0, instance_index)
    selected_signal = torch.index_select(signal, 0, instance_index)
    selected_label = torch.index_select(label, 0, instance_index)
    return selected_kmer, selected_signal, selected_label


def load_data_thread(train_dir_1: str,
                     train_dir_2: str,
                     train_batch_size: int,
                     valid_batch_size: int,
                     data_Q : Queue,
                     kmer_size: int,
                     max_epoch: int):
    train_files_1 = [os.path.join(train_dir_1, x) for x in os.listdir(train_dir_1) if x.endswith("npy")]
    train_files_2 = [os.path.join(train_dir_2, x) for x in os.listdir(train_dir_2) if x.endswith("npy")]
    random.shuffle(train_files_1)
    # train_files_1 = train_files_1[:len(train_files_2)]
    for epoch in range(max_epoch):
        random.shuffle(train_files_1)
        random.shuffle(train_files_2)
        file_cnt = min(len(train_files_1), len(train_files_2))
        split = int(file_cnt * 0.7)
        train_list_1 = train_files_1[: split]
        valid_list_1 = train_files_1[split: file_cnt]

        train_list_2 = train_files_2[: split]
        valid_list_2 = train_files_2[split: file_cnt]

        for i in range(len(train_list_1)):
        # for i in range(1):
            train_dataset = Dataset_npy_2(train_list_1[i], train_list_2[i], kmer=kmer_size)
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=train_batch_size,
                                                       num_workers=16,
                                                       pin_memory=True,
                                                       shuffle=True)
            data_Q.put((copy.deepcopy(train_loader), "train"))
            print("loaded train data from {} and {}".format(train_list_1[i], train_list_2[i]))

        for i in range(len(valid_list_2)):
        # for i in range(1):
            valid_dataset = Dataset_npy_2(valid_list_1[i], valid_list_2[i], kmer=kmer_size)
            valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                       batch_size=valid_batch_size,
                                                       num_workers=16,
                                                       pin_memory=True,
                                                       shuffle=True)
            data_Q.put((copy.deepcopy(valid_loader), "valid"))
            print("loaded valid data from {} and {}".format(valid_list_1[i], valid_list_2[i]))

        data_Q.put((None, "next"))


def train_data_thread(model,
                      data_Q : Queue,
                      model_save: str,
                      model_type: str,
                      signal_len: int,
                      kmer_size: int,
                      max_epoch=25,
                      learning_rate = 0.005,
                      ):
    f_accuracy = torchmetrics.Accuracy('binary', num_classes=2).cuda()
    f_precision = torchmetrics.Precision("binary", num_classes=2).cuda()
    f_recall = torchmetrics.Recall('binary', num_classes=2).cuda()
    f_F1_score = torchmetrics.F1Score('binary', num_classes=2).cuda()
    model = model.cuda()
    weight_rank = torch.from_numpy(np.array([1, 1])).float()
    weight_rank = weight_rank.cuda()
    criterion = nn.CrossEntropyLoss(weight=weight_rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    # scheduler = CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=0.00001)
    global_best_accuracy = 0
    curr_epoch_accuracy = 0
    model.train()
    for epoch in range(max_epoch):
        tlosses = torch.tensor([]).cuda()
        vlosses, vaccus, vprecs, vrecas, vf1score = torch.tensor([]).cuda(), \
            torch.tensor([]).cuda(), torch.tensor([]).cuda(), \
            torch.tensor([]).cuda(), torch.tensor([]).cuda()
        start = time.time()
        print("try to get a chunk of data")
        data_loader, flag = data_Q.get()
        print("got a chunk of data, start to training")
        while flag != "next":
            # LOGGER.info("Start to process data")
            if flag == "train":
                model.train()
                for sfeatures in data_loader:
                    kmer, signals, labels = sfeatures
                    kmer = kmer.cuda()
                    signals = signals.float().cuda()
                    labels = labels.long().cuda()
                    outputs, _ = model(kmer, signals)
                    loss = criterion(outputs, labels)
                    tlosses = torch.concatenate((tlosses, loss.detach().reshape(1)))
                    if (torch.isnan(loss)):
                        print("find loss nan")
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), .7)
                    optimizer.step()
            elif flag == "valid":
                model.eval()
                with torch.no_grad():
                    for vsfeatures in data_loader:
                        vkmer, vsignals, vlabels = vsfeatures
                        vkmer = vkmer.cuda()
                        vsignals = vsignals.float().cuda()
                        vlabels = vlabels.long().cuda()
                        voutputs, vlogits = model(vkmer, vsignals)

                        vloss = criterion(voutputs, vlabels)

                        if (torch.isnan(vloss)):
                            print("find loss nan")

                        _, vpredicted = torch.max(vlogits.data, 1)
                        vpredicted = vpredicted.cuda()

                        vaccus = torch.concatenate((vaccus, f_accuracy(vpredicted, vlabels).reshape(1)))
                        vprecs = torch.concatenate((vprecs, f_precision(vpredicted, vlabels).reshape(1)))
                        vrecas = torch.concatenate((vrecas, f_recall(vpredicted, vlabels).reshape(1)))
                        vf1score = torch.concatenate((vf1score, f_F1_score(vpredicted, vlabels).reshape(1)))
                        vlosses = torch.concatenate((vlosses, vloss.detach().reshape(1)))
            # LOGGER.info("process data done!")
            data_loader, flag = data_Q.get()
        curr_epoch_accuracy = torch.mean(vaccus)
        if curr_epoch_accuracy > global_best_accuracy - 0.0002:
            traced_script_module = torch.jit.trace(model, (vkmer, vsignals))
            torch.save(model.state_dict(),
                       model_save + model_type + 'b{}_s{}_epoch{}_accuracy:{:.4f}.pt'.format(kmer_size,
                                                                                              signal_len,
                                                                                              epoch + 1,
                                                                                              curr_epoch_accuracy))
            traced_script_module.save(model_save + model_type + 'script_b{}_s{}_epoch{}_accuracy:{:.4f}.pt'.format(kmer_size,
                                                signal_len,
                                                epoch + 1, curr_epoch_accuracy))
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
        f_F1_score.reset()
        start = time.time()
        sys.stdout.flush()
        scheduler.step()

if __name__ == "__main__":
    """
    this is a python script for train lstm model from scratch, 
    the data is split into separate chunks due to its large memory usagecd
    """
    # train_dir_1 = "/public1/YHC/QiTanTechData/YF6418_fe/merge"
    # train_dir_2 = "/public1/YHC/QiTanTechData/YF6419_fe/merge"
    train_dir_1 = "/public1/YHC/QiTanTechData/YF6418_fe"
    train_dir_2 = "/public1/YHC/QiTanTechData/YF6419_fe"

    save_dir = "/tmp/model/"
    model_type = "bilstm_adamw"
    signal_len = 15
    kmer_size = 21
    train_batch_size = 1024
    valid_batch_size = 4096
    max_epoch = 25
    multiprocessing.set_start_method('spawn')
    model = ModelBiLSTM_v2(kmer=kmer_size,
                           hidden_size=256,
                           embed_size=[16, 4],
                           dropout_rate=0.5,
                           num_layer1=2,
                           num_layer2=3,
                           num_classes=2)
    data_Q = multiprocessing.Queue(maxsize=1)

    load_proc = Process(target=load_data_thread, args=(train_dir_1,
                                                       train_dir_2,
                                                       train_batch_size,
                                                       valid_batch_size,
                                                       data_Q,
                                                       kmer_size,
                                                       max_epoch))
    train_proc = Process(target=train_data_thread, args=(model,
                                                         data_Q,
                                                         save_dir,
                                                         model_type,
                                                         signal_len,
                                                         kmer_size,
                                                         max_epoch))
    load_proc.start()
    train_proc.start()

    load_proc.join()
    train_proc.join()