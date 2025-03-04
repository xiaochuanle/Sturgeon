import os
import multiprocessing
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
os.environ["OMP_NUM_THREADS"] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import argparse
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
from dataloader import load_numpy_data, ctc_dataset
import numpy as np
import time
from utils.utils_func import get_logger
from torch.utils.data.distributed import DistributedSampler

from models.decode_utils import *
from cpp_components.ctc_decoder import vertibi_decode_fast
from tqdm import tqdm
LOGGER = get_logger(__name__)

def save_checkpoint(epoch, model, optimizer, scheduler, path="checkpoint.pth"):
    checkpoint = {
        "epoch": epoch,  # 当前训练的epoch
        "model_state_dict": model.state_dict(),  # 模型参数
        "optimizer_state_dict": optimizer.state_dict(),  # 优化器状态
        "scheduler_state_dict": scheduler.save_state_dict(),  # 学习率调度器状态
        # "gradients": {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None},  # 保存梯度
    }
    torch.save(checkpoint, path)
    # print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, scheduler, device=None, path="checkpoint.pth"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # for name, param in model.named_parameters():
    #     if name in checkpoint["gradients"]:
    #         param.grad = checkpoint["gradients"][name]

    epoch = checkpoint["epoch"]  # 恢复当前epoch
    # print(f"Checkpoint loaded from {path}, resuming from epoch {epoch}")
    return epoch + 1

def distributed_data_parallel_train(args):
    st = time.time()
    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda', args.local_rank)
    torch.distributed.init_process_group(backend='nccl')
    init(args.seed, 'cuda')

    # model = CTC_encoder(n_hid=args.n_hid).to(device)
    model = Transformer_CTC_encoder(n_hid=args.n_hid, num_tf=12).to(device)
    # model = Mamba_CTC_encoder(n_hid=args.n_hid, num_layers=3).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[args.local_rank],
                                                      output_device=args.local_rank,
                                                      # find_unused_parameters=True
                                                      )

    LOGGER.info("Loading training data")
    train_data, valid_data = load_numpy_data(args.directory, args.split)
    train_data_set = ctc_dataset(train_data)
    valid_data_set = ctc_dataset(valid_data)
    train_sampler = DistributedSampler(train_data_set, shuffle=True)
    valid_sampler = DistributedSampler(valid_data_set, shuffle=False)
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               sampler=train_sampler,
                                               batch_size=args.batch_size,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_data_set,
                                               sampler=valid_sampler,
                                               batch_size=args.batch_size // 2,
                                               pin_memory=True)
    LOGGER.info("Load data finished!")

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CTCLoss()
    # criterion = SmoothCTCLoss(weight=0.0001)
    step_interval = int((len(train_loader) + 4) * args.step_rate)

    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0.00001)
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                              first_cycle_steps=len(train_loader) * args.num_epochs,
                                              cycle_mult=1.0,
                                              max_lr=args.learning_rate,
                                              min_lr=0.00001,
                                              warmup_steps=500,
                                              gamma=1.0)
    epoch_st = 0
    if args.load_previous is not None:
        LOGGER.info("Loading previous model")
        # model.load_state_dict(torch.load(args.load_previous, map_location=device, weights_only=True))
        epoch_st = load_checkpoint(model, optimizer, scheduler, device, args.load_previous)
    model.train()
    total_tlosses, tlosses, vlosses, vaccs, step_vloss, step_vacc = [], [], [], [], [], []
    global_vloss = 10.
    scalar = torch.amp.GradScaler()

    if args.local_rank == 0:
        train_result_path = os.path.join(args.model_save, "{}_training_detail.csv".format(args.save_tag))
        valid_result_path = os.path.join(args.model_save, "{}_validation_detail.csv".format(args.save_tag))
        if os.path.exists(train_result_path):
            train_logger = open(train_result_path, 'a')
            valid_logger = open(valid_result_path, 'a')
        else:
            train_logger = open(train_result_path, 'w')
            valid_logger = open(valid_result_path, 'w')
            train_logger.write("Epoch,batch,learning rate,losses\n")
            valid_logger.write("Epoch,batch,train_losses,valid_loss,acc\n")


    for epoch in range(epoch_st, args.num_epochs):
        st_epoch = time.time()
        if args.progress_bar and args.local_rank == 0:
            pbar_train = tqdm(enumerate(train_loader), total=len(train_loader))
        else:
            pbar_train = enumerate(train_loader)
        train_sampler.set_epoch(epoch)
        for batch_idx, (datas, targets, target_lengths) in pbar_train:
            optimizer.zero_grad()
            # use torch automatic mixed precision to accelerate training speed
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                datas, targets, target_lengths = datas.to(device), targets.to(device), target_lengths.to(device)
                inputs = model(datas)
                input_lengths = torch.full(size=(inputs.shape[1],),
                                          fill_value=inputs.shape[0],
                                          dtype=torch.long).to(device)
                loss = criterion(inputs, targets, input_lengths, target_lengths)
            if torch.isnan(loss):
                print("NaN loss detected! Skipping this batch.")
                continue  # Skip this batch
            scalar.scale(loss).backward()
            scalar.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scalar.step(optimizer)
            scalar.update()
            tlosses.append(loss.detach().cpu().numpy()[()])
            if args.progress_bar and args.local_rank == 0:
                pbar_train.set_description("Loss: {:.4f}".format(loss.item()))
            if args.local_rank == 0:
                train_logger.write("{},{},{:6f},{:6f}\n".format(epoch + 1, batch_idx, scheduler.get_lr()[0], loss.item()))
            if (batch_idx + 1) % step_interval == 0 or (batch_idx + 1) == len(train_loader):
                # code for valid
                model.eval()
                vlosses, vaccs = [], []
                with torch.no_grad():
                    for _, (datas, targets, target_lengths) in enumerate(valid_loader):
                        datas, targets, target_lengths = datas.to(device), targets.to(device), target_lengths.to(device)
                        inputs = model(datas)
                        input_lengths = torch.full(size=(inputs.shape[1],),
                                                  fill_value=inputs.shape[0],
                                                  dtype=torch.long).to(device)
                        loss = criterion(inputs, targets, input_lengths, target_lengths)
                        seqs_, traces, quals = vertibi_decode_fast(inputs.cpu().numpy())
                        seqs = ["".join(alphabet[x] for x in seq if x != 0) for seq in seqs_]
                        refs = [decode_ref(target, alphabet) for target in targets]
                        accs = [
                            accuracy(ref, seq, min_coverage=0.3) if len(seq) else 0. for ref, seq in zip(refs, seqs)
                        ]
                        vaccs += accs
                        vlosses.append(loss.detach().cpu().numpy()[()])
                if args.local_rank == 0:
                    print("Epoch: {}, step: [{}/{}], train loss: {:4f},  valid loss:{:4f}, mean acc:{:4f}, time cost:{:4f}"
                          .format(epoch + 1, batch_idx + 1, len(train_loader),
                                                          np.mean(tlosses), np.mean(vlosses), np.mean(vaccs), time.time() - st_epoch))
                    valid_logger.write("{},{},{},{},{}\n".format(epoch + 1, batch_idx + 1,
                                                                  np.mean(tlosses), np.mean(vlosses), np.mean(vaccs)))
                st_epoch = time.time()
                step_vloss.append(np.mean(vlosses))
                step_vacc.append(np.mean(vaccs))
                model.train()
                total_tlosses += tlosses
                tlosses = []

        if global_vloss > np.mean(vlosses):
            global_vloss = np.mean(vlosses)
            # early_stop_cnt = 0
            save_path = os.path.join(args.model_save, "{}_epoch:{}_loss:{:4f}_lr:{:6f}_device-{}_model.pt"
                                                        .format(args.save_tag, epoch + 1,
                                                                np.mean(vlosses), scheduler.get_lr()[0], args.local_rank))
            save_checkpoint(epoch, model, optimizer, scheduler, save_path)

        torch.cuda.empty_cache()
        # train_sampler = DistributedSampler(train_data_set)
        # valid_sampler = DistributedSampler(valid_data_set)
        # train_loader = torch.utils.data.DataLoader(train_data_set,
        #                                            sampler=train_sampler,
        #                                            batch_size=args.batch_size,
        #                                            num_workers=16,
        #                                            pin_memory=True)
        # valid_loader = torch.utils.data.DataLoader(valid_data_set,
        #                                            sampler=valid_sampler,
        #                                            batch_size=args.batch_size,
        #                                            num_workers=16,
        #                                            pin_memory=True)
        tlosses, vlosses, vaccs = [], [], []

    LOGGER.info("traing finished, cost: {} seconds!".format(time.time() - st))
    return

def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=True
    )

    parser.add_argument("model_save", type=str,
                        help="path to save models and training info")

    parser.add_argument("directory", type=str,
                        help="path to load training data")
    parser.add_argument("--save_tag", type=str, default="CTC",
                        help="tag add to saving file")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="number of epochs to train")
    parser.add_argument("--T_max", type=int, default=40,
                        help="parameter for CosineAnnealingLR")
    parser.add_argument("--step_rate", type=float, default=0.25,
                        help="default steps (total_step * step_rate) to valid")
    parser.add_argument("--local-rank", type=int, default=-1,
                        help="")
    parser.add_argument("--split", type=float, default=0.99,
                        help="percentage of data to use for training")
    parser.add_argument("--n_hid", type=int, default=512,
                        help="num of hidden layers for lstm (or d_model for transformer-based)"
                            )
    parser.add_argument("--batch_size", type=int, default=128,
                        help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--clip", type=float, default=0.7,
                        help="gradient clipping")
    parser.add_argument("--load_previous", type=str, default=None,
                        help="load previous saved models")
    parser.add_argument('--progress_bar', action='store_true',
        help='If set, displays a progress bar during training')
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed")

    return parser



if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn')
    parser = argparser()

    args = parser.parse_args()

    distributed_data_parallel_train(args)
