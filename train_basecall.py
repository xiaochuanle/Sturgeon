import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
from dataloader import load_numpy_data, ctc_dataset
from models.model import CRF_encoder, ctc_loss
import numpy as np
import time
import pandas as pd
import os
from utils.utils_func import get_logger, generate_5mer_dict
from models.decode_utils import *
from cpp_components.ctc_decoder import vertibi_decode_fast
from tqdm import tqdm
LOGGER = get_logger(__name__)


def train(args):
    st = time.time()
    torch.cuda.set_device(0)
    LOGGER.info("current device cuda:{}".format(torch.cuda.current_device()))
    init(args.seed, "cuda")
    LOGGER.info("Loading training data")
    train_data, valid_data = load_numpy_data(args.directory, args.split)
    train_data_set = ctc_dataset(train_data)
    valid_data_set = ctc_dataset(valid_data)
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=16,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_data_set,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=16,
                                               pin_memory=True)
    LOGGER.info("Load data finished!")
    model = CTC_encoder(n_hid=args.n_hid).cuda()
    # model = Transformer_CTC_encoder(n_hid=args.n_hid, num_tf=6)
    if args.load_previous is not None:
        LOGGER.info("Loading previous model")
        model.load_state_dict(torch.load(args.load_previous))
    model = model.cuda()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CTCLoss()
    # criterion = SmoothCTCLoss(weight=0.0001)
    step_interval = int((len(train_loader) + 4) * args.step_rate)

    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=0.000001)
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                              first_cycle_steps=1000,
                                              cycle_mult=1.0,
                                              max_lr=args.learning_rate,
                                              min_lr=0.000001,
                                              warmup_steps=250,
                                              gamma=1.0)
    model.train()
    total_tlosses, tlosses, vlosses, vaccs, step_vloss, step_vacc = [], [], [], [], [], []
    global_vloss = 1.
    scalar = torch.cuda.amp.GradScaler()

    for epoch in range(args.num_epochs):
        pbar_train = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (datas, targets, target_lengths) in pbar_train:
            optimizer.zero_grad()
            # use torch automatic mixed precision to accelerate training speed
            with torch.autocast(device_type='cuda'):
                datas, targets, target_lengths = datas.cuda(), targets.cuda(), target_lengths.cuda()
                inputs = model(datas)
                input_lengths = torch.full(size=(inputs.shape[1],),
                                          fill_value=inputs.shape[0],
                                          dtype=torch.long).cuda()
                # loss = ctc_loss(inputs, targets, target_lengths)
                loss = criterion(inputs, targets, input_lengths, target_lengths)

            if torch.isnan(loss):
                print("NaN loss detected! Skipping this batch.")
                continue  # Skip this batch
            scalar.scale(loss).backward()
            # loss.backward()
            scalar.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            # optimizer.step()
            scalar.step(optimizer)
            scalar.update()
            scheduler.step()
            tlosses.append(loss.detach().cpu().numpy()[()])
            pbar_train.set_description("Loss: {:.4f}".format(loss.item()))
            if (batch_idx + 1) % step_interval == 0 or (batch_idx + 1) == len(train_loader):
                # code for valid
                model.eval()
                vlosses, vaccs = [], []
                with torch.no_grad():
                    for _, (datas, targets, target_lengths) in enumerate(valid_loader):
                        datas, targets, target_lengths = datas.cuda(), targets.cuda(), target_lengths.cuda()
                        inputs = model(datas)
                        input_lengths = torch.full(size=(inputs.shape[1],),
                                                  fill_value=inputs.shape[0],
                                                  dtype=torch.long).cuda()
                        loss = criterion(inputs, targets, input_lengths, target_lengths)
                        seqs_, traces, quals = vertibi_decode_fast(inputs.cpu().numpy())
                        seqs = ["".join(alphabet[x] for x in seq if x != 0) for seq in seqs_]
                        refs = [decode_ref(target, alphabet) for target in targets]
                        accs = [
                            accuracy(ref, seq, min_coverage=0.3) if len(seq) else 0. for ref, seq in zip(refs, seqs)
                        ]
                        vaccs += accs
                        vlosses.append(loss.detach().cpu().numpy()[()])

                print("Epoch: {}, step: [{}/{}], train loss: {:4f},  valid loss:{:4f}, mean acc:{:4f}"
                      .format(epoch + 1, batch_idx + 1, len(train_loader),
                                                      np.mean(tlosses), np.mean(vlosses), np.mean(vaccs))
                              )
                step_vloss.append(np.mean(vlosses))
                step_vacc.append(np.mean(vaccs))
                model.train()
                total_tlosses += tlosses
                tlosses = []
        if global_vloss > np.mean(vlosses):
            global_vloss = np.mean(vlosses)
            torch.save(model.state_dict(), os.path.join(args.model_save, "{}_epoch:{}_loss:{:4f}_model.pt".format(args.save_tag, epoch + 1, np.mean(vlosses), np.mean(vaccs))))
        tlosses, vlosses, vaccs = [], [], []

    pd.DataFrame(
        data={
            "train_loss_per_batch_{}".format(args.batch_size): total_tlosses,
        },
        index=range(1, len(total_tlosses) + 1)
    ).to_csv(os.path.join(args.model_save, "{}_training_detail_epoch{}.csv".format(args.save_tag, epoch)))

    pd.DataFrame(
        data = {
            "step_vloss" : step_vloss,
            "step_vacc" : step_vacc,
        }
    ).to_csv(os.path.join(args.model_save, "{}_valid_per_step.csv".format(args.save_tag)))

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
    parser.add_argument("--split", type=float, default=0.99,
                        help="percentage of data to use for training")
    parser.add_argument("--n_hid", type=int, default=512,
                        help="num of hidden layers for lstm (or d_model for transformer-based)"
                            )
    parser.add_argument("--batch_size", type=int, default=128,
                        help="batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="learning rate")
    parser.add_argument("--clip", type=float, default=0.7,
                        help="gradient clipping")
    parser.add_argument("--load_previous", type=str, default=None,
                        help="load previous saved models")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed")

    return parser



if __name__ == "__main__":
    parser = argparser()

    args = parser.parse_args()

    train(args)
