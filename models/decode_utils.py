import torch
import numpy as np

# import koi
# from koi.ctc import SequenceDist, Max, Log, semiring, grad
# from koi.ctc import logZ_cu, viterbi_alignments, logZ_cu_sparse, bwd_scores_cu_sparse, fwd_scores_cu_sparse
# from koi.decode import beam_search
# from fast_ctc_decode import beam_search
from torch.cuda import get_device_capability
import parasail
import re
from collections import defaultdict

from models.model import *

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1
                 ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in
                    self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    # 保存状态的方法
    def save_state_dict(self):
        """保存调度器的状态"""
        state_dict = {
            "cycle": self.cycle,  # 当前周期数
            "step_in_cycle": self.step_in_cycle,  # 当前周期内的步数
            "cur_cycle_steps": self.cur_cycle_steps,  # 当前周期总步数
            "max_lr": self.max_lr,  # 当前周期的最大学习率
            "last_epoch": self.last_epoch,  # 上一次训练的 epoch
        }
        return state_dict

    # 加载状态的方法
    def load_state_dict(self, state_dict):
        """加载调度器的状态"""
        self.cycle = state_dict["cycle"]  # 恢复当前周期数
        self.step_in_cycle = state_dict["step_in_cycle"]  # 恢复周期内步数
        self.cur_cycle_steps = state_dict["cur_cycle_steps"]  # 恢复当前周期总步数
        self.max_lr = state_dict["max_lr"]  # 恢复当前最大学习率
        self.last_epoch = state_dict["last_epoch"]  # 恢复上一次训练的 epoch

        # 手动更新学习率以保持状态一致
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr

split_cigar = re.compile(r"(?P<len>\d+)(?P<op>\D+)")

# def posteriors(scores, S:semiring=Log):
#     f = lambda x : logZ(x, S).sum()
#     return grad(f, scores)
# def viterbi( scores):
#     traceback = posteriors(scores, Max)
#     a_traceback = traceback.argmax(2)
#     moves = (a_traceback % len(alphabet)) != 0
#     paths = 1 + (torch.div(a_traceback, len(alphabet), rounding_mode="floor") % n_base)
#     return torch.where(moves, paths, 0)

def path_to_str( path):
    alphabet = ['N', 'A', 'C', 'G', 'T']
    alphabet = np.frombuffer(''.join(alphabet).encode(), dtype='u1')
    seq = alphabet[path[path != 0].cpu()]
    return seq.tobytes().decode()

# def decode_batch(x):
#     scores = posteriors(x.to(torch.float32)) + 1e-8
#     tracebacks = viterbi(scores.log()).to(torch.int16).T
#     return [path_to_str(x) for x in tracebacks]

def decode_ref(encoded, labels):
    # labels = {}
    # for line in f:
    #     line_s = line.strip().split("\t")
    #     labels[int(line_s[1])] = line_s[0]
    return ''.join(labels[e] for e in encoded.tolist() if e)

def parasail_to_sam(result, seq):
    """
    Extract reference start and sam compatible cigar string.

    :param result: parasail alignment result.
    :param seq: query sequence.

    :returns: reference start coordinate, cigar string.
    """
    cigstr = result.cigar.decode.decode()
    first = re.search(split_cigar, cigstr)

    first_count, first_op = first.groups()
    prefix = first.group()
    rstart = result.cigar.beg_ref
    cliplen = result.cigar.beg_query

    clip = '' if cliplen == 0 else '{}S'.format(cliplen)
    if first_op == 'I':
        pre = '{}S'.format(int(first_count) + cliplen)
    elif first_op == 'D':
        pre = clip
        rstart = int(first_count)
    else:
        pre = '{}{}'.format(clip, prefix)

    mid = cigstr[len(prefix):]
    end_clip = len(seq) - result.end_query - 1
    suf = '{}S'.format(end_clip) if end_clip > 0 else ''
    new_cigstr = ''.join((pre, mid, suf))
    return rstart, new_cigstr

def accuracy(ref, seq, balanced=False, min_coverage=0.0):
    """
    Calculate the accuracy between `ref` and `seq`
    """
    alignment = parasail.sw_trace_striped_32(seq, ref, 8, 4, parasail.dnafull)
    counts = defaultdict(int)

    q_coverage = len(alignment.traceback.query) / len(seq)
    r_coverage = len(alignment.traceback.ref) / len(ref)

    if r_coverage < min_coverage:
        return 0.0

    _, cigar = parasail_to_sam(alignment, seq)

    for count, op  in re.findall(split_cigar, cigar):
        counts[op] += int(count)

    if balanced:
        accuracy = (counts['='] - counts['I']) / (counts['='] + counts['X'] + counts['D'])
    else:
        accuracy = counts['='] / (counts['='] + counts['I'] + counts['X'] + counts['D'])
    return accuracy * 100

alphabet = {0 : 'N', 1 : 'A', 2 : 'C', 3 : 'G', 4 : 'T', }
def phred(x):
    """
    Convert probability to phred quality score.
    """
    x = np.array(x, dtype=np.float32)

    x = np.clip(x, 1e-7, 1.0 - 1e-7)

    return -10 * np.log10(1 - x)
def viterbi_decode(inputs):
    T, N, C = inputs.shape
    soft_inputs = np.exp(inputs.numpy().astype(np.float32))
    logits = soft_inputs.argmax(2)
    seqs, moves, quals = [], [], []
    for i in range(N):
        seq = np.zeros(T, dtype=np.uint8)
        ctc_pred = logits[:,i]
        move = np.zeros(T, dtype=np.uint8)
        qual = np.zeros(T, dtype=np.float32)
        if ctc_pred[0] != 0:
            seq[0] = ctc_pred[0]
            qual[0] = soft_inputs[0,i,ctc_pred[0]][()]
            move[0] = 1
        for j in range(1, T):
            if ctc_pred[j] != ctc_pred[j-1] and ctc_pred[j] != 0:
                seq[j] = ctc_pred[j]
                qual[j] = soft_inputs[j, i, ctc_pred[j]][()]
                move[j] = 1

        seqs.append(seq)
        moves.append(np.array(move, dtype=np.uint8))
        quals.append(np.array([x + 33 if x > 1e-6 else x for x in phred(qual)], dtype=np.uint8))
    return seqs, moves, quals

# def beam_search_decode(inputs):
#     T, N, C = inputs.shape
#     soft_inputs = np.exp(inputs.numpy().astype(np.float32))
#     base2code =  {'N':0, 'A':1, 'C':2, 'G':3, 'T':4 }
#     seqs = []
#     moves = []
#     quals = []
#     for i in range(N):
#         move = np.zeros(T, dtype=np.uint8)
#         qual = np.zeros(T, dtype=np.float32)
#         seq, path = beam_search(inputs[:,i,:], "NACGT", beam_size=5,  beam_cut_threshold=0.1)
#         seq = np.frombuffer(seq.encode('utf-8'), dtype=np.uint8)
#     # 施工中

def compute_scores(model, batch, beam_width=32, beam_cut=100.0, scale=1.0, offset=0.0, blank_score=2.0, reverse=False):
    """
    Compute scores for model.
    """
    def half_supported():
        """
        Returns whether FP16 is support on the GPU
        """
        try:
            return get_device_capability()[0] >= 7
        except:
            return False
    with torch.inference_mode():
        device = next(model.parameters()).device
        dtype = torch.float16 if half_supported() else torch.float32
        scores = model(batch.to(dtype).to(device))
        if reverse:
            scores = model.seqdist.reverse_complement(scores)
        scores = scores.permute(1, 0, 2)
        scores = scores.contiguous()
        # print(scores.shape)
        # with torch.cuda.device(scores.device):
            # sequence, qstring, moves = beam_search(
            #     scores, beam_width=beam_width, beam_cut=beam_cut,
            #     scale=scale, offset=offset, blank_score=blank_score
            # )
        # return {
        #     'moves': moves,
        #     'qstring': qstring,
        #     'sequence': sequence,
        # }