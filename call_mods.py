import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
import torch
import numpy as np
import argparse
import os
import h5py
import mappy
import sys
import time
import re
from tqdm import tqdm
import random
import os

from multiprocessing import Pool, Manager
from utils.utils_func import *
from utils.readers import *
LOGGER = get_logger(__name__)
from models.model import *


def write_result_process(resQueue : multiprocessing.Queue,
                         write_path : str):
    LOGGER.info("Write result process started")
    w_file = open(write_path, "w")
    while True:
        try:
            siteinfo_batch, p_rate = resQueue.get()
            if siteinfo_batch is None: break
            for i in range(len(siteinfo_batch)):
                w_file.write(siteinfo_batch[i] + "\t" + str(p_rate[i]) + "\n")
        except:
            LOGGER.error("Error occured while writing result")
            time.sleep(1000)
    LOGGER.info("write result process finished")

def module_inference_process(dataQueue : multiprocessing.Queue,
                             resQueue: multiprocessing.Queue,
                             module_path : str,
                             ):
    LOGGER.info("Model inference process started")
    model = torch.jit.load(module_path)
    model = model.cuda()

    while True:
        try:
            siteinfo_batch, kmer_batch, signal_batch = dataQueue.get()
            if siteinfo_batch is None: break
            kmer_batch = torch.tensor(kmer_batch).to(torch.int32).cuda()
            signal_batch = torch.tensor(signal_batch).to(torch.float16).cuda()

            _, logits = model(kmer_batch, signal_batch)

            p_rate = logits[:, 1].cpu().detach().numpy()

            resQueue.put((siteinfo_batch, p_rate))

        except Exception as e:
            LOGGER.error("Error occured while running model")
            error_handler(e)
            time.sleep(100)
            # exit()

    resQueue.put((None, None))
    LOGGER.info("Model inference queue finished")


def extract_features(fastq_path : str,
                     h5_path : str,
                     ref_path : str,
                     dataQueue : multiprocessing.Queue(),
                     k_size : int = 21,
                     sig_len : int = 15,
                     down_sample_ratio : int = 5,
                     batch_size : int = 256,
                     ):

    st = time.time()

    LOGGER.info("Extract feature for {} and {} start".format(fastq_path.split("/")[-1], h5_path.split("/")[-1]))
    ref_genome = mappy.Aligner(ref_path)
    h5_reads = h5py.File(h5_path, 'r')

    fastq_l = []
    for line in open(fastq_path):
        fastq_l.append(line)

    num_bases = (k_size - 1) // 2
    cnt_total, cnt_selected = 0, 0
    siteinfo_batch = ["",] * batch_size
    kmer_batch = np.zeros((batch_size, k_size), dtype=np.int64)
    signal_batch = np.zeros((batch_size,  k_size, down_sample_ratio * 3 + 4), dtype=np.float32)
    fe_cnt = 0
    for i in range(int(len(fastq_l) / 4)):
        info = fastq_l[4 * i]
        seq = fastq_l[4 * i + 1]
        c = fastq_l[4 * i + 2]
        qual = fastq_l[4 * i + 3]
        f = Fastq_Read(info, seq, qual)

        signal = h5_reads["Raw_data"][f.read_id][()] # get signal
        first_hit = next(ref_genome.map(f.seq), None) # get mapping info
        if first_hit is None: continue

        cnt_total += 1

        # filter by some condition
        if len(signal) <= 6400: continue
        if f.q_score <= 9.0: continue
        if first_hit.mapq < 15: continue

        ref_seq = ref_genome.seq(first_hit.ctg)[first_hit.r_st : first_hit.r_en].upper()
        # query_seq = f.seq[first_hit.q_st: first_hit.q_en]
        if first_hit.strand == -1:
            ref_seq = complement_seq(ref_seq)
        strand_code = 1 if first_hit.strand == 1 else 0
        try:
            r_to_q_poss = parse_cigar(first_hit.cigar, strand_code, len(ref_seq))
        except Exception as e:
            error_handler(e)

        cnt_selected += 1
        base2signal = f.get_base2signal()
        motifs_loc = f.get_refloc_of_methysite_in_motif(ref_seq,["CG"], 0)
        if (len(motifs_loc) == 0): continue

        base_qual = f.get_seq_quality()
        # basecode = f.get_base_code()
        sshift, sscale = get_normalize_factor(signal)
        if sscale != 0:
            signal_norm = (signal - sshift) / sscale
        else:
            signal_norm = signal
        signal_group, sig_len_l = group_signal_by_base2signal(signal_norm,
                                                              base2signal,
                                                              f.trimed_start,
                                                              sig_len=sig_len,
                                                              stride=f.stride)

        ref_readlocs = np.zeros(len(ref_seq), dtype=np.float32)
        ref_signal_grp = np.zeros([len(ref_seq), sig_len], dtype=np.float32)
        ref_baseprobs = np.zeros(len(ref_seq), dtype=np.float32)
        ref_sig_len = np.zeros(len(ref_seq), dtype=np.float32)

        for ref_pos, q_pos in enumerate(r_to_q_poss[:-1]):
            ref_readlocs[ref_pos] = q_pos + first_hit.q_st
            ref_signal_grp[ref_pos] = signal_group[q_pos + first_hit.q_st]
            ref_baseprobs[ref_pos] = base_qual[q_pos + first_hit.q_st]
            ref_sig_len[ref_pos] = sig_len_l[q_pos + first_hit.q_st]


        for off_loc in motifs_loc:
            if off_loc < num_bases or off_loc >= len(ref_seq) - num_bases: continue
            abs_loc = (first_hit.r_st + off_loc) if strand_code == 1 else (first_hit.r_en - 1 - off_loc)

            # if f.read_id == "7FC02B65-4FC1-4888-B6AF-288B951AC26C" and off_loc == 270:
            #     LOGGER.info("found bug loc!!!!!")

            try:
                kmer_base = np.array([ base2code_dna[x]  for x in ref_seq[(off_loc - num_bases) : (off_loc + num_bases + 1)]],dtype=np.int32)
                k_seq_qual = ref_baseprobs[(off_loc - num_bases) : (off_loc + num_bases + 1)]
                k_signals = ref_signal_grp[(off_loc - num_bases) : (off_loc + num_bases + 1)]

                k_signal_lens = ref_sig_len[(off_loc - num_bases) : (off_loc + num_bases + 1)]

                signal_means = np.array([np.mean(x) for x in k_signals], dtype=np.float32)
                signal_stds = np.array([np.std(x) for x in k_signals], dtype=np.float32)
                # k_signals_rect = np.asarray(get_signals_rect(k_signals, signals_len=signal_len), dtype=np.float32)

                signal_means = signal_means.reshape(k_size, -1)
                signal_stds = signal_stds.reshape(k_size, -1)
                k_signal_lens = k_signal_lens.reshape(k_size, -1)
                k_seq_qual = k_seq_qual.reshape(k_size, -1)
                k_signals = k_signals.reshape(k_size, -1)

                signal_cat = np.concatenate((signal_means, signal_stds, k_signal_lens, k_seq_qual,
                                          k_signals), axis=1).reshape(k_size, 3 * down_sample_ratio + 4)

                if np.sum(np.isnan(signal_cat)) != 0 or np.sum(np.isnan(kmer_base)):
                    LOGGER.warning("Found NAN at h5 file {}, read {}, offloc {}".format(f.h5_file_name, f.read_id, off_loc))
                    continue

                site_info = f.read_id + "\t" + \
                    str(first_hit.r_st) + "\t" + \
                    str(first_hit.r_en) + "\t" + \
                    str(first_hit.ctg) + "\t" + \
                    str(abs_loc) + "\t" + str(strand_code)

                # if f.read_id == "035E29E6-0FC1-4541-BD63-403537636ACF" and abs_loc == 1255431:
                #     time.sleep(1)

                siteinfo_batch[fe_cnt % batch_size] = site_info
                kmer_batch[fe_cnt % batch_size] = kmer_base
                signal_batch[fe_cnt % batch_size] = signal_cat

                fe_cnt += 1
                if fe_cnt == batch_size:
                    fe_cnt = 0
                    dataQueue.put((siteinfo_batch, kmer_batch, signal_batch))
            except Exception as e:
                LOGGER.error("Error occured at h5 file {}, read {}, offloc {}".format(f.h5_file_name, f.read_id, off_loc))
                error_handler(e)

    if fe_cnt > 0:
        dataQueue.put((siteinfo_batch[:fe_cnt], kmer_batch[:fe_cnt], signal_batch[:fe_cnt]))
        # pass
    ed = time.time()

    LOGGER.info("Extract features for file {} end, total read {},"
                " selected read {},time const {} seconds".format(f.h5_file_name,
                                                                 cnt_total, cnt_selected,  ed - st))

if __name__ == "__main__":
    st = time.time()

    h5_dir = "/data1/YHC/QiTan_data/Fruitfly/YF6419_h5"
    fastq_dir = "/data1/YHC/QiTan_data/Fruitfly/YF6419_h5/Sturgeon/reads_splited/"
    ref_path = "/data1/YHC/QiTan_data/Fruitfly/FRUITFLY.hifiasm.diploid.fasta"
    result_file = "/data1/YHC/QiTan_data/Fruitfly/YF6419_h5/Sturgeon/mod_result.txt"
    module_path = "/data1/YHC/Mod_save/trace_scirpt_module_half_acc:0.9587.pt"

    # ref_genome = mappy.Aligner(ref_path)

    # dataQueue = multiprocessing.Queue(maxsize=100)

    manager = Manager()
    dataQueue = manager.Queue(maxsize=100)
    resQueue = manager.Queue(maxsize=100)

    motifs = ["CG"]
    f_list = os.listdir(h5_dir)
    h5_files = []
    for f in f_list:
        if os.path.splitext(f)[1] == ".h5":
            h5_files.append(f)

    LOGGER.info("Found {} h5 files".format(len(h5_files)))
    pool = Pool(24)

    np.random.seed(41)
    np.random.shuffle(h5_files)

    model_infe_p = multiprocessing.Process(target=module_inference_process, args=(dataQueue, resQueue, module_path))
    write_result_p = multiprocessing.Process(target=write_result_process, args=(resQueue, result_file))
    model_infe_p.start()
    write_result_p.start()
    for h5_f in h5_files[:80]:
        if len(h5_f.split(".")) == 2:
            # if h5_f != "reads_0062_370520B6-495E-40A9-A054-129AD9FDF16B.h5": continue
            h5_file = os.path.join(h5_dir, h5_f)
            fastq_file = os.path.join(fastq_dir ,h5_f.split(".")[0] + ".fastq")
            if not os.path.exists(fastq_file): continue
            pool.apply_async(extract_features, args=(
                fastq_file,
                h5_file,
                ref_path,
                dataQueue,
            ))
            # LOGGER.info("task submitted")
    pool.close()
    pool.join()


    # for h5_f in h5_files:
    #     if len(h5_f.split(".")) == 2:
    #         h5_file = h5_dir + h5_f
    #         fastq_file = fastq_dir + h5_f.split(".")[0] + ".fastq"
    #         if not os.path.exists(fastq_file): continue
    #         extract_features(
    #             fastq_file,
    #             h5_file,
    #             ref_path,
    #             dataQueue,
    #         )


    dataQueue.put((None, None, None))

    model_infe_p.join()
    write_result_p.join()

    LOGGER.info("Extract features end, "
                "total cost {} seconds".format(time.time() - st))