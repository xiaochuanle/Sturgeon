import multiprocessing

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
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


def data_writer(dataQueue : multiprocessing.Queue, output_dir : str, stop_event):
    sig_chunks = []
    target_chunks_ = []
    target_lens = []
    base_dict = {"N" : 0, "A" : 1, "C" : 2, "G" : 3, "T" : 4}
    # dict_5mer = generate_5mer_dict()
    cnt = 0
    while True:
        chunks = dataQueue.get()
        if chunks is None: break
        try:
            for chunk in chunks:
                sig_chunks.append(np.array(chunk['sig'], dtype=np.float32))
                target_chunks_.append(
                    np.array(
                        [base_dict[base] for base in chunk['seq']  if base != 0],
                        dtype=np.uint8
                    )
                )
                target_lens.append(len(chunk['seq']))
                cnt += 1
                if cnt % 100000 == 0:
                    LOGGER.info("Received 100000 chunks!")
        except Exception as e:
            LOGGER.error("Error occurred while merging data!")
            error_handler(e)
        if cnt >= 1500000:
            LOGGER.info("Received total {} chunks of data, shut down the producer and write result".format(cnt))
            stop_event.set()
            break
    try:
        sig_chunks = np.array(sig_chunks, dtype=np.float32)
        target_chunks = np.zeros((sig_chunks.shape[0], np.max(target_lens)), dtype=np.uint8)
        for idx, target in enumerate(target_chunks_): target_chunks[idx, :len(target)] = target
        target_lens = np.array(target_lens, dtype=np.uint16)

        indices = np.random.permutation(typical_indices(target_lens))

        sig_chunks = sig_chunks[indices]
        target_chunks = target_chunks[indices]
        target_lens = target_lens[indices]
    except Exception as e:
        LOGGER.error("Error occurred while processing merged chunks!")
        error_handler(e)

    np.save(os.path.join(output_dir, "chunks.npy"), sig_chunks)
    np.save(os.path.join(output_dir, "references.npy"), target_chunks)
    np.save(os.path.join(output_dir, "reference_lengths.npy"), target_lens)
    LOGGER.info("Totally written {} chunks of ctc training data".format(sig_chunks.shape[0]))


def extract_features(fastq_path : str,
                     h5_path : str,
                     ref_path : str,
                     dataQueue ,
                     stop_event ,
                     chunkSize : int = 3000,
                     overlap : int = 500,
                     ):

    st = time.time()
    if stop_event.is_set():
        return
    # LOGGER.info("Extract ctc training data from {} and {} start".format(fastq_path.split("/")[-1], h5_path.split("/")[-1]))
    ref_genome = mappy.Aligner(ref_path)
    # h5_reads = h5py.File(h5_path, 'r')
    h5_reads = H5_Reads(h5_path)

    fastq_l = []
    for line in open(fastq_path):
        fastq_l.append(line)

    cnt_total, cnt_selected = 0, 0
    fe_cnt = 0
    total_chunks = []
    for i in range(int(len(fastq_l) / 4)):
        try:
            info = fastq_l[4 * i]
            seq = fastq_l[4 * i + 1]
            c = fastq_l[4 * i + 2]
            qual = fastq_l[4 * i + 3]
            f = Fastq_Read(info, seq, qual, True)

            # if (f.read_id == "507AA7A9-4845-4F12-9340-87472A199140") :
            #     print("1")

            # filter by some condition
            # if f.signal_len <= 6400: continue
            first_hit = next(ref_genome.map(f.seq), None) # get mapping info
            if first_hit is None: continue
            if f.q_score <= 15: continue
            if first_hit.mapq <= 20: continue
            # if first_hit.mlen / (first_hit.r_en - first_hit.r_st) <= 0.70: continue

            read = h5_reads.get_read(f.read_id)
            if len(read.signal) <= 3 * chunkSize: continue

            read_chunks = read.split_chunks(chunkSize, overlap)

            cnt_total += 1


            ref_seq = ref_genome.seq(first_hit.ctg)[first_hit.r_st : first_hit.r_en].upper()
            query_seq = f.seq[first_hit.q_st: first_hit.q_en]
            if first_hit.strand == -1:
                ref_seq = complement_seq(ref_seq)
            strand_code = 1 if first_hit.strand == 1 else 0

            base2signal = f.get_base2signal()
            assert len(f.seq) == len(base2signal)
            r_to_q_poss = parse_cigar(first_hit.cigar, strand_code, first_hit.r_en - first_hit.r_st)
            cnt_selected += 1

            read_ref_locs = [-1,] * len(f.seq)

            q_pos_pre = first_hit.q_st
            for ref_pos, q_pos in enumerate(r_to_q_poss[:-1]):
                curr_q_pos = q_pos + first_hit.q_st
                if curr_q_pos - q_pos_pre > 1:
                    read_ref_locs[q_pos_pre + 1 : curr_q_pos] = [ref_pos,] * (curr_q_pos - q_pos_pre - 1)
                read_ref_locs[curr_q_pos] = ref_pos
                q_pos_pre = curr_q_pos

            read_chunks = get_ref_seq_for_chunk(read_chunks, base2signal, read_ref_locs, ref_seq, f.seq)
            read_chunks = [x for x in read_chunks if np.sum(np.abs(x['sig']) > 2.3) == 0]
            if len(read_chunks) > 0:
                total_chunks += read_chunks
        except Exception as e:
            LOGGER.error("Error occured at h5 file {}, read {}".format(f.h5_file_name, f.read_id))
            error_handler(e)
        if stop_event.is_set():
            total_chunks = []
            break

    if len(total_chunks) > 0 and not stop_event.is_set():
        dataQueue.put(
            total_chunks
        )
    ed = time.time()

    # LOGGER.info("Extract ctc training data for file {} end, total read {},"
    #             " extracted chunk {},time const {} seconds".format(f.h5_file_name,
    #                                                              cnt_total, len(total_chunks),  ed - st))


def argparser():
    parser = argparse.ArgumentParser(description="generate CTC training data for QiTan nanopore data.")

    parser.add_argument("h5_dir", type=str,
                        help="directory which stores h5 data")
    parser.add_argument("fastq_dir", type=str,
                        help="directory which stores basecalling fastq data")
    parser.add_argument("ref_path", type=str,
                        help="path to reference genome")
    parser.add_argument("output_dir", type=str, default="./",
                        help="directory to store ctc training data")
    parser.add_argument("--num_proc", type=int, default=48,
                        help="number of processes to extract ctc training data")
    parser.add_argument("--chunk_size", type=int, default=6000,
                        help="chunk size for ctc training data")
    parser.add_argument("--overlap", type=int, default=600,
                        help="overlap size for ctc training data")


    return parser

if __name__ == "__main__":
    st = time.time()
    args = argparser().parse_args()

    manager = Manager()
    dataQueue = manager.Queue(maxsize=100)
    stop_event = manager.Event()

    h5_files = [x for x in os.listdir(args.h5_dir) if x.endswith(".h5")]
    h5_f_dict = { x : os.path.join(args.h5_dir, x) for x in h5_files}

    LOGGER.info("Total h5 files: {}".format(len(h5_files)))
    # np.random.seed(1)
    np.random.shuffle(h5_files)

    num_file = min(1000, len(h5_files))

    LOGGER.info("randomly selected {} files to extract ctc data".format(num_file))

    pool = Pool(processes=args.num_proc)
    writer = multiprocessing.Process(target=data_writer, args=(dataQueue, args.output_dir, stop_event))
    writer.start()
    for i in range(num_file):
        if stop_event.is_set() : break
        time.sleep(0.1)
        fastq_path = os.path.join(args.fastq_dir, h5_files[i].split(".")[0] + ".fastq")
        h5_path = h5_f_dict[h5_files[i]]
        if not os.path.exists(fastq_path) or not os.path.exists(h5_path): continue
            # LOGGER.info("{} and {} exist".format(fastq_path, h5_path))
        pool.apply_async(extract_features, args = (
                         fastq_path,
                         h5_path,
                         args.ref_path,
                         dataQueue,
                         stop_event,
                            args.chunk_size,
                            args.overlap,)
                         )
    pool.close()
    pool.join()

    dataQueue.put(None)

    # for i in range(1):
    #     h5_file = "reads_0005_A055F5D1-C4A2-4823-AD72-319A3BFBB998.h5"
    #     fastq_path = os.path.join(args.fastq_dir, h5_file.split(".")[0] + ".fastq")
    #     h5_path = h5_f_dict[h5_file]
    #     extract_features(fastq_path,
    #                      h5_path,
    #                      args.ref_path,
    #                      dataQueue)
    # dataQueue.put(None)
    writer.join()

    LOGGER.info("Extract CTC data end, "
                "total cost {} seconds".format(time.time() - st))