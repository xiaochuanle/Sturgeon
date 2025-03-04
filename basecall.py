import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
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
from utils.basecall_utils import *
from models.decode_utils import *
from models.model import CRF_encoder

LOGGER = get_logger(__name__)



def get_chunk_process(h5_directory : str,
                      chunks : multiprocessing.Queue,
                      chunk_size : int = 6000,
                      overlap : int = 600,
                      num_proc : int = 4,
                      batch_size : int = 128,
                      seed : int = 41):
    st = time.time()
    h5_files = [x for x in os.listdir(h5_directory) if x.endswith(".h5")]
    h5_f_dict = {x : os.path.join(h5_directory, x) for x in h5_files}

    LOGGER.info("Found {} h5 files totally!".format(len(h5_files)))
    np.random.seed(seed)
    np.random.shuffle(h5_files)

    pool = Pool(processes=num_proc)
    for h5_file in h5_files[:500]:
        if h5_file != "reads_0050_3A9F1EA1-726C-407C-B3C4-0D743B1A046F.h5": continue
        h5_file_path = h5_f_dict[h5_file]
        pool.apply_async(get_chunk_subprocess, args=(h5_file_path,
                                                     chunks,
                                                     chunk_size,
                                                     overlap,
                                                     batch_size)
                         )

    pool.close()
    pool.join()
    chunks.put(None)
    chunks.put(None)
    LOGGER.info("get basecall chunks process ended")



def inference_process(model_path : str,
                      data_chunks,
                      decode_chunks,
                      batch_size : int = 128,
                      is_half : bool = True,
                      num_proc_decode : int = 4,
                      device_id : int = 0,
                      ):
    device = torch.device("cuda:{}".format(device_id))
    LOGGER.info("Runing with {}".format(device))
    try:
        model = CTC_encoder(n_hid=512)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        if is_half: model.half()
        model.eval()
    except Exception as e:
        LOGGER.error("error occurred in loading model, info-{}".format(e))

    cnt = 0
    with torch.inference_mode():
        while True:
            # try:
            chunk = data_chunks.get()

            if chunk is None: break
            batch = np.array([x["sig"] for x in chunk], dtype=np.float16 if is_half else np.float32)
            if len(batch) >  batch_size:
                batch_list = [batch[i:i+batch_size] for i in range(0, len(batch), batch_size)]
                batch_tensors = [torch.from_numpy(x).reshape(x.shape[0], 1, x.shape[1]) for x in batch_list]
                inputs = []
                for batch_tensor in batch_tensors:
                    input = model(batch_tensor.contiguous().to(device))
                    inputs.append(input.cpu())

                inputs = torch.cat(inputs, dim=1)
            elif len(batch) > 0:
                # print(len(batch))
                batch = torch.from_numpy(batch).reshape(batch.shape[0], 1, batch.shape[1])
                batch = batch.contiguous().to(device)
                inputs = model(batch)
            decode_chunks.put((chunk, inputs.detach().cpu().numpy()))
            cnt += 1
            # print("model infered batch-{}".format(cnt))
        # except:
        #     LOGGER.error("error occurred in model inference process")
    if device_id == 0:
        for i in range(num_proc_decode):
            decode_chunks.put([None,None])
    LOGGER.info("basecalling process-{} ended".format(device_id))
    return


def decode_process(decode_chunks : multiprocessing.Queue,
                   stich_chunks : multiprocessing.Queue,
                   num_proc : int = 1,
                   num_proc_stich : int = 1,
                   ):
    decode_procs = []
    for i in range(num_proc):
        decode_procs.append(
            multiprocessing.Process(target=decode_subprocess,
                                    args=(decode_chunks,
                                          stich_chunks,
                                          )
                                    )
        )

    for i in range(num_proc): decode_procs[i].start()
    for i in range(num_proc): decode_procs[i].join()

    for i in range(num_proc_stich):
        stich_chunks.put([None, None])

    LOGGER.info("decode process ended")




def stich_results_process(stich_chunks : multiprocessing.Queue,
                          res_chunks : multiprocessing.Queue,
                          mod_chunks : multiprocessing.Queue,
                          num_proc : int = 4,
                          chunk_size : int = 6000,
                          overlap : int = 600,
                          stride : int = 5,
                          ):
    stich_procs = []
    for i in range(num_proc):
        stich_procs.append(
            multiprocessing.Process(target=stich_result_subprocess,
                                    args=(
                                          stich_chunks,
                                          res_chunks,
                                          mod_chunks,
                                          chunk_size,
                                          overlap,
                                          stride,
                                          )
                                    )
        )

    for i in range(num_proc): stich_procs[i].start()
    for i in range(num_proc): stich_procs[i].join()

    if mod_chunks is not None:
        for i in range(16): mod_chunks.put(None)

    res_chunks.put(None)

    LOGGER.info("stitch result process finished")
    # mod_chunks.put(None)



def writer_process(write_dir : str,
                   res_chunks ,
                   filter_Q : float = 10.0,
                   output_moves : bool = False,
                   ):
    f_pass = open(os.path.join(write_dir,"pass.fastq"), 'w')
    f_fail = open(os.path.join(write_dir, "fail.fastq"), 'w')

    cnt_pass = 0
    cnt_fail = 0
    while True:
        result_write = res_chunks.get()
        if result_write is None:
            # print("received end signal, stop writing")
            break

        header = "@" + result_write["read_id"] + "\t" \
            + "qs:i:" + str(int(result_write['filter_Q'])) + "\t" \
            + "ts:i:" + str(result_write['ts']) + "\t" \
            + "fn:Z:" + result_write['fn'] + "\t" \
            + "sd:i:" + str(result_write['sd']) + "\t"

        # if result_write['read_id'] == "035E29E6-0FC1-4541-BD63-403537636ACF":
        #     print('received read first time')
        if output_moves:
            write_str = header \
                    + "base2signal:Z:" + result_write['moves'] + "\n" \
                    + result_write["sequence"] + "\n" \
                    + "+" + "\n" \
                    + result_write["qstring"] + "\n"
        else :
            write_str = header + "\n" \
                    + result_write["sequence"] + "\n" \
                    + "+" + "\n" \
                    + result_write["qstring"] + "\n"
        if result_write['filter_Q'] < filter_Q:
            cnt_fail += 1
            f_fail.write(write_str)
        else:
            cnt_pass += 1
            f_pass.write(write_str)
        if (cnt_pass + cnt_fail) % 100000 == 0:
            LOGGER.info("wrote {} reads".format(cnt_pass + cnt_fail))
            # cnt = 0
        # print("writer wrote result of a read: {}".format(result_write["read_id"]))
        # i = j
    f_pass.close()
    f_fail.close()
    LOGGER.info("write process finished, totally wrote {} reads, passed {} reads, failed {} reads".format(cnt_pass + cnt_fail, cnt_pass, cnt_fail))

def call_modification(
        mod_chunks: multiprocessing.Queue,
        dataQueue : multiprocessing.Queue,
        resQueue : multiprocessing.Queue,
        ref_path : str,
        module_path : str,
        write_path : str,
        batch_size: int = 2048,
        kmer_size : int = 21,
        sig_len : int = 15,
        is_half : bool = True,
):
    device_count = torch.cuda.device_count()
    mod_workers = []
    for i in range(16):
        mod_workers.append(
            multiprocessing.Process(target=mod_feature_extractor,
                                    args=(mod_chunks, dataQueue, ref_path, batch_size, kmer_size, sig_len),
                                    name="feature extract proc-{}".format(i)))
    model_infer_p = []
    for device_id in range(device_count):
        model_infer_p.append(
            multiprocessing.Process(target=integrated_call_mods,
                                    args=(dataQueue, resQueue, module_path, kmer_size, device_id, is_half),
                                    name="modification infer proc-{}".format(device_id)))
    # model_infer_p = multiprocessing.Process(target=integrated_call_mods, args=(dataQueue, resQueue, module_path, kmer_size), name="modification calling")
    writer_p = multiprocessing.Process(target=write_mod_result, args=(resQueue, write_path), name="modification result write")
    for w in mod_workers: w.start()
    for proc in model_infer_p: proc.start()
    writer_p.start()
    for w in mod_workers: w.join()
    for proc in model_infer_p:
        dataQueue.put((None, None, None))
    writer_p.join()
    for proc in model_infer_p:
        proc.terminate()
        proc.join()
    LOGGER.info("Modification calling finished")

class Basecall_Worker:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.init_queues()
        self.detect_cuda_devices()
        self.init_worker_process()

    def detect_cuda_devices(self,):
        if torch.cuda.is_available():
            self.device_count = torch.cuda.device_count()
            LOGGER.info(f"detected num of CUDA devices: {self.device_count}")
        else:
            LOGGER.info("CUDA is not available.")
            exit(1)

    def init_queues(self,):
        self.data_chunks = Manager().Queue(maxsize=10)
        self.decode_chunks = multiprocessing.Queue(maxsize=10)
        self.stitch_chunks = multiprocessing.Queue(maxsize=10)
        self.res_chunks = multiprocessing.Queue(maxsize=10)

        # will not put any data if call mods is set with false
        self.mod_chunks = multiprocessing.Queue(maxsize=10) if self.call_mods else None
        self.mod_dataQueue = multiprocessing.Queue(maxsize=10) if self.call_mods else None
        self.mod_resQueue = multiprocessing.Queue(maxsize=10) if self.call_mods else None

    def init_worker_process(self,):
        self.get_chunk_module = multiprocessing.Process(target=get_chunk_process,
                                                        args=(self.h5_directory,
                                                              self.data_chunks,
                                                              self.chunk_size,
                                                              self.overlap,
                                                              self.num_get_chunk,
                                                              self.batch_size,
                                                              self.seed,
                                                              ),
                                                        name="get basecall chunk")
        self.basecall_module = []
        for device_id in range(self.device_count):
            self.basecall_module.append(
                multiprocessing.Process(target=inference_process,
                                        args=(self.model_path,
                                              self.data_chunks,
                                              self.decode_chunks,
                                              self.batch_size,
                                              self.is_half,
                                              self.num_decode,
                                              device_id,
                                              ),
                                        name="inference proc-{}".format(device_id),
                                        )
            )
        self.decode_module = multiprocessing.Process(target=decode_process,
                                                     args=(self.decode_chunks,
                                                           self.stitch_chunks,
                                                           self.num_decode,
                                                           self.num_stitch,
                                                           ),
                                                     name="decode procs group")
        self.stitch_module = multiprocessing.Process(target=stich_results_process,
                                                     args=(self.stitch_chunks,
                                                           self.res_chunks,
                                                           self.mod_chunks if args.call_mods else None,
                                                           self.num_stitch,
                                                           self.chunk_size,
                                                           self.overlap,
                                                           self.stride,
                                                           ),
                                                     name="Stitch result procs group")

        if self.call_mods:
            self.mod_module = multiprocessing.Process(
                target=call_modification,
                args=(
                    self.mod_chunks,
                    self.mod_dataQueue,
                    self.mod_resQueue,
                    self.ref_genome,
                    self.mod_module_path,
                    self.mod_result_path,
                    self.mod_batch_size,
                    self.kmer_size,
                    self.sig_len,
                    self.is_half,
                ),
                name="call mods module")

        self.Fastq_writer = multiprocessing.Process(target=writer_process,
                                                    args=(self.output_dir,
                                                          self.res_chunks,
                                                          self.filter_q,
                                                          self.output_moves,
                                                          ), name="Fastq Writer")

    def _start(self, ):
        self.get_chunk_module.start()
        for proc in self.basecall_module:
            proc.start()
        self.decode_module.start()
        self.stitch_module.start()
        if args.call_mods: self.mod_module.start()
        self.Fastq_writer.start()

    def _join(self, ):
        self.get_chunk_module.join()
        self.decode_module.join()
        self.stitch_module.join()
        self.Fastq_writer.join()
        if args.call_mods: self.mod_module.join()
        for proc in self.basecall_module:
            if proc.is_alive():
                proc.terminate()
                proc.join()

    def run(self,):
        st = time.time()
        self._start()
        self._join()
        ed = time.time()
        return ed - st
def basecall(args : argparse.Namespace):
    st = time.time()
    worker = Basecall_Worker(**vars(args))
    worker.run()
    ed = time.time()
    LOGGER.info("Basecalling for data: {} finished, cost {} seconds".format(args.h5_directory , ed - st))


def argparser():
    parser = argparse.ArgumentParser(description="A basecaller for QiTan nanopore data.")


    parser.add_argument("model_path", type=str,
                        help="path to saved models")
    parser.add_argument("h5_directory", type=str,
                        help="path to h5 electric signals"
                        )
    parser.add_argument("-o", "--output_dir", type=str, default=".",
                        help="path to output directory")
    # parser.add_argument("-r", "--ref_path", type=str, default=None,
    #                     help="path to reference genome")
    parser.add_argument("--num_get_chunk", type=int, default=12, help="process used for extract chunks")
    parser.add_argument("--num_decode", type=int, default=4, help="process used for decode inputs")
    parser.add_argument("--num_stitch", type=int, default=12, help="process used for stich results")
    parser.add_argument("--is_half", type=bool, default=True, help="run model inference"
                                                                   " in half precision")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size to "
                                                                    "basecall while running basecalling")
    parser.add_argument("--chunk_size", type=int, default=6000, help="size of chunks")
    parser.add_argument("--overlap", type=int, default=600, help="overlap size of chunks")
    parser.add_argument("--stride", type=int, default=5, help="stride size of chunks")
    parser.add_argument("--filter_q", type=float, default=10.0, help="filter reads with Q score lower than this value")
    parser.add_argument("--output_moves", action="store_true",
                        help="output moves for align raw signal and basecalling seq")
    parser.add_argument("--call_mods", action="store_true",
                        help="output moves for align raw signal and basecalling seq")
    parser.add_argument("--ref_genome", type=str, default=None,
                        help="path to reference genome, needed for calling modification")
    parser.add_argument("--mod_module_path", type=str, default=None,
                        help="path to module for calling modification")
    parser.add_argument("--mod_result_path", type=str, default=None,
                        help="path to write modification result")
    parser.add_argument("--mod_batch_size", type=int, default=1024,
                        help="batch size for calling modification data")
    parser.add_argument("--kmer_size", type=int, default=21, help="kmer size for call modification")
    parser.add_argument("--sig_len", type=int, default=15, help="signal sliced from basecall signal length")
    parser.add_argument("--seed", type=int, default=41, help="random seed")

    return parser


if __name__ == "__main__":
    parser = argparser()

    args = parser.parse_args()

    basecall(args)