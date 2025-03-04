import argparse
import time
import re
from tqdm import tqdm
from functools import cmp_to_key
import random
import os
# os.chdir("..")
# print(os.path.abspath(os.getcwd()))
import sys
sys.path.append(os.path.abspath('..'))
# from utils.utils_func import *
from utils.readers import *
LOGGER = get_logger(__name__)

def compare(x, y):
    if x.h5_file_name < y.h5_file_name:
        return -1
    elif x.h5_file_name > y.h5_file_name:
        return 1
    else:
        return 0

def sort_and_write(fastq_cache : list, write_file : str):
    sorted_fastq_list = sorted(fastq_cache, key=cmp_to_key(compare))
    f = open(write_file, 'w')
    for fastq in sorted_fastq_list:
        f.write("{}\n{}\n+\n{}\n".format(fastq.info,
                                         fastq.seq,
                                         fastq.qual))
    f.close()

# todo: implement a customized external merge sort
def split_chunks(fastq_path : str, save_dir : str, chunk_size : int = 1e5):
    LOGGER.info("Start to count lines")
    cnt = 0
    fastq_cache = []
    chunk_id = 0
    with open(fastq_path, 'r') as file:
        while True:
            lines = [file.readline().strip() for _ in range(4)]

            # Break if end of file
            if not any(lines) or len(lines) < 4:
                break
            info, seq, qual = lines[0], lines[1], lines[3]
            fastq = Fastq_Read(info, seq, qual)
            fastq_cache.append(fastq)
            cnt += 1
            if cnt % chunk_size == 0:
                # sorted_fastq_list = sorted(fastq_cache, key=cmp_to_key(compare))
                write_file = os.path.join(save_dir, "temp_{}.fastq".format(chunk_id))
                sort_and_write(fastq_cache, write_file)
                chunk_id += 1
                fastq_cache = []
    if len(fastq_cache) > 0:
        write_file = os.path.join(save_dir, "temp_{}.fastq".format(chunk_id))
        sort_and_write(fastq_cache, write_file)
        chunk_id += 1

    return chunk_id, cnt


def merge_two_chunks(chunk1 : int, chunk2 : int, target_chunk : int, save_dir : str, prefix : str = "temp_"):
    fastq_file1 = os.path.join(save_dir, "{}.fastq".format(prefix + str(chunk1)))
    fastq_file2 = os.path.join(save_dir, "{}.fastq".format(prefix + str(chunk2)))
    fastq_target = os.path.join(save_dir, "{}_.fastq".format(prefix + str(target_chunk)))
    fastq1, fastq2 = None, None
    f3 = open(fastq_target, 'w')
    with open(fastq_file1, 'r') as f1, open(fastq_file2, 'r') as f2:
        while True:
            if fastq1 is None: line1 = [f1.readline().strip() for _ in range(4)]
            if fastq2 is None: line2 = [f2.readline().strip() for _ in range(4)]
            if not any(line1) and not any(line2):
                break

            fastq1 = Fastq_Read(line1[0], line1[1], line1[3]) if any(line1) else None
            fastq2 = Fastq_Read(line2[0], line2[1], line2[3]) if any(line2) else None

            if fastq1 is None:
                f3.write("{}\n{}\n+\n{}\n".format(fastq2.info, fastq2.seq, fastq2.qual))
                fastq2 = None
            elif fastq2 is None:
                f3.write(
                    "{}\n{}\n+\n{}\n".format(fastq1.info, fastq1.seq, fastq1.qual))
                fastq1 = None
            elif fastq1.h5_file_name <= fastq2.h5_file_name:
                f3.write(
                    "{}\n{}\n+\n{}\n".format(fastq1.info, fastq1.seq, fastq1.qual))
                fastq1 = None
            else:
                f3.write(
                    "{}\n{}\n+\n{}\n".format(fastq2.info, fastq2.seq, fastq2.qual))
                fastq2 = None

    f1.close()
    f2.close()
    f3.close()
    os.remove(fastq_file1)
    os.remove(fastq_file2)
    os.rename(fastq_target, fastq_file1)

    return


def external_merge_sort(fastq_path : str, save_dir : str) -> None:
    st = time.time()
    num_chunk, pre_cnt = split_chunks(fastq_path, save_dir, chunk_size=1e5)
    LOGGER.info("splited {} chunks for external merge sort".format(num_chunk))
    while num_chunk > 1:
        for i in range(num_chunk // 2):
            merge_two_chunks(i, i + num_chunk // 2, i, save_dir)
        if num_chunk % 2 == 1:
            os.rename(os.path.join(save_dir, "temp_{}.fastq".format(num_chunk - 1)),
                      os.path.join(save_dir, "temp_{}.fastq".format(num_chunk // 2)))
            num_chunk = (num_chunk // 2) + 1
        else: num_chunk = (num_chunk // 2)
    final_chunk_path = os.path.join(save_dir, "temp_{}.fastq".format(0))

    read_cnt = 0
    fastq_list = []
    with open(final_chunk_path, 'r') as f:
        while True:
            lines = [f.readline().strip() for _ in range(4)]
            if not any(lines) or len(lines) < 4:
                break
            read_cnt += 1
            info, seq, qual = lines[0], lines[1], lines[3]
            fastq = Fastq_Read(info, seq, qual)

            if len(fastq_list) != 0 and fastq_list[-1].h5_file_name != fastq.h5_file_name:
                fw = open(os.path.join(save_dir, "{}.fastq".format(fastq_list[-1].h5_file_name.split(".")[0])), 'w')
                for fastq_ in fastq_list:
                    fw.write("{}\n{}\n+\n{}\n".format(fastq_.info, fastq_.seq, fastq_.qual))
                fw.close()
                fastq_list = []
            fastq_list.append(fastq)
    if len(fastq_list) != 0:
        fw = open(os.path.join(save_dir, "{}.fastq".format(fastq_list[-1].h5_file_name.split(".")[0])), 'w')
        for fastq_ in fastq_list:
            fw.write("{}\n{}\n+\n{}\n".format(fastq_.info, fastq_.seq, fastq_.qual))
        fw.close()
    os.remove(final_chunk_path)
    assert read_cnt == pre_cnt
    ed = time.time()
    LOGGER.info("External sort finished, cost {} seconds".format(ed - st))

def split_fastq(fastq_path : str, save_dir : str) -> None:

    st = time.time()
    LOGGER.info("Reading fastq files")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    read_cnt = 0

    fastq_list = []

    with open(fastq_path, 'r') as file:
        while True:
            # Read four lines
            lines = [file.readline().strip() for _ in range(4)]

            # Break if end of file
            if not any(lines) or len(lines) < 4:
                break
            read_cnt += 1
            info, seq, qual = lines[0], lines[1], lines[3]
            fastq = Fastq_Read(info, seq, qual)
            fastq_list.append(fastq)
    LOGGER.info("Read finished, start to sort")
    sorted_fastq_list = sorted(fastq_list, key=cmp_to_key(compare))

    LOGGER.info("Sort finished, begin to write split fastq")
    i, j = 0, 0
    f = open(os.path.join(save_dir, "{}.fastq".format(sorted_fastq_list[i].h5_file_name.split(".")[0])), 'w')
    while j < len(sorted_fastq_list):
        while j < len(sorted_fastq_list) and sorted_fastq_list[j].h5_file_name == sorted_fastq_list[i].h5_file_name:
            f.write("{}\n{}\n+\n{}\n".format(sorted_fastq_list[j].info,
                                             sorted_fastq_list[j].seq,
                                             sorted_fastq_list[j].qual))
            j += 1
        if j < len(sorted_fastq_list):
            i = j
            f.close()
            f = open(os.path.join(save_dir, "{}.fastq".format(sorted_fastq_list[i].h5_file_name.split(".")[0])), 'w')
    f.close()
    LOGGER.info("finished, parsed {} reads, cost {} seconds".format(read_cnt, time.time() - st))


def argparser():
    parser = argparse.ArgumentParser(description="split and sort fastq files by file name")
    parser.add_argument("--fastq_path", type=str, default="", help="path to fastq file")
    parser.add_argument("--save_dir", type=str, default="", help="directory to store split fastq files")
    parser.add_argument("--is_external", type=int, default=False, help="whether to use external merge sort")
    return parser

if __name__ == "__main__":

    args = argparser().parse_args()
    if args.is_external:
        LOGGER.info("Use external merge sort to reduce memory cost")
        external_merge_sort(fastq_path=args.fastq_path, save_dir=args.save_dir)
    else:
        split_fastq(fastq_path=args.fastq_path, save_dir=args.save_dir)


