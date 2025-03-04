import mappy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import pysam
from utils.readers import *


if __name__ == "__main__":
    ref_path = "/data1/YHC/QiTan_data/ara/Sturgeon/ARA.SEQUELII.hifiasm.p_ctg.fasta"
    fastq_path = "/data1/YHC/QiTan_data/oryza/Sturgeon_2/test.fastq"
    ref_genome = mappy.Aligner(ref_path, preset="map-ont", n_threads=10, best_n=1)

    fastq_l = []
    for line in open(fastq_path):
        fastq_l.append(line)
    for i in tqdm(range(int(len(fastq_l) / 4))):
        info = fastq_l[4 * i]
        seq = fastq_l[4 * i + 1]
        c = fastq_l[4 * i + 2]
        qual = fastq_l[4 * i + 3]
        f = Fastq_Read(info, seq, qual)

        first_hit = next(ref_genome.map(f.seq), None)
        if first_hit is None: continue
        if first_hit.strand == -1:
            seq = complement_seq(seq)

        q_st = first_hit.q_st
        query_seq = seq[q_st:]


