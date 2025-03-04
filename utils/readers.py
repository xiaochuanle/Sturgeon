import numpy as np
from .utils_func import *
import h5py
import os

class Fastq_Read:
    def __init__(self, info, seq, qual, moves: bool = False):
        self.info = info.strip()
        info_s = self.info.split("\t")
        self.read_id = info_s[0][1 : 37]
        self.q_score = float(info_s[1][5:])
        # self.signal_used = int(info_s[2][5:])
        # self.signal_len = int(info_s[1][5:])
        self.trimed_start = int(info_s[2][5:])
        self.h5_file_name = info_s[3][5:]
        self.stride = int(info_s[4][5:])
        if moves:
            self.base2signal_str = info_s[5][len("base2signal:Z:"):]
        self.seq = seq.strip()
        self.qual = qual.strip()
    def get_base2signal(self):
        base2signal = np.array([int(x) for x in self.base2signal_str.split(",")], dtype=np.int32)
        assert  (len(base2signal) == len(self.seq))
        # return base2signal[ : np.sum(base2signal <= self.signal_len)] # only return the useful part
        return base2signal

    def get_motifs_loc(self, motif_set : list, loc_in_motif : int):
        motifset = set(motif_set)
        strlen = len(self.seq)
        motiflen = len(list(motifset)[0])
        sites = []
        for i in range(0, strlen - motiflen + 1):
            if self.seq[i:i + motiflen] in motifset:
                sites.append(i + loc_in_motif)
        return sites

    def get_refloc_of_methysite_in_motif(self, ref_seq, motif_set : list, loc_in_motif : int):
        motifset = set(motif_set)
        strlen = len(ref_seq)
        motiflen = len(list(motifset)[0])
        sites = []
        for i in range(0, strlen - motiflen + 1):
            if ref_seq[i:i + motiflen] in motifset:
                sites.append(i + loc_in_motif)
        return sites
    def get_seq_quality(self):
        return np.array([float(ord(x)) / 12. for x in self.qual], dtype=np.float32)

    def get_base_code(self):
        return np.array([base2code_dna[x] for x in self.seq], dtype=np.int32)




class H5_Reads:
    def __init__(self, h5_path : str):
        self.reads = h5py.File(h5_path, "r")
        self.h5_path = h5_path
        self.file_name = os.path.basename(h5_path)
        self.read_ids = list(self.reads["Raw_data"].keys())
        self.read_cnt = len(self.read_ids)



    def get_read(self, read_id : str):
        if read_id not in self.read_ids:
            return None
        return H5_Read(self.reads["Raw_data"][read_id][()], read_id, self.file_name)

    def __iter__(self):
        for read_id in self.read_ids:
            yield self.get_read(read_id)


class H5_Read:
    def __init__(self, signal, read_id, fn):

        """
        class to handle specific h5 read
        The first and last 200 sampling points will be removed in QiTanTech Basecalling
        """

        self.signal = signal
        self.read_id = read_id
        self.sshift, self.sscale = get_normalize_factor(self.signal)
        self.file_name = fn

        self.trim_start = trim_st(self.signal, threshold=self.sscale * 2.0 + self.sshift)

        self.normed_sig = (self.signal - self.sshift) / self.sscale


    def split_chunks(self, chunk_size=6000, overlap=600, is_auto_trim : bool = False):

        if is_auto_trim:
            chunk_curr = read_chunks(self.normed_sig[self.trim_start:-200], chunk_size, overlap)
        else: # default trim ?
            chunk_curr = read_chunks((self.signal[200:-200] - self.sshift) / self.sscale, chunk_size, overlap)

        for itr in chunk_curr:
            itr["read_id"] = self.read_id
            itr['ts'] = self.trim_start
            itr['fn'] = self.file_name
        return chunk_curr

