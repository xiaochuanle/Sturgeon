import numpy as np
import logging
import random
from statsmodels import robust
import multiprocessing
import traceback
import copy
from models.decode_utils import accuracy



basepairs = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N',
             'W': 'W', 'S': 'S', 'M': 'K', 'K': 'M', 'R': 'Y',
             'Y': 'R', 'B': 'V', 'V': 'B', 'D': 'H', 'H': "D",
             'Z': 'Z'}
basepairs_rna = {'A': 'U', 'C': 'G', 'G': 'C', 'U': 'A', 'N': 'N',
                 'W': 'W', 'S': 'S', 'M': 'K', 'K': 'M', 'R': 'Y',
                 'Y': 'R', 'B': 'V', 'V': 'B', 'D': 'H', 'H': "D",
                 'Z': 'Z'}

base2code_dna = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4,
                 'W': 5, 'S': 6, 'M': 7, 'K': 8, 'R': 9,
                 'Y': 10, 'B': 11, 'V': 12, 'D': 13, 'H': 14,
                 'Z': 15}
code2base_dna = dict((v, k) for k, v in base2code_dna.items())
base2code_rna = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'N': 4,
                 'W': 5, 'S': 6, 'M': 7, 'K': 8, 'R': 9,
                 'Y': 10, 'B': 11, 'V': 12, 'D': 13, 'H': 14,
                 'Z': 15}
code2base_rna = dict((v, k) for k, v in base2code_rna.items())

iupac_alphabets = {'A': ['A'], 'T': ['T'], 'C': ['C'], 'G': ['G'],
                   'R': ['A', 'G'], 'M': ['A', 'C'], 'S': ['C', 'G'],
                   'Y': ['C', 'T'], 'K': ['G', 'T'], 'W': ['A', 'T'],
                   'B': ['C', 'G', 'T'], 'D': ['A', 'G', 'T'],
                   'H': ['A', 'C', 'T'], 'V': ['A', 'C', 'G'],
                   'N': ['A', 'C', 'G', 'T']}
iupac_alphabets_rna = {'A': ['A'], 'C': ['C'], 'G': ['G'], 'U': ['U'],
                       'R': ['A', 'G'], 'M': ['A', 'C'], 'S': ['C', 'G'],
                       'Y': ['C', 'U'], 'K': ['G', 'U'], 'W': ['A', 'U'],
                       'B': ['C', 'G', 'U'], 'D': ['A', 'G', 'U'],
                       'H': ['A', 'C', 'U'], 'V': ['A', 'C', 'G'],
                       'N': ['A', 'C', 'G', 'U']}

# logging functions ================================================================
# log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
log_datefmt = "%Y-%m-%d %H:%M:%S"
log_formatter = logging.Formatter(log_fmt, log_datefmt)
LOG_FN = '/tmp/QiTanBasecall.log'

def get_logger(module="", level=logging.INFO):
    logger = logging.getLogger(module)
    logger.setLevel(level)

    fh = logging.FileHandler(LOG_FN)
    fh.setLevel(level)
    fh.setFormatter(log_formatter)
    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(log_formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def get_mp_logger(level=logging.INFO):
    logger = multiprocessing.log_to_stderr()
    logger.setLevel(level)
    return logger


def error_handler(e):
    error_type = type(e).__name__
    tb = traceback.format_exc()
    print(f"Error type: {error_type}")
    print("Traceback details:")
    print(tb)
def _alphabet(letter, dbasepairs):
    if letter in dbasepairs.keys():
        return dbasepairs[letter]
    return 'N'
def complement_seq(base_seq, seq_type="DNA"):
    rbase_seq = base_seq[::-1]
    comseq = ''
    try:
        if seq_type == "DNA":
            comseq = ''.join([_alphabet(x, basepairs) for x in rbase_seq])
        elif seq_type == "RNA":
            comseq = ''.join([_alphabet(x, basepairs_rna) for x in rbase_seq])
        else:
            raise ValueError("the seq_type must be DNA or RNA")
    except Exception:
        print('something wrong in the dna/rna sequence.')
    return comseq


def generate_5mer_dict(default_mode = "encode"):
    keys = []
    alphabet = ['A', 'C', 'G', 'T']

    cur = ""
    def construct(cur, kmer_size):
        if len(cur) >= kmer_size:
            keys.append(cur)
            return
        for i in range(4):
            cur += alphabet[i]
            construct(cur, kmer_size)
            cur = cur[:-1]
        return

    construct(cur, 5)

    dict_5mer = {}
    if default_mode == "encode":
        for i in range(len(keys)):
            dict_5mer[keys[i]] = i + 1
    elif default_mode == "decode":
        for i in range(len(keys)):
            dict_5mer[i + 1] = keys[i]

    return dict_5mer

def get_normalize_factor(signals, normalize_method="mad"):
    signals = np.array(signals, dtype=np.float32)
    if normalize_method == 'zscore':
        sshift, sscale = np.mean(signals), float(np.std(signals))
    elif normalize_method == 'mad':
        sshift, sscale = np.median(signals), float(robust.mad(signals))
    else:
        raise ValueError("")
    # if sscale == 0.0:
    #     norm_signals = signals
    # else:
    #     norm_signals = (signals - sshift) / sscale
    # return np.around(norm_signals, decimals=6),
    return sshift, sscale

def group_signal_by_base2signal(signal, base2signal, trim_start, sig_len = 15, stride = 5):
    signal_group = np.zeros([len(base2signal), sig_len], dtype=np.float32)
    trim_sig = signal[trim_start:]
    sig_len_l = []
    for i in range(len(base2signal)):
        st = base2signal[i] - stride
        ed = st + sig_len
        if ed > len(trim_sig):
            st, ed = len(trim_sig) - sig_len, len(trim_sig)
        if st < 0:
            st, ed = 0, sig_len
        signal_group[i] = trim_sig[st : ed]
    for i in range(len(base2signal) - 1):
        sig_len_l.append(base2signal[i + 1] - base2signal[i])
    sig_len_l.append(len(trim_sig) - base2signal[-1])

    return signal_group, sig_len_l

def get_refloc_of_methysite_in_motif(ref_seq, motif_set : list, loc_in_motif : int):
    motifset = set(motif_set)
    strlen = len(ref_seq)
    motiflen = len(list(motifset)[0])
    sites = []
    for i in range(0, strlen - motiflen + 1):
        if ref_seq[i:i + motiflen] in motifset:
            sites.append(i + loc_in_motif)
    return sites

def trim_st(signal, window_size=40, threshold=2.4, min_trim=10, min_elements=3, max_samples=8000, max_trim=0.3):

    seen_peak = False
    num_windows = min(max_samples, len(signal)) // window_size

    for pos in range(num_windows):
        start = pos * window_size + min_trim
        end = start + window_size
        window = signal[start:end]
        if len(window[window > threshold]) > min_elements or seen_peak:
            seen_peak = True # 已经找到至少min_elements个元素，大于threshold
            if window[-1] > threshold:  # 窗口最后一个值不能大于threshold, 否则说明没有达到进入碱基的时间
                continue
            if end >= min(max_samples, len(signal)) or end / len(signal) > max_trim:
                # 截取的过多，返回默认最小截断值
                return min_trim
            return end

    return min_trim





def read_chunks(signal, chunksize=6000, overlap=600, stride=5):
    """
    Split a Read in fixed sized ReadChunks
    """
    if len(signal) < chunksize:
        return []

    size, offset = divmod(len(signal) - chunksize, chunksize - overlap)
    chunks = []

    st, ed = 0, chunksize
    while ed < len(signal):
        chunks.append({
           "sig" : copy.deepcopy(signal[st:ed]),
            "st" : st,
            "ed" : ed,
            "read_id" : "",
            "fn" : "",
            "sd" : 5,
        })
        st += chunksize - overlap
        ed += chunksize - overlap
    if offset > 0:
        _, trimed = divmod(offset, stride)
        st = len(signal) - chunksize - trimed
        ed = len(signal) - trimed
        chunks.append({
           "sig" : signal[st : ed],
            "st" : st,
            "ed" : ed,
            "read_id" : "",
            "fn": "",
            "sd": 5,
        })
    return chunks

def parse_cigar(r_cigar, strand, ref_len):
    fill_invalid = -1
    # get each base calls genomic position
    r_to_q_poss = np.full(ref_len + 1, fill_invalid, dtype=np.int32)
    # process cigar ops in read direction
    curr_r_pos, curr_q_pos = 0, 0
    cigar_ops = r_cigar if strand == 1 else r_cigar[::-1]
    for op_len, op in cigar_ops:
        if op == 1: # 1 : insertion to the reference
            curr_q_pos += op_len
        elif op in (2, 3):  # 2, 3 : delection from the reference, skipped region from the reference
            for r_pos in range(curr_r_pos, curr_r_pos + op_len):
                r_to_q_poss[r_pos] = curr_q_pos
            curr_r_pos += op_len
        elif op in (0, 7, 8): # 0 : alignment match(can be match or mismatch) ,
                              # 7 : sequence match
                              # 8 : sequence mismatch
            for op_offset in range(op_len):
                r_to_q_poss[curr_r_pos + op_offset] = curr_q_pos + op_offset
            curr_q_pos += op_len
            curr_r_pos += op_len
        elif op == 6: # 6 : padding (slient delection from padded reference
            # padding (shouldn't happen in mappy)
            pass
    r_to_q_poss[curr_r_pos] = curr_q_pos
    if r_to_q_poss[-1] == fill_invalid:
        raise ValueError((
            'Invalid cigar string encountered. Reference length: {}  Cigar ' +
            'implied reference length: {}').format(ref_len, curr_r_pos))

    return r_to_q_poss

def group_sigal_locs_by_movetable(base2signal, sig_len , stride = 6, stim_st=200):
    signal_locs = []
    for i in range(len(base2signal) - 1):
        signal_locs.append((base2signal[i], base2signal[i + 1]))
    signal_locs.append((base2signal[-1], sig_len))
    return signal_locs

def get_ref_seq_for_chunk(chunks, base2signal, read_ref_locs, ref_seq, query_seq, filter_acc = 90):
    """
    chunks is a list of tuples:
        (signal_chunk,
        start_from_signal,
        end_from_signal,
        seq_waited_to_be_added_in)
    """

    chunks_ = []
    for chunk in chunks:
        st = chunk['st']
        ed = chunk['ed']
        if ed > base2signal[-1]: continue
        st_idx = np.searchsorted(base2signal, st, side='right')
        ed_idx = np.searchsorted(base2signal, ed, side='left')
        ref_st = read_ref_locs[st_idx] if st_idx < len(read_ref_locs) else read_ref_locs[-1]
        if (ed_idx > len(read_ref_locs)):
            print("Found it")
        ref_ed = read_ref_locs[ed_idx] if ed_idx < len(read_ref_locs) else read_ref_locs[-1]

        if ref_st < 0  or ref_ed < 0 or ref_ed <= ref_st or ref_ed - ref_st > 1000: continue
        curr_ref_seq = ref_seq[ref_st : ref_ed ]
        curr_query_seq = query_seq[st_idx : ed_idx]
        curr_acc = accuracy(curr_ref_seq, curr_query_seq, min_coverage=0.95)
        if curr_acc <= filter_acc:
            continue
        chunk['seq'] = curr_ref_seq
        chunks_.append(chunk)
    return chunks_


def typical_indices(x, n = 2.5):
    """
    Remove sequences that are too long or too short
    """
    mu, sd = np.mean(x), np.std(x)
    idx, = np.where((mu - n*sd < x) & (x < mu + n*sd))
    return idx

