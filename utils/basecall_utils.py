# import multiprocessing
import numpy as np

from utils.readers import *
# from utils.basecall_utils import *
from models.decode_utils import *
from cpp_components.ctc_decoder import vertibi_decode_fast
import random
import copy
import time
import mappy

LOGGER = get_logger(__name__)
def get_chunk_subprocess(h5_file_path,
                         chunks : multiprocessing.Queue,
                         chunk_size : int = 6000,
                         overlap : int = 600,
                         batch_size : int = 128):

    h5_reads = H5_Reads(h5_path=h5_file_path)
    chunk_batch = []
    cnt = 0

    # random_filter = 0.2

    for h5_read in h5_reads:
        # print(h5_read.read_id)
        chunk_curr = h5_read.split_chunks(chunk_size=chunk_size, overlap=overlap, is_auto_trim=True)

            # itr['fn'] = h5_read.fi
        # if random.random() > random_filter:
        #     continue
        chunk_batch += chunk_curr
        if len(chunk_batch) >= batch_size:
            cnt += 1
            chunks.put(chunk_batch)
            # print("send batch-{}".format(cnt))
            chunk_batch = []
    if len(chunk_batch) > 0:
        chunks.put(chunk_batch)

    LOGGER.info("basecalling for h5 file: {} finished".format(h5_file_path))

def decode_subprocess(decode_chunks,
                      stich_chunks,
                     ):
    cnt = 0
    while True:
        chunks, inputs = decode_chunks.get()
        if chunks is None:
            # print("received none sig, stop decode")
            break
        # for chunk in chunks: del chunk['sig']
        try:
            seqs, moves, qstrings = vertibi_decode_fast(inputs)
        except Exception as e:
            LOGGER.error("error occured during viterbi_decode, info: {}".format(e))
        stich_chunks.put((chunks, {"sequence" : seqs,
                                "qstring": qstrings,
                                "moves" : moves ,
                                }))
        cnt += 1
        # print("decoded batch-{} stitch".format(cnt))
    return


def stich_result(chunks : list,
                 moves : np.array,
                 qstrings : np.array,
                 seqs : np.array,
                 chunk_size : int = 6000,
                 overlap : int = 600,
                 stride : int = 5,
                 ):
    # if (chunks[0]['read_id'] == "035E29E6-0FC1-4541-BD63-403537636ACF"):
    #     LOGGER.info("find it")

    if len(seqs) >= 2:
        semi_overlap = overlap // 2
        start, end = semi_overlap // stride, (chunk_size - semi_overlap) // stride
        last_st = (chunks[-2]["ed"] - chunks[-1]['st'] - semi_overlap) // stride

        seq_ = np.concatenate([seqs[0][:end],
                               *[seq[start:end] for seq in seqs[1:-1]],
                               seqs[-1][last_st:]])
        qstring_ = np.concatenate([qstrings[0][:end],
                                   *[qstring[start:end] for qstring in qstrings[1:-1]],
                                   qstrings[-1][last_st:]])
        move_ = np.concatenate([moves[0][:end],
                                *[move[start:end] for move in moves[1:-1]],
                                moves[-1][last_st:]])
        signal = np.concatenate([chunks[0]['sig'][:end * stride],
                                *[chunk['sig'][start * stride : end * stride] for chunk in chunks[1:-1]],
                                chunks[-1]['sig'][last_st * stride:]
                                ])

    else:
        seq_ = seqs[0]
        qstring_ = qstrings[0]
        move_ = moves[0]
        signal = chunks[0]['sig']

    seq_ = "".join([alphabet[x] for x in seq_ if x != 0])

    q_arr = np.array([x for x in qstring_ if x != 0])
    mean_q = (np.mean(q_arr) - 33) if len(q_arr) else 0.0
    qstring_ = "".join([chr(x) for x in q_arr])

    move_ = np.argwhere(move_ == 1)[:, 0] * stride

    assert (len(seq_) == len(qstring_) == len(move_))

    move_str = ",".join([str(x) for x in move_])

    result_fastq = {
        "read_id": chunks[0]["read_id"],
        'ts' : chunks[0]['ts'],
        'fn' : chunks[0]['fn'],
        "moves": move_str,
        "qstring": qstring_,
        "sequence": seq_,
        "filter_Q" : mean_q,
        "sd" : chunks[0]['sd'],
        # "signal" : signal,
    }

    result_mod = {
        "read_id" : chunks[0]["read_id"],
        'moves': move_,
        "q_arr" : q_arr,
        "sequence" : seq_,
        "filter_Q" : mean_q,
        "signal" : signal,
        "sd" : chunks[0]['sd'],
        'fn': chunks[0]['fn'],
    }

    return result_fastq, result_mod


def stich_result_subprocess(
                 stich_chunk : multiprocessing.Queue,
                 res_chunks : multiprocessing.Queue,
                 mod_chunks : multiprocessing.Queue,
                 chunk_size : int = 6000,
                 overlap : int = 600,
                 stride : int = 5,
                 ):
    while True:
        chunks, results = stich_chunk.get()
        # print("get a decode result")
        if chunks is None:
            # print("received none sig, stop stich result")
            break
        moves, qstrings, seqs = results['moves'], results['qstring'], results['sequence']
        i, j = 0, 0
        while j < len(chunks):
            while  j < len(chunks) and chunks[j]['read_id'] == chunks[i]['read_id'] : j += 1
            try:
                result_write, result_mod = stich_result(chunks[i:j],
                                     moves[i:j],
                                     qstrings[i:j],
                                     seqs[i:j],
                                     chunk_size,
                                     overlap,
                                     stride
                                     )
                res_chunks.put(result_write)
                if mod_chunks is not None:
                    mod_chunks.put(result_mod)
            except Exception as e:
                LOGGER.error("Error occurred while stitching result, for read: {}, except info: {}".format(chunks[i]['read_id'], e))
                error_handler(e)
                break
            i = j

def mod_feature_extractor(
    mod_chunks: multiprocessing.Queue,
    dataQueue: multiprocessing.Queue,
    ref_path : str,
    batch_size: int ,
    kmer_size : int ,
    sig_len : int,
):
    ref_genome = mappy.Aligner(ref_path, preset='map-ont')
    num_bases = kmer_size // 2
    fe_cnt = 0
    site_info_batch = ["", ] * batch_size
    kmer_batch = np.zeros((batch_size, kmer_size), dtype=np.int64)
    signal_batch = np.zeros((batch_size, kmer_size, sig_len + 4), dtype=np.float32)
    while True:
        result = mod_chunks.get()
        if result is None: break
        if result['filter_Q'] < 10: continue
        first_hit = next(ref_genome.map(result['sequence']), None)
        if first_hit is None: continue
        ref_seq = ref_genome.seq(first_hit.ctg)[first_hit.r_st: first_hit.r_en]
        if first_hit.strand == -1:
            ref_seq = complement_seq(ref_seq, seq_type="DNA")
        strand_code = 1 if first_hit.strand == 1 else 0

        r_to_q_poss = parse_cigar(first_hit.cigar, strand_code, len(ref_seq))
        # if result['read_id'] == "035E29E6-0FC1-4541-BD63-403537636ACF":
        #     print("1111")

        # basecode =  np.array([base2code_dna[x] for x in result['sequence']], dtype=np.uint8)
        motifs_loc = get_refloc_of_methysite_in_motif(ref_seq, ["CG"], 0)
        signal_group, sig_len_l = group_signal_by_base2signal(
            result['signal'],
            result['moves'],
            0,
            sig_len=sig_len,
            stride=result['sd'],
        )

        ref_readlocs = np.zeros(len(ref_seq), dtype=np.float32)
        ref_signal_grp = np.zeros([len(ref_seq), sig_len], dtype=np.float32)
        ref_baseprobs = np.zeros(len(ref_seq), dtype=np.float32)
        ref_sig_len = np.zeros(len(ref_seq), dtype=np.float32)

        result['q_arr'] = result['q_arr'] / 12
        for ref_pos, q_pos in enumerate(r_to_q_poss[:-1]):
            ref_readlocs[ref_pos] = q_pos + first_hit.q_st
            ref_signal_grp[ref_pos] = signal_group[q_pos + first_hit.q_st]
            ref_baseprobs[ref_pos] = result['q_arr'][q_pos + first_hit.q_st]
            ref_sig_len[ref_pos] = sig_len_l[q_pos + first_hit.q_st]

        for off_loc in motifs_loc:
            if off_loc < num_bases or off_loc >= len(ref_seq) - num_bases: continue
            abs_loc = (first_hit.r_st + off_loc) if strand_code == 1 else (first_hit.r_en - 1 - off_loc)
            try:
                kmer_base = np.array(
                    [base2code_dna[x] for x in ref_seq[(off_loc - num_bases): (off_loc + num_bases + 1)]],
                    dtype=np.int32)
                k_seq_qual = ref_baseprobs[(off_loc - num_bases): (off_loc + num_bases + 1)]
                k_signals = ref_signal_grp[(off_loc - num_bases): (off_loc + num_bases + 1)]

                k_signal_lens = ref_sig_len[(off_loc - num_bases): (off_loc + num_bases + 1)]

                signal_means = np.array([np.mean(x) for x in k_signals], dtype=np.float32)
                signal_stds = np.array([np.std(x) for x in k_signals], dtype=np.float32)
                # k_signals_rect = np.asarray(get_signals_rect(k_signals, signals_len=signal_len), dtype=np.float32)

                signal_means = signal_means.reshape(kmer_size, -1)
                signal_stds = signal_stds.reshape(kmer_size, -1)
                k_signal_lens = k_signal_lens.reshape(kmer_size, -1)
                k_seq_qual = k_seq_qual.reshape(kmer_size, -1)
                k_signals = k_signals.reshape(kmer_size, -1)

                signal_cat = np.concatenate((signal_means, signal_stds, k_signal_lens, k_seq_qual,
                                             k_signals), axis=1).reshape(kmer_size, 3 * result['sd'] + 4)

                if np.sum(np.isnan(signal_cat)) != 0 or np.sum(np.isnan(kmer_base)):
                    LOGGER.warning(
                        "Found NAN at h5 file {}, read {}, offloc {}".format(result['fn'], result["read_id"], off_loc))
                    continue

                site_info = result["read_id"] + "\t" + \
                            str(first_hit.r_st) + "\t" + \
                            str(first_hit.r_en) + "\t" + \
                            str(first_hit.ctg) + "\t" + \
                            str(abs_loc) + "\t" + str(strand_code)

                # if result['read_id'] == "035E29E6-0FC1-4541-BD63-403537636ACF" and abs_loc == 1255431:
                #     time.sleep(1)

                site_info_batch[fe_cnt % batch_size] = site_info
                kmer_batch[fe_cnt % batch_size] = kmer_base
                signal_batch[fe_cnt % batch_size] = signal_cat

                fe_cnt += 1
                if fe_cnt == batch_size:
                    fe_cnt = 0
                    dataQueue.put((copy.deepcopy(site_info_batch),
                                   copy.deepcopy(kmer_batch),
                                   copy.deepcopy(signal_batch)), block=True)
                    # time.sleep(5)
            except Exception as e:
                error_handler(e)
    if fe_cnt > 0:
        dataQueue.put((site_info_batch[:fe_cnt], kmer_batch[:fe_cnt], signal_batch[:fe_cnt]), block=True)
    # LOGGER.info("Finished writing to queue")

def integrated_call_mods(dataQueue : multiprocessing.Queue,
                         resQueue: multiprocessing.Queue,
                         module_path : str,
                         kmer_size : int = 21,
                         device_id : int = 0,
                         is_half : bool = True,
                             ):
    device = torch.device("cuda:{}".format(device_id))
    LOGGER.info("modification inference process started, running device-{}".format(device))
    try:
        model = BiLSTM_attn(kmer=kmer_size,
                        hidden_size=256,
                        embed_size=[16, 4],
                        dropout_rate=0.3,
                        num_layer1=2,
                        num_layer2=3,
                        num_classes=2)
        # model = torch.jit.load(module_path)
        state_dict = torch.load(module_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        if is_half:
            model = model.half()
        model.eval()
    except Exception as e:
        LOGGER.error("Error occurred while loading modification model")
        error_handler(e)

    while True:
        try:
            site_info_batch, kmer_batch, signal_batch = dataQueue.get( block=True)
            if site_info_batch is None: break
            kmer_batch = torch.tensor(kmer_batch).to(torch.int32).to(device)
            signal_batch = torch.tensor(signal_batch).to(torch.float16).to(device)

            _, logits = model(kmer_batch, signal_batch)

            p_rate = logits[:, 1].cpu().detach().numpy()

            resQueue.put((site_info_batch, p_rate), block=True)

        except Exception as e:
            LOGGER.error("Error occured while running model")
            error_handler(e)
            time.sleep(100)
            # exit()

    resQueue.put((None, None), block=True)
    LOGGER.info("Model inference process finished")
    return

def write_mod_result(resQueue : multiprocessing.Queue,
                      write_path : str,
                      ):
    LOGGER.info("Write result process started")
    w_file = open(write_path, "w")
    while True:
        try:
            site_info_batch, p_rate = resQueue.get(block=True)
            if site_info_batch is None: break
            for i in range(len(site_info_batch)):
                w_file.write(site_info_batch[i] + "\t" + str(p_rate[i]) + "\n")
        except Exception as e:
            LOGGER.error("Error occured while writing modification result")
            # time.sleep(1000)
            error_handler(e)
    LOGGER.info("write modification result process finished")

