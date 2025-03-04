import h5py
import numpy as np
import time
import traceback
import os
from multiprocessing import Pool, Manager
from utils.utils_func import *
from utils.readers import *
import mappy
LOGGER = get_logger(__name__)
def extract_features(fastq_path : str,
                     h5_path : str,
                     ref_path : str,
                     write_path : str,
                     label : int = -1,
                     k_size : int = 21,
                     sig_len : int = 15,
                     ):

    st = time.time()

    LOGGER.info("Extract feature for {} and {} start".format(fastq_path.split("/")[-1], h5_path.split("/")[-1]))
    h5_reads = h5py.File(h5_path, 'r')

    fastq_l = []
    for line in open(fastq_path):
        fastq_l.append(line)

    ref_genome = mappy.Aligner(ref_path)

    num_bases = (k_size - 1) // 2
    cnt_total, cnt_selected = 0, 0
    feature_lists = []
    for i in range(int(len(fastq_l) / 4)):
        info = fastq_l[4 * i]
        seq = fastq_l[4 * i + 1]
        c = fastq_l[4 * i + 2]
        qual = fastq_l[4 * i + 3]
        f = Fastq_Read(info, seq, qual)

        signal = h5_reads["Raw_data"][f.read_id][()]
        first_hit = next(ref_genome.map(f.seq), None)
        if first_hit is None: continue


        cnt_total += 1
        if len(signal) <= 6400: continue
        if f.q_score <= 11.0: continue
        if first_hit.mapq < 15: continue

        ref_seq = ref_genome.seq(first_hit.ctg)[first_hit.r_st : first_hit.r_en]
        if first_hit.strand == -1:
            ref_seq = complement_seq(ref_seq, seq_type="DNA")
        strand_code = 1 if first_hit.strand == 1 else 0
        r_to_q_poss = parse_cigar(first_hit.cigar, strand_code, len(ref_seq))

        base2signal = f.get_base2signal()
        motifs_loc = f.get_refloc_of_methysite_in_motif(ref_seq,["CG"], 0)
        if (len(motifs_loc) == 0): continue
        cnt_selected += 1

        base_qual = f.get_seq_quality()
        basecode = f.get_base_code()
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
            if f.h5_file_name == "reads_0023_F9C5190C-FF31-4BC3-9CE5-95F7640CF4B3.h5" and f.read_id == "1B8D3D14-777F-4FBB-A18E-F591EB3CC040" and off_loc == 13650:
                LOGGER.info("debug specific read loc")
            try:
                kmer_base = np.array([ base2code_dna[x]  for x in ref_seq[(off_loc - num_bases) : (off_loc + num_bases + 1)]],dtype=np.int32)
                k_seq_qual = ref_baseprobs[(off_loc - num_bases) : (off_loc + num_bases + 1)]
                k_signals = ref_signal_grp[(off_loc - num_bases) : (off_loc + num_bases + 1)]

                # signal_lens = np.array([len(x) for x in k_signals], dtype=np.int32)
                k_signal_lens = ref_sig_len[(off_loc - num_bases) : (off_loc + num_bases + 1)]


                signal_means = np.array([np.mean(x) for x in k_signals], dtype=np.float32)
                signal_stds = np.array([np.std(x) for x in k_signals], dtype=np.float32)
                # k_signals_rect = np.asarray(get_signals_rect(k_signals, signals_len=signal_len), dtype=np.float32)

                signal_means = signal_means.reshape(k_size, -1)
                signal_stds = signal_stds.reshape(k_size, -1)
                k_signal_lens = k_signal_lens.reshape(k_size, -1)
                k_seq_qual = k_seq_qual.reshape(k_size, -1)
                k_signals = k_signals.reshape(k_size, -1)

                feature = np.concatenate((signal_means, signal_stds, k_signal_lens, k_seq_qual,
                                          k_signals), axis=1, dtype=np.float32).reshape(-1)
                feature = np.append(kmer_base, feature)
                if np.sum(np.isnan(feature)) != 0:
                    LOGGER.warning("Found NAN at h5 file {}, read {}".format(f.h5_file_name, f.read_id))
                    continue
                feature_lists.append(np.append(feature, label).astype(np.float32))
                # cnt_selected += 1
            except Exception as e:
                LOGGER.error("Error occurred at h5 file: {}, read: {}, off_loc: {}, error info: {}".format(f.h5_file_name, f.read_id, off_loc, e))
                error_type = type(e).__name__
                tb = traceback.format_exc()
                print(f"Error type: {error_type}")
                print("Traceback details:")
                print(tb)
                time.sleep(1000)
    np.save(write_path, np.array(feature_lists, dtype=np.float32))

    ed = time.time()

    LOGGER.info("Extract features for file {} end, total read {},"
                " selected read {}, time const {} seconds".format(f.h5_file_name,
                                                                 cnt_total, cnt_selected,  ed - st))




if __name__ == "__main__":
    st = time.time()
    h5_dir = "/mnt/sdg2/QiTan_fruitfly/YF6419_h5"
    fastq_dir = "/mnt/sdg2/QiTan_fruitfly/YF6419_fastq/splited_pass"
    ref_path = "/mnt/sdg2/QiTan_fruitfly/FRUITFLY.hifiasm.diploid.fasta"
    # write_dir = "/public1/YHC/QiTanTechData/YF6419_fe/"
    write_dir = "."
    motifs = ["CG"]
    f_list = os.listdir(h5_dir)
    h5_files = []
    for f in f_list:
        if os.path.splitext(f)[1] == ".h5":
            h5_files.append(f)

    np.random.seed(0)
    np.random.shuffle(h5_files)

    LOGGER.info("Found {} h5 files".format(len(h5_files)))
    # pool = Pool(48)
    # for h5_f in h5_files:
    #     if len(h5_f.split(".")) == 2:
    #         time.sleep(5) # avoid too many io
    #         h5_file = os.path.join(h5_dir, h5_f)
    #         fastq_file = os.path.join(fastq_dir, h5_f.split(".")[0] + ".fastq")
    #         write_file = os.path.join(write_dir, h5_f.split(".")[0])
    #         pool.apply_async(extract_features, args=(
    #             fastq_file,
    #             h5_file,
    #             ref_path,
    #             write_file,
    #             0
    #         ))
    # pool.close()
    # pool.join()

    for h5_f in h5_files:
        if len(h5_f.split(".")) == 2:
            h5_file = os.path.join(h5_dir, h5_f)
            if h5_f != "reads_0032_07FF203C-8175-4872-A4BC-8BDEA7B84CE2.h5": continue
            fastq_file = os.path.join(fastq_dir , h5_f.split(".")[0] + ".fastq")
            write_file = os.path.join(write_dir , h5_f.split(".")[0])
            extract_features(fastq_file,
                             h5_file,
                             ref_path,
                             write_file,
                             0)

    LOGGER.info("Extract features end, "
                "total cost {} seconds".format(time.time() - st))