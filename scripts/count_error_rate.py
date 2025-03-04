import os.path

import mappy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import pysam
# from utils.utils_func import *

# f = open("/data1/YHC/QiTan_data/Fruitfly/YF6418_h5/Sturgeon_2/high_err_loc.txt", 'w')
def counter(path, filter_err : list = [0, 1]):
    f = open(os.path.join(os.path.dirname(path), "high_err_loc_{}_{}.txt".format(filter_err[0], filter_err[1])), 'w')
    bam_file = pysam.AlignmentFile(path, 'r', threads=10)
    mapped_reads, total_reads = 0, 0
    total_error_rate = []
    total_seq_len = []
    total_error_list = []
    total_qual_list = []
    # cnt = 0
    for read in tqdm(bam_file):
        total_reads += 1
        if total_reads >= 1000000: break
        if read.is_unmapped: continue

        if read.query_sequence is None or len(read.query_alignment_sequence) <= 2000:
            continue
        query_seq = read.query_sequence
        ref_seq = read.get_reference_sequence()
        mapped_reads += 1
        M = 0
        insert, delete, soft_clip, hard_clip = 0, 0, 0, 0
        seq_len, ref_len = 0, 0
        num_ins, num_del = 0, 0
        mis_match_list = []
        insert_list = []
        delete_list = []
        qual_list = []
        cnt_curr = 500
        rate_mis_match_bp = 0
        rate_insert_bp = 0
        rate_delete_bp = 0
        base_ref_cnt = 0
        base_query_cnt = 0

        # if read.query_name != "0CE9E98D-4F9C-4AA6-B09A-48737101CFF9": continue
        for (op, op_len) in read.cigar:
            ref_len_pre = ref_len
            if op == 0:
                M += op_len
                for i in range(op_len):
                    if query_seq[seq_len + i] != ref_seq[ref_len + i]:
                        rate_mis_match_bp += 1
                seq_len += op_len
                ref_len += op_len
            elif op == 1:
                insert += op_len
                num_ins += 1
                rate_insert_bp += 1
                seq_len += op_len
            elif op == 2:
                delete += op_len
                num_del += 1
                rate_delete_bp += 1
                ref_len += op_len
            elif op == 4:
                soft_clip += op_len
                seq_len += op_len
            elif op == 5:
                hard_clip += op_len
            else:
                print("find other tag-{}".format(op))
            if ref_len > ref_len_pre and ref_len >= cnt_curr:  # 达到500bp的长度
                cnt_curr = min(cnt_curr + 500, len(ref_seq))
                if (ref_len - base_ref_cnt) <= 300: continue
                temp_mis_match = rate_mis_match_bp / (ref_len - base_ref_cnt)
                temp_insert = rate_insert_bp / (ref_len - base_ref_cnt)
                temp_delete = rate_delete_bp / (ref_len - base_ref_cnt)

                mis_match_list.append(temp_mis_match)
                insert_list.append(temp_insert)
                delete_list.append(temp_delete)
                qual_list.append(np.mean(read.query_qualities[base_query_cnt : base_query_cnt + seq_len]))
                if (temp_mis_match + temp_delete + temp_insert) >= filter_err[0] and (temp_mis_match + temp_delete + temp_insert) < filter_err[1]:
                    # print("Find high error rate subseq, read: {}, ref name: {}, relative loc: [{}, {}] ".format(
                    #     read.query_name, read.reference_name, base_cnt, ref_len
                    # ))
                    f.write("{},{},{},{},{},{}\n".format(read.query_name,
                                                      read.reference_name,
                                                      read.reference_start,
                                                      read.reference_end,
                                                      read.reference_start + base_ref_cnt,
                                                      read.reference_start + ref_len))
                base_ref_cnt = ref_len
                base_query_cnt = seq_len
                rate_mis_match_bp = 0
                rate_insert_bp = 0
                rate_delete_bp = 0
        try:
            NM = read.get_tag("NM")  # mismatches, inserted, deleted
        except:
            continue  # skip reads that have no tags
        mis_match = (NM - insert - delete)
        assert len(mis_match_list) == len(insert_list) == len(delete_list)
        error_list = (np.array(mis_match_list) + np.array(insert_list) + np.array(delete_list))
        total_error_list.append(error_list)
        total_qual_list.append(np.array(qual_list))
        error_rate = (mis_match + num_ins + num_del) / ref_len
        total_error_rate.append(error_rate)
        total_seq_len.append(seq_len)
    f.close()
    return (total_error_rate, total_error_list
            # , total_qual_list
            # , high_err_loc_fraction, total_seq_len
            )

total_error_rate1, total_error_list1 = counter("/data1/YHC/QiTan_data/ara/Sturgeon_2/alignment.bam")
#
# print(np.mean([np.corrcoef(-x, y)[0][1] for x, y in zip(total_error_list1, total_qual_list1)]))

# print(cosine_similarity = np.dot(total_error_list1, total_qual_list1) / (np.linalg.norm(total_error_list1) * np.linalg.norm(total_qual_list1)))

total_error_rate2, total_error_list2 = counter("/data1/YHC/QiTan_data/oryza/Sturgeon_2/alignment.bam",
                                               # [0, 0.05]
                                               )
# total_error_rate2, total_error_list2 = counter("/data1/YHC/QiTan_data/oryza/Sturgeon_2/alignment.bam",
#                                                [0.05, 0.1])
# total_error_rate2, total_error_list2 = counter("/data1/YHC/QiTan_data/oryza/Sturgeon_2/alignment.bam",
#                                                [0.1, 0.15])
# total_error_rate2, total_error_list2 = counter("/data1/YHC/QiTan_data/oryza/Sturgeon_2/alignment.bam",
#                                                [0.15, 0.2])
# total_error_rate2, total_error_list2 = counter("/data1/YHC/QiTan_data/oryza/Sturgeon_2/alignment.bam",
#                                                [0.2, 0.25])
# total_error_rate2, total_error_list2 = counter("/data1/YHC/QiTan_data/oryza/Sturgeon_2/alignment.bam",
#                                                [0.25, 0.3])
# total_error_rate2, total_error_list2 = counter("/data1/YHC/QiTan_data/oryza/Sturgeon_2/alignment.bam",
#                                                [0.3, 1])
# total_error_rate3, total_error_list3 = counter("/data1/YHC/QiTan_data/Fruitfly/YF6418_h5/Sturgeon_2/alignment.bam",
#                                                # [0, 0.05]
#                                                )
# total_error_rate4, total_error_list4 = counter("/data1/YHC/QiTan_data/Fruitfly/YF6419_h5/Sturgeon_2/alignment.bam",
#                                                # [0, 0.05]
#                                                )

# total_error_rate3, total_error_list3 = counter("/data1/YHC/QiTan_data/Fruitfly/YF6418_h5/Sturgeon_2/alignment.bam", [0.05, 0.1])
# total_error_rate4, total_error_list4 = counter("/data1/YHC/QiTan_data/Fruitfly/YF6419_h5/Sturgeon_2/alignment.bam", [0.05, 0.1])

# total_error_rate3, total_error_list3 = counter("/data1/YHC/QiTan_data/Fruitfly/YF6418_h5/Sturgeon_2/alignment.bam", [0.1, 0.15])
# total_error_rate4, total_error_list4 = counter("/data1/YHC/QiTan_data/Fruitfly/YF6419_h5/Sturgeon_2/alignment.bam", [0.1, 0.15])

# total_error_rate3, total_error_list3 = counter("/data1/YHC/QiTan_data/Fruitfly/YF6418_h5/Sturgeon_2/alignment.bam", [0.15, 0.2])
# total_error_rate4, total_error_list4 = counter("/data1/YHC/QiTan_data/Fruitfly/YF6419_h5/Sturgeon_2/alignment.bam", [0.15, 0.2])
#
# total_error_rate3, total_error_list3 = counter("/data1/YHC/QiTan_data/Fruitfly/YF6418_h5/Sturgeon_2/alignment.bam", [0.2, 0.25])
# total_error_rate4, total_error_list4 = counter("/data1/YHC/QiTan_data/Fruitfly/YF6419_h5/Sturgeon_2/alignment.bam", [0.2, 0.25])
#
# total_error_rate3, total_error_list3 = counter("/data1/YHC/QiTan_data/Fruitfly/YF6418_h5/Sturgeon_2/alignment.bam", [0.25, 0.3])
# total_error_rate4, total_error_list4 = counter("/data1/YHC/QiTan_data/Fruitfly/YF6419_h5/Sturgeon_2/alignment.bam", [0.25, 0.3])
#
total_error_rate3, total_error_list3 = counter("/data1/YHC/QiTan_data/Fruitfly/YF6418_h5/Sturgeon_2/alignment.bam",
                                               # [0.3, 1]
                                               )
total_error_rate4, total_error_list4 = counter("/data1/YHC/QiTan_data/Fruitfly/YF6419_h5/Sturgeon_2/alignment.bam",
                                               # [0.3, 1]
                                               )

# plt.figure()
# plt.scatter(list(high_err_loc_fraction.keys()), [np.mean(high_err_loc_fraction[x]) for x in list(high_err_loc_fraction.keys())])
# plt.plot()


plt.figure(figsize=(6, 4))
sns.kdeplot({
            "A. thaliana" :total_error_rate1,
            "O. sativa" : total_error_rate2,
            "YF6418": total_error_rate3,
            "YF6419": total_error_rate4
             })
plt.xlabel("error rate(%)")
plt.ylabel("fraction of reads(%)")
plt.title("error rate per read distribution")
plt.savefig("fig5_1.svg", format='svg', bbox_inches='tight')
plt.show()

# pre_1000base_error_rate = float(np.mean([x[:2] for x in total_error_list1 if len(x) >= 2]))
# post_1000base_error_rate = float(np.mean([x[-2:] for x in total_error_list1 if len(x) >= 2]))
#
# print("A. theliana, pre 1k base: {:.4f}%, post 1k base: {:.4f}%".format(pre_1000base_error_rate * 100,
#                                                           post_1000base_error_rate * 100))

# pre_1000base_error_rate = float(np.mean([x[:2] for x in total_error_list2 if len(x) >= 2]))
# post_1000base_error_rate = float(np.mean([x[-2:] for x in total_error_list2 if len(x) >= 2]))
#
# print("O. sativa, pre 1k base: {:.4f}%, post 1k base: {:.4f}%".format(pre_1000base_error_rate * 100,
#                                                           post_1000base_error_rate * 100))
#
# pre_1000base_error_rate = float(np.mean([x[:2] for x in total_error_list3 if len(x) >= 2]))
# post_1000base_error_rate = float(np.mean([x[-2:] for x in total_error_list3 if len(x) >= 2]))
#
# print("YF6418, pre 1k base: {:.4f}%, post 1k base: {:.4f}%".format(pre_1000base_error_rate * 100,
#                                                           post_1000base_error_rate * 100))
#
# pre_1000base_error_rate = float(np.mean([x[:2] for x in total_error_list4 if len(x) >= 2]))
# post_1000base_error_rate = float(np.mean([x[-2:] for x in total_error_list4 if len(x) >= 2]))
#
# print("YF6419, pre 1k base: {:.4f}%, post 1k base: {:.4f}%".format(pre_1000base_error_rate * 100,
#                                                           post_1000base_error_rate * 100))


# f.close()
