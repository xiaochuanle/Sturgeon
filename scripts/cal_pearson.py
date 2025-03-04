import os
import numpy as np
import time
from tqdm import tqdm
import torch
from torchmetrics import regression


def load_detail_dict(path, target_depth : float = 1., total_depth : float = 1.):
    filter_ratio = target_depth / total_depth
    print("reading file {}".format(path))
    detail_dict = {}
    detail_set = set()
    for line in tqdm(open(path)):
        line_s = line.strip("\n").split("\t")
        if line_s[0] == "read_id" or np.random.rand() > filter_ratio: continue
        key = line_s[3] + "\t" + line_s[4]
        if key not in detail_set:
            detail_set.add(key)
            detail_dict[key] = np.array([0, 0], dtype=float)
        detail_dict[key][0] += float(float(line_s[-1]) > 0.5)
        detail_dict[key][1] += 1
    for key in detail_dict.keys():
        detail_dict[key][0] = detail_dict[key][0] / detail_dict[key][1]
    return detail_dict, detail_set


def load_bisulfite_dict(path):
    print("reading file {}".format(path))
    bisulfite_dict = {}
    for line in tqdm(open(path)):
        line_s = line.strip().split("\t")
        # try:
        key = line_s[0] + "\t" + str(int(line_s[1]))
        bisulfite_dict[key] = np.array([float(line_s[3]),
                                            float(line_s[4]) + float(line_s[5])], dtype=float)
        # except:
        #     print("error line: {}".format(line))
        #     pass
    return bisulfite_dict


def cal_pearson(bisulfite_dict, detail_dict, filter_size_thresh=-1):
    x = []
    y = []
    keys = set(list(detail_dict.keys()))
    for key in list(bisulfite_dict.keys()):
        if key in keys:
            if bisulfite_dict[key][1] <= 5 and detail_dict[key][1] <= 5: continue
            if filter_size_thresh > 0 and bisulfite_dict[key][0] <= filter_size_thresh: continue
            x.append(bisulfite_dict[key][0])
            y.append(detail_dict[key][0])

    X = torch.tensor(x)
    Y = torch.tensor(y)

    pearson = regression.PearsonCorrCoef()

    coef = pearson(X, Y)
    if filter_size_thresh > 0:
        print("filter threshold: {}, counted sites: {}, calculated Pearson coef: {}".format( filter_size_thresh, len(X), coef))
    else:
        print("counted sites: {}, calculated Pearson coef: {}".format(len(X),coef))

    # print("calculated Pearson coef: {}".format(coef))



if __name__ == "__main__":

    st = time.time()
    detail_path = "/public3/YHC/mod_result/oryza_mod_result.txt"
    bisulfite_path = "/mnt/sdg2/oryza/Ory.CpG.gz.bismark.zero.cov"

    detail_dict, detail_set = load_detail_dict(detail_path, 100,60.3)
    bisulfite_dict = load_bisulfite_dict(bisulfite_path)

    cal_pearson(bisulfite_dict, detail_dict,-1)
    # cal_pearson(bisulfite_dict, detail_dict, 0.02)
    # cal_pearson(bisulfite_dict, detail_dict, 0.05)
    # cal_pearson(bisulfite_dict, detail_dict, 0.1)
    # cal_pearson(bisulfite_dict, detail_dict, 0.15)
    # cal_pearson(bisulfite_dict, detail_dict, 0.2)

    print("calculated Pearson finished, cost time: {} seconds".format( time.time() - st))