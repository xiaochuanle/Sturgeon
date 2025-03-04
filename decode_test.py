import torch

import numpy as np
import ctypes
import time
import os
# os.chdir("../../")
# print(os.c)

lib = ctypes.CDLL("./cpp_components/build/libctc_decode.so")

lib.ctc_veterbi_decode.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=3, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=2, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
lib.ctc_veterbi_decode.restype = ctypes.c_bool

def vertibi_decode_fast(inputs):
    T, N, C = inputs.shape
    inputs = inputs.astype(np.float32)
    seqs = np.zeros([N, T], dtype=np.uint8)
    moves = np.zeros([N, T], dtype=np.uint8)
    quals = np.zeros([N, T], dtype=np.uint8)
    lib.ctc_veterbi_decode(inputs, seqs, moves, quals, T, N, C)
    return seqs, moves, quals
def log_softmax(x, axis=None):
    """
    Compute the log softmax of each element along a specified axis of x.

    Parameters:
    x : ndarray
        Input array.
    axis : int or None
        Axis along which to compute the log softmax. If None, compute over the entire array.

    Returns:
    ndarray
        An array the same shape as x with the log softmax applied along the specified axis.
    """
    # Subtract the max for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    shifted_x = x - x_max

    # Compute log softmax
    log_sum_exp = np.log(np.sum(np.exp(shifted_x), axis=axis, keepdims=True))
    return shifted_x - log_sum_exp

def ctc_veterbi_decode_test():
    np.random.seed(1)
    T,N,C = 1200, 128, 5

    inputs = log_softmax(np.random.rand(T, N, C).astype(np.float32), axis=2)
    # seqs = np.zeros([N, T], dtype=np.uint8)
    # moves = np.zeros([N, T], dtype=np.uint8)
    # quals = np.zeros([N, T], dtype=np.uint8)
    st = time.time()
    # lib.ctc_veterbi_decode(inputs, seqs, moves, quals, T, N, C)
    seqs, moves, quals = vertibi_decode_fast(inputs)
    ed = time.time()

    print(ed - st)

ctc_veterbi_decode_test()