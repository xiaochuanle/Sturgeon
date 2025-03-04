import numpy as np
import ctypes
import time
import os
# os.chdir("../../")

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