import numpy as np
import ctypes
import time
from tqdm import tqdm
from ctc_decoder import vertibi_decode_fast

lib = ctypes.CDLL("./build/libctc_decode.so")

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

    st = time.time()
    seqs_2, moves_2, quals_2 = viterbi_decode(inputs)
    ed = time.time()
    print(ed - st)

    print("start to test result")
    for i in range(N):
        for j in range(T):
            if seqs[i, j] != seqs_2[i][j]:
                print(i, j, seqs[i, j], seqs_2[i][j])
            if moves[i, j] != moves_2[i][j]:
                print(i, j, moves[i, j], moves_2[i][j])
            if quals[i, j] != quals_2[i][j]:
                print(i, j, quals[i][j], quals_2[i][j])
    print("end")


def phred(x):
    """
    Convert probability to phred quality score.
    """
    x = np.array(x, dtype=np.float32)

    x = np.clip(x, 1e-7, 1.0 - 1e-7)

    return -10 * np.log10(1 - x)
def viterbi_decode(inputs):
    T, N, C = inputs.shape
    soft_inputs = np.exp(inputs.astype(np.float32))
    logits = soft_inputs.argmax(2)
    seqs, moves, quals = [], [], []
    for i in range(N):
        seq = np.zeros(T, dtype=np.uint8)
        ctc_pred = logits[:,i]
        move = np.zeros(T, dtype=np.uint8)
        qual = np.zeros(T, dtype=np.float32)
        if ctc_pred[0] != 0:
            seq[0] = ctc_pred[0]
            qual[0] = soft_inputs[0, i, ctc_pred[0]][()]
            move[0] = 1
        for j in range(1, T):
            if ctc_pred[j] != ctc_pred[j-1] and ctc_pred[j] != 0:
                seq[j] = ctc_pred[j]
                qual[j] = soft_inputs[j, i, ctc_pred[j]][()]
                move[j] = 1

        seqs.append(seq)
        moves.append(np.array(move, dtype=np.uint8))
        quals.append(np.array([x + 33 if x > 1e-6 else x for x in phred(qual)], dtype=np.uint8))
    return seqs, moves, quals


ctc_veterbi_decode_test()
