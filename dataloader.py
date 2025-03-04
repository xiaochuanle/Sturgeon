import torch
from torch.utils.data import Dataset
import os
import numpy as np


def load_numpy_data(directory : str, split = 0.95):

    chunks = np.load(os.path.join(directory, "chunks.npy"))
    targets = np.load(os.path.join(directory, "references.npy"))
    lengths = np.load(os.path.join(directory, "reference_lengths.npy"))

    np.random.seed(1)
    indices = np.random.permutation(len(lengths))
    chunks = chunks[indices]
    targets = targets[indices]
    lengths = lengths[indices]

    split_idx = int(split * len(lengths))

    train_data = [chunks[:split_idx], targets[:split_idx], lengths[:split_idx]]
    valid_data = [chunks[split_idx:], targets[split_idx:], lengths[split_idx:]]

    return train_data, valid_data

class ctc_dataset(Dataset):
    def __init__(self, data, transform=None):
        self.chunks = data[0]
        self.targets = data[1]
        self.lengths = data[2]
        self.data_size = len(self.lengths)

    def __getitem__(self, index):
        return (
            self.chunks[index].astype(np.float32).reshape(1,-1),
            self.targets[index].astype(np.int64),
            self.lengths[index].astype(np.int64),
        )

    def __len__(self):
        return self.data_size

class Dataset_npy(Dataset):
    """
    Dataset loader for binary numpy files.
    """

    def __init__(self, filepath: str, kmer: int = 21, transform: object = None) -> Dataset:
        self._data = np.load(filepath)
        np.random.shuffle(self._data)
        self._kmer = kmer
        self._transform = transform

    def __getitem__(self, idx: int) -> tuple[np.array]:
        kmer = self._data[idx][:self._kmer]
        signal = np.reshape(self._data[idx][self._kmer: -1], ( self._kmer, -1))
        label = self._data[idx][-1]
        return kmer, signal, label

    def __len__(self, ) -> int:
        return self._data.shape[0]

class Dataset_npy_2(Dataset):
    """
    Dataset loader for binary numpy files.
    """

    def __init__(self, filepath_1: str, filepath_2: str, kmer: int = 21, transform: object = None) -> Dataset:
        matrix1 = np.load(filepath_1)
        matrix2 = np.load(filepath_2)
        np.random.shuffle(matrix2)
        np.random.shuffle(matrix1)
        trim = min(matrix1.shape[0], matrix2.shape[0])
        matrix1 = matrix1[:trim]
        matrix2 = matrix2[:trim]
        matrix1[:, -1] = 1
        matrix2[:, -1] = 0
        self._data = np.concatenate((matrix1, matrix2), axis=0)
        # del matrix1, matrix2
        np.random.shuffle(self._data)
        self._kmer = kmer
        self._transform = transform

    def __getitem__(self, idx: int) -> tuple[np.array]:
        kmer = self._data[idx][:self._kmer]
        signal = np.reshape(self._data[idx][self._kmer: -1], (self._kmer, -1))
        label = self._data[idx][-1]
        return kmer, signal, label

    def __len__(self, ) -> int:
        return self._data.shape[0]