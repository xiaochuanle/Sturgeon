import numpy as np
import os

def load_numpy_data(directory : str):

    chunks = np.load(os.path.join(directory, "chunks.npy"))
    references = np.load(os.path.join(directory, "references.npy"))
    lengths = np.load(os.path.join(directory, "reference_lengths.npy"))

    return chunks, references, lengths

def merge_ctc_data(directories : [str], target_dir : str):
    chunk_list = []
    reference_list = []
    length_list = []
    for directory in directories:
        chunks, references, lengths = load_numpy_data(directory)
        chunk_list.append(chunks)
        reference_list.append(references)
        length_list.append(lengths)

    total_chunks = np.concatenate(chunk_list, axis=0, dtype=np.float32)

    max_ref_len = max([ref.shape[1] for ref in reference_list])
    total_references = np.zeros((total_chunks.shape[0], max_ref_len), dtype=np.uint8)

    st_idx = 0
    for  ref in reference_list:
        total_references[st_idx : st_idx + ref.shape[0], :ref.shape[1]] = ref
        st_idx += ref.shape[0]

    total_lengths = np.concatenate(length_list, axis=0, dtype=np.uint16)

    assert (total_chunks.shape[0] == total_lengths.shape[0] and total_chunks.shape[0] == total_references.shape[0])

    indices = np.random.permutation(len(total_lengths))
    total_chunks = total_chunks[indices]
    total_references = total_references[indices]
    total_lengths = total_lengths[indices]

    np.save(os.path.join(target_dir, "chunks.npy"), total_chunks)
    np.save(os.path.join(target_dir, "references.npy"), total_references)
    np.save(os.path.join(target_dir, "reference_lengths.npy"), total_lengths)

    print("Total merge data chunk size:{}".format(len(total_lengths)))

if __name__ == "__main__":
    directories = [
        "/data1/YHC/QiTanTrain/ARA",
        "/data1/YHC/QiTanTrain/ORYZA",
        "/data1/YHC/QiTanTrain/FRUITFLY/YF6418",
        "/data1/YHC/QiTanTrain/FRUITFLY/YF6419",
    ]
    target_dir = "/data1/YHC/QiTanTrain/QiTan_basecall_train_all_3"

    merge_ctc_data(directories, target_dir)