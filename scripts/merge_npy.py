import numpy as np
import os
import glob
from tqdm import tqdm
import shutil
import traceback
file_path = "/public1/YHC/QiTanTechData/YF6419_fe/"
file_list = os.listdir(file_path)
file_list = [x for x in file_list if x.endswith(".npy")]
file_list = [os.path.join(file_path , x) for x in file_list]
# file_list = glob.glob("/data1/YHC/Train/neg/**/*.npz", recursive=True)
# file_list = [ x for x in file_list if x.endswith("npz")]
np.random.shuffle(file_list)
print("find {} npy files".format(len(file_list)))

a = 0
kmer = 21
file_size = 10
chunk_size = int((len(file_list) + file_size - 1) // file_size)

merge_dir = os.path.join(file_path, "merge")
if os.path.exists(merge_dir):
    shutil.rmtree(merge_dir)
os.mkdir(merge_dir)

print("merge directory: {}".format(merge_dir))

print("Merge every {} npy files into a chunk".format(chunk_size))

while a < len(file_list) - 1:
    file_chunk = file_list[a : a + chunk_size]
    matrix = np.load(file_chunk[0])
    for f in file_chunk[1:]:
        try:
            matrixx = np.load(f)
            if len(matrixx) == 0: continue
            assert matrixx.shape[1] == 20 * kmer + 1
            matrix = np.append(matrix, matrixx, axis=0)
        except Exception as e:
            print("{} is not a valid npy file".format(f))
            error_type = type(e).__name__
            tb = traceback.format_exc()
            print(f"Error type: {error_type}")
            print("Traceback details:")
            print(tb)
            continue
    write_file = os.path.join(merge_dir , "chunk_{}".format(int(a / chunk_size)))
    np.random.shuffle(matrix)
    assert matrix.shape[1] == 20 * kmer + 1
    np.save(write_file, matrix)
    chunk_id = int(a / chunk_size)
    print("Written chunk_{} to file: {}".format(chunk_id, write_file))
    a += chunk_size