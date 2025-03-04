import numpy as np
import os

if __name__ == "__main__":

    train_dir_1 = "/public1/YHC/QiTanTechData/YF6418_fe/merge"
    train_dir_2 = "/public1/YHC/QiTanTechData/YF6419_fe/merge"
    train_dir_save = "/public1/YHC/QiTanTechData/train"
    train_files_1 = [os.path.join(train_dir_1, x) for x in os.listdir(train_dir_1) if x.endswith("npy")]
    train_files_2 = [os.path.join(train_dir_2, x) for x in os.listdir(train_dir_2) if x.endswith("npy")]

    np.random.shuffle(train_files_1)
    np.random.shuffle(train_files_2)

    for i in range(min(len(train_files_1), len(train_files_2))):
        print(train_files_1[i],train_files_2[i])
        arr1 = np.load(train_files_1[i])
        arr2 = np.load(train_files_2[i])
        np.random.shuffle(arr1)
        np.random.shuffle(arr2)
        trim = min(arr1.shape[0], arr2.shape[0])
        arr1 = arr1[:trim]
        arr2 = arr2[:trim]
        arr1[:, -1] = 1
        arr2[:, -1] = 0
        arr3 = np.concatenate([arr1, arr2], axis=0)
        np.random.shuffle(arr3)
        save_file = os.path.join(train_dir_save, "chunk_{}".format(i))
        np.save(save_file, arr3)
        print("saved {}".format(save_file))

