import os, sys, glob, pickle, faiss
import numpy as np
from collections import defaultdict
from itertools import chain


root_dir = '/data/unagi0/dataset/S3DIS/Stanford3dDataset_v1.2_Aligned_Version'
class_names = ["ceiling", "floor", "wall", "beam", "column", "window", "door", "table", "chair", "sofa", "bookcase", "board", "clutter"]


def detect_border_gpu(pcs):
    dim = 3
    sem_all = np.array(list(chain.from_iterable([[i] * len(pc) for i, pc in enumerate(pcs)])))
    pcs_all = np.concatenate(pcs, axis=0).astype(np.float32)

    index = np.random.choice(len(pcs_all), int(len(pcs_all)/10), replace=False)
    pcs_chosen = pcs_all[index]
    sem_chosen = sem_all[index]

    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatL2(dim)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index.add(pcs_chosen)

    k = 1024
    nq = 100000
    thresh = 0.01
    border = []

    for j in range(int(len(pcs_all)/nq)+1):
        query = pcs_all[j*nq:(j+1)*nq]
        print("{} queries for {}th batch".format(len(query), j))
        D, I = gpu_index.search(query, k)
        for i, (dis, ind) in enumerate(zip(D, I)):
            neighbor = ind[dis<thresh]
            if len(np.unique(sem_chosen[neighbor])) > 1:
                border.append(j*nq+i)

    return np.array(border)


def create_pkl(output_dir, data_type):
    if data_type == 'train':
        dir_template = os.path.join(root_dir, 'Area_[1-5]/office_*/') # Annotations/{}_*.txt')
    elif data_type == 'test':
        dir_template = os.path.join(root_dir, 'Area_6/office_*/') # Annotations/{}_*.txt')
    else:
        raise ValueError('Invalid data_type {} was given to create_pkl()'.format(data_type))

    with open(os.path.join(output_dir, "s3dis_{}.pickle".format(data_type)), 'rb') as f:
        data = pickle.load(f)

    dirs = glob.glob(dir_template)
    borders = []

    for i, d in enumerate(dirs): # for scene
        print('{}/{}'.format(i, len(dirs)), d)
        # d = '/data/unagi0/dataset/S3DIS/Stanford3dDataset_v1.2_Aligned_Version/Area_1/office_20/'

        data_path = os.path.join(d, "Annotations")
        files = glob.glob(os.path.join(data_path, "*txt"))
        pcs = []
        for file in files:
            class_name = file.split("/")[-1].split("_")[0]
            if not class_name in class_names:
                continue
            with open(file, "r") as f:
                lines = f.readlines()
            pc = np.array([list(map(float, line.split()[:3])) for line in lines])
            pcs.append(pc)
        assert sum([len(pc) for pc in pcs]) == len(data[i])

        border = detect_border_gpu(pcs)
        borders.append(border)

    with open(os.path.join(output_dir, "{}_border.pkl".format(data_type)), 'wb') as f:
        pickle.dump(borders, f)


def main():
    splits = ["test", "train"]
    output_dir = "../data/s3dis_data_pointnet2"

    for split in splits:
        create_pkl(output_dir, split)


if __name__ == '__main__':
    main()
