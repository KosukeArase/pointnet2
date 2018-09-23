import os, sys, glob, pickle
import numpy as np
from collections import defaultdict
from itertools import chain


root_dir = '/data/unagi0/dataset/S3DIS/Stanford3dDataset_v1.2_Aligned_Version'
class_names = ["ceiling", "floor", "wall", "beam", "column", "window", "door", "table", "chair", "sofa", "bookcase", "board", "clutter"]


def create_pkl(output_file, data_type):
    if data_type == 'train':
        dir_template = os.path.join(root_dir, 'Area_[1-5]/office_*/') # Annotations/{}_*.txt')
    elif data_type == 'test':
        dir_template = os.path.join(root_dir, 'Area_6/office_*/') # Annotations/{}_*.txt')
    else:
        raise ValueError('Invalid data_type {} was given to create_pkl()'.format(data_type))

    dirs = glob.glob(dir_template)
    pcs = []
    sems = []
    colors = []

    for i, d in enumerate(dirs): # for scene
        print('{}/{}'.format(i, len(dirs)))
        # d = '/data/unagi0/dataset/S3DIS/Stanford3dDataset_v1.2_Aligned_Version/Area_1/office_20/'
        scene_name = '_'.join(d.split('Stanford3dDataset_v1.2_Aligned_Version/')[-1].split('/')[:-1]) # Area_6_office_25
        file_template = os.path.join(d, 'Annotations', '{}_*.txt')
        lines = []
        classes = []

        for ind, class_name in enumerate(class_names):
            files = glob.glob(file_template.format(class_name))
            lines += list(chain.from_iterable([open(file, 'r').readlines() for file in files])) # '1.1 2.2 3.3 123 234 222'
            classes += [ind] * (len(lines) - len(classes))

        pc = np.array([map(float, line.split()[:3]) for line in lines], dtype=np.float16)
        color = np.array([map(float, line.split()[3:]) for line in lines], dtype=np.int16)
        sem = np.array(classes, dtype=np.int8)

        assert len(pc[1]) == 3
        assert len(pc) == len(sem)
        assert min(sem) == 0
        assert max(sem) == 12
        assert len(color[1]) == 3
        assert len(color) == len(sem)


        pcs.append(pc)
        sems.append(sem)
        colors.append(color)

    assert len(pcs) == len(dirs)
    assert len(sems) == len(dirs)
    assert len(colors) == len(dirs)

    with open(output_file.format(data_type), 'wb') as f:
        pickle.dump(pcs, f)
        pickle.dump(sems, f)
        pickle.dump(colors, f)

    with open('filelist.pkl', 'wb') as f:
        pickle.dump(dirs, f)


def main():
    data_type = sys.argv[1]
    output_file = '{}_color.pkl'

    create_pkl(output_file, data_type=data_type)


if __name__ == '__main__':
    main()
