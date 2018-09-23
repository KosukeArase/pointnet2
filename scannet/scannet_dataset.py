import pickle
import os
import numpy as np
import scene_util


class ScannetDataset():
    def __init__(self, root, npoints=8192, split='train', dataset='s3dis', num_classes=13):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_filename = os.path.join(self.root, '{}_{}.pickle'.format(dataset, split))
        self.border_filename = os.path.join(self.root, '{}_border.pkl'.format(split))
        self.data_length = 0

    def load_data(self):
        with open(self.data_filename, 'rb') as fp:
            scene_points_list = pickle.load(fp)
            semantic_labels_list = pickle.load(fp)
        with open(self.border_filename, 'rb') as fp:
            sparse_borders_list = pickle.load(fp)
            borders_list = []
            for i, border in enumerate(sparse_borders_list):
                var = np.zeros([len(semantic_labels_list[i]), 1])
                var[border] = 1
                borders_list.append(var)

        if self.split == 'train':
            labelweights = np.zeros(self.num_classes)
            for seg in semantic_labels_list:
                tmp, _ = np.histogram(seg, range(self.num_classes+1))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = 1/np.log(1.2+labelweights)
        elif self.split == 'test':
            self.labelweights = np.ones(self.num_classes)

        self.data_length = len(scene_points_list)

        return {
            "scene_points_list": scene_points_list,
            "semantic_labels_list": semantic_labels_list,
            "borders_list": borders_list,
            "virtual_smpidx": None,
            }

    def sample(self, point_set, semantic_seg, border):
        semantic_seg = semantic_seg.astype(np.int32)
        border = border.astype(np.int32)
        coordmax = np.max(point_set, axis=0)
        coordmin = np.min(point_set, axis=0)
        smpmin = np.maximum(coordmax-[1.5, 1.5, 3.0],  coordmin)
        smpmin[2] = coordmin[2]
        smpsz = np.minimum(coordmax-smpmin, [1.5, 1.5, 3.0])
        smpsz[2] = coordmax[2]-coordmin[2]
        isvalid = False
        for i in range(10):
            curcenter = point_set[np.random.choice(len(semantic_seg), 1)[0], :]
            curmin = curcenter-[0.75, 0.75, 1.5]
            curmax = curcenter+[0.75, 0.75, 1.5]
            curmin[2] = coordmin[2]
            curmax[2] = coordmax[2]
            curchoice = np.sum((point_set >= (curmin-0.2))*(point_set <= (curmax+0.2)), axis=1) == 3
            cur_point_set = point_set[curchoice, :]
            cur_semantic_seg = semantic_seg[curchoice]
            cur_border = border[curchoice]
            if len(cur_semantic_seg) == 0:
                continue
            mask = np.sum((cur_point_set >= (curmin-0.01))*(cur_point_set <= (curmax+0.01)), axis=1) == 3
            vidx = np.ceil((cur_point_set[mask, :]-curmin)/(curmax-curmin)*[31.0, 31.0, 62.0])
            vidx = np.unique(vidx[:, 0]*31.0*62.0+vidx[:, 1]*62.0+vidx[:, 2])
            isvalid = np.sum(cur_semantic_seg > 0)/len(cur_semantic_seg) >= 0.7 and len(vidx)/31.0/31.0/62.0 >= 0.02
            if isvalid:
                break
        choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True)
        point_set = cur_point_set[choice, :]
        semantic_seg = cur_semantic_seg[choice]
        border = cur_border[choice]
        mask = mask[choice]
        sample_weight = self.labelweights[semantic_seg]
        sample_weight *= mask
        return point_set, semantic_seg, border, sample_weight

    def __len__(self):
        return self.data_length


class ScannetDatasetWholeScene():
    def __init__(self, root, npoints=8192, split='train', dataset='s3dis', num_classes=13):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_filename = os.path.join(self.root, '{}_{}.pickle'.format(dataset, split))
        self.border_filename = os.path.join(self.root, '{}_border.pkl'.format(split))
        self.data_length = 0

    def load_data(self):
        with open(self.data_filename, 'rb') as fp:
            scene_points_list = pickle.load(fp)
            semantic_labels_list = pickle.load(fp)
        with open(self.border_filename, 'rb') as fp:
            sparse_borders_list = pickle.load(fp)
            borders_list = []
            for i, border in enumerate(sparse_borders_list):
                var = np.zeros([len(semantic_labels_list[i]), 1])
                var[border] = 1
                borders_list.append(var)

        if self.split == 'train':
            labelweights = np.zeros(self.num_classes)
            for seg in semantic_labels_list:
                tmp, _ = np.histogram(seg, range(self.num_classes+1))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = 1/np.log(1.2+labelweights)
        elif self.split == 'test':
            self.labelweights = np.ones(self.num_classes)

        self.data_length = len(scene_points_list)

        return {
            "scene_points_list": scene_points_list,
            "semantic_labels_list": semantic_labels_list,
            "borders_list": borders_list,
            "virtual_smpidx": None,
            }

    def sample(self, point_set_ini, semantic_seg_ini, border_ini):
        semantic_seg_ini = semantic_seg_ini.astype(np.int32)
        border_ini = border_ini.astype(np.int32)
        coordmax = np.max(point_set_ini, axis=0)
        coordmin = np.min(point_set_ini, axis=0)
        nsubvolume_x = np.ceil((coordmax[0]-coordmin[0])/1.5).astype(np.int32)
        nsubvolume_y = np.ceil((coordmax[1]-coordmin[1])/1.5).astype(np.int32)
        point_sets = list()
        semantic_segs = list()
        borders = list()
        sample_weights = list()
        for i in range(nsubvolume_x):
            for j in range(nsubvolume_y):
                curmin = coordmin+[i*1.5, j*1.5, 0]
                curmax = coordmin+[(i+1)*1.5, (j+1)*1.5, coordmax[2]-coordmin[2]]
                curchoice = np.sum((point_set_ini >= (curmin-0.2))*(point_set_ini<=(curmax+0.2)), axis=1)==3
                cur_point_set = point_set_ini[curchoice, :]
                cur_semantic_seg = semantic_seg_ini[curchoice]
                cur_border = border_ini[curchoice]
                if len(cur_semantic_seg)==0:
                    continue
                mask = np.sum((cur_point_set >= (curmin-0.001))*(cur_point_set<=(curmax+0.001)), axis=1)==3
                choice = np.random.choice(len(cur_semantic_seg),  self.npoints,  replace=True)
                point_set = cur_point_set[choice, :]  # Nx3
                semantic_seg = cur_semantic_seg[choice]  # N
                border = cur_border[choice]  # N
                mask = mask[choice]
                if sum(mask)/float(len(mask))<0.01:
                    continue
                sample_weight = self.labelweights[semantic_seg]
                sample_weight *= mask  # N
                point_sets.append(np.expand_dims(point_set, 0))  # 1xNx3
                semantic_segs.append(np.expand_dims(semantic_seg, 0))  # 1xN
                borders.append(np.expand_dims(border, 0))  # 1xN
                sample_weights.append(np.expand_dims(sample_weight, 0))  # 1xN
        point_sets = np.concatenate(tuple(point_sets), axis=0)
        semantic_segs = np.concatenate(tuple(semantic_segs), axis=0)
        borders = np.concatenate(tuple(borders), axis=0)
        sample_weights = np.concatenate(tuple(sample_weights), axis=0)
        return point_sets, semantic_segs, borders, sample_weights

    def __len__(self):
        return self.data_length


class ScannetDatasetVirtualScan():
    def __init__(self, root, npoints=8192, split='train', dataset='s3dis', num_classes=13):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_filename = os.path.join(self.root, '{}_{}.pickle'.format(dataset, split))
        self.border_filename = os.path.join(self.root, '{}_border.pkl'.format(split))
        self.smpidx_filename = os.path.join(self.root, '{}_{}_smpidx.pickle'.format(dataset, split))
        self.data_length = 0

    def load_data(self):
        with open(self.data_filename, 'rb') as fp:
            scene_points_list = pickle.load(fp)
            semantic_labels_list = pickle.load(fp)
        with open(self.border_filename, 'rb') as fp:
            sparse_borders_list = pickle.load(fp)
            borders_list = []
            for i, border in enumerate(sparse_borders_list):
                var = np.zeros([len(semantic_labels_list[i]), 1])
                var[border] = 1
                borders_list.append(var)

        if self.split == 'train':
            labelweights = np.zeros(self.num_classes)
            for seg in semantic_labels_list:
                tmp, _ = np.histogram(seg, range(self.num_classes+1))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = 1/np.log(1.2+labelweights)
        elif self.split == 'test':
            self.labelweights = np.ones(self.num_classes)

        self.data_length = len(scene_points_list)

        if os.path.exists(self.smpidx_filename):
            print('Load indexes for virtual scan.')
            with open(self.smpidx_filename, 'rb') as fp:
                virtual_smpidx = pickle.load(fp)
        else:
            print('Start creating indexes for virtual scan.')
            virtual_smpidx = self.__create_smpidx()
            print('End creating indexes for virtual scan.')

        return {
            "scene_points_list": scene_points_list,
            "semantic_labels_list": semantic_labels_list,
            "borders_list": borders_list,
            "virtual_smpidx": virtual_smpidx,
            }

    def __create_smpidx(self):
        virtual_smpidx = list()
        for point_set in self.scene_points_list:
            smpidx = list()
            for i in xrange(8):
                var = scene_util.virtual_scan(point_set,mode=i)
                smpidx.append(np.expand_dims(var, 0)) # 1xpoints
            virtual_smpidx.append(smpidx) # datax8xpoints

        assert len(virtual_smpidx) == len(self.scene_points_list)
        assert len(virtual_smpidx[0]) == 8

        with open(self.smpidx_filename,'wb') as fp:
            pickle.dump(virtual_smpidx, fp)

        return virtual_smpidx

    def __get_rotation_matrix(self, i):
        theta = (i-4)*np.pi/4.0    # Rotation about the pole (Z).
        phi = 0 #phi * 2.0 * np.pi     # For direction of pole deflection.
        z = 0 # z * 2.0 * deflection    # For magnitude of pole deflection.

        r = np.sqrt(z)
        V = (
            np.sin(phi) * r,
            np.cos(phi) * r,
            np.sqrt(2.0 - z))

        st = np.sin(theta)
        ct = np.cos(theta)

        R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

        # Construct the rotation matrix  ( V Transpose(V) - I ) R.
        M = (np.outer(V, V) - np.eye(3)).dot(R)
        return M


    def sample(self, point_set_ini, semantic_seg_ini, border_ini, smpidx, view_ind):
        semantic_seg_ini = semantic_seg_ini.astype(np.int32)
        border_ini = border_ini.astype(np.int32)
        sample_weight_ini = self.labelweights[semantic_seg_ini]
        xyz = point_set_ini.copy()

        assert len(smpidx) > (self.npoints/4.)

        point_set = point_set_ini[smpidx,:]
        semantic_seg = semantic_seg_ini[smpidx]
        border = border_ini[smpidx]
        sample_weight = sample_weight_ini[smpidx]

        choice = np.random.choice(len(semantic_seg), self.npoints, replace=True)
        point_set = point_set[choice,:] # Nx3
        semantic_seg = semantic_seg[choice] # N
        border = border[choice] # N
        sample_weight = sample_weight[choice] # N

        camloc = np.mean(xyz,axis=0)
        camloc[2] = 1.5
        view_dr = np.array([np.pi/4.*view_ind, 0])
        camloc[:2] -= np.array([np.cos(view_dr[0]),np.sin(view_dr[0])])
        point_set[:, :2] -= camloc[:2]

        r_rotation = self.__get_rotation_matrix(-view_ind+1)
        rotated = point_set.dot(r_rotation)

        return rotated, semantic_seg, border, sample_weight

    def __len__(self):
        return self.data_length
