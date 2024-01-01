""" Data Loader for Generating Tasks

Author: Zhao Na, 2020
"""
import os
import random
import math
import glob
import numpy as np
import h5py as h5
import transforms3d
from itertools import  combinations
import copy
import torch
from torch.utils.data import Dataset
import open3d as o3d


def sample_K_pointclouds(data_path, num_point, pc_attribs, pc_augm, pc_augm_config, scan_names, sampled_class, sampled_classes,
                         is_support=False, use_label_noise=False, NoiseInFold=0, ReturnCluster=False, partial_noise=False):
    '''
    :param data_path: data path to sample 2048 pcd
    :param num_point:
    :param pc_attribs:
    :param pc_augm:
    :param pc_augm_config:
    :param scan_names: bolck name. to get 2048 pcd
    :param sampled_class: only for support. which class to get mask
    :param sampled_classes: for query. only sampled_classes has valid label. others will be treated as bg (0).
    :param is_support:
    :param use_label_noise: whther get noisy label as the pcd label
    :param NoiseInFold: noise in which fold
    :param ReturnCluster: whther use cluster in the model. If not ReturnCluster, cluster will be all 0.
    :return:
    '''

    ptclouds  = []
    labels = []
    gt_labels = []
    cluster_labels = []
    for scan_name in scan_names:
        ptcloud, label, gt_label, cluster_label = sample_pointcloud_universal(data_path, num_point, pc_attribs, pc_augm, pc_augm_config,
                                           scan_name, sampled_classes, sampled_class, support=is_support,
                                    use_label_noise=use_label_noise, NoiseInFold=NoiseInFold, ReturnCluster=ReturnCluster,
                                                                              partial_noise=partial_noise) # (2048, 9) (2048, 1)
        ptclouds.append(ptcloud)
        labels.append(label)
        gt_labels.append(gt_label)
        cluster_labels.append(cluster_label)

    ptclouds = np.stack(ptclouds, axis=0) # (Ns, 2048, 9)
    labels = np.stack(labels, axis=0) # (Ns, 2048)
    gt_labels = np.stack(gt_labels, axis=0)
    cluster_labels = np.stack(cluster_labels, axis=0) # (Ns, 2048).cluster label range >=0. No outliers!

    return ptclouds, labels, gt_labels, cluster_labels


def sample_pointcloud(data_path, num_point, pc_attribs, pc_augm, pc_augm_config, scan_name,
                      sampled_classes, sampled_class=0, support=False, random_sample=False, use_label_noise=False):
    '''
    :param use_label_noise: whther use noisy label to sample indices and generate label
    :param NoiseInFold: only used on the condition that use_label_noise == True
    :return:
    '''
    sampled_classes = list(sampled_classes) # pre-train: train classes.  train: 2-way classes name
    data = np.load(os.path.join(data_path, 'data', '%s.npy' %scan_name))
    N = data.shape[0] #number of points in this scan (BLOACK)

    # ------------------- get 2048 points indices: when not random sample, should consisder noisy label in sampling ------------------
    if random_sample:
        sampled_point_inds = np.random.choice(np.arange(N), num_point, replace=(N < num_point)) # random sample 2048 points in this block
    elif random_sample == False and use_label_noise == False: # clean meta-train dataset
        # If this point cloud is for support/query set, make sure that the sampled points contain target class
        valid_point_inds = np.nonzero(data[:,6] == sampled_class)[0]  # indices of points belonging to the sampled class. Here!! should consider noise

        if N < num_point:
            sampled_valid_point_num = len(valid_point_inds)
        else:
            valid_ratio = len(valid_point_inds)/float(N)
            sampled_valid_point_num = int(valid_ratio*num_point)

        sampled_valid_point_inds = np.random.choice(valid_point_inds, sampled_valid_point_num, replace=False)
        sampled_other_point_inds = np.random.choice(np.arange(N), num_point-sampled_valid_point_num, replace=(N<num_point))
        sampled_point_inds = np.concatenate([sampled_valid_point_inds, sampled_other_point_inds])
    elif random_sample == False and use_label_noise == True:
        # If this point cloud is for support/query set, make sure that the sampled points contain target class
        valid_point_inds = np.nonzero(data[:,7] == sampled_class)[0]  # indices of points belonging to the sampled class. Here!! should consider noise

        if N < num_point:
            sampled_valid_point_num = len(valid_point_inds)
        else:
            valid_ratio = len(valid_point_inds)/float(N)
            sampled_valid_point_num = int(valid_ratio*num_point)

        sampled_valid_point_inds = np.random.choice(valid_point_inds, sampled_valid_point_num, replace=False)
        sampled_other_point_inds = np.random.choice(np.arange(N), num_point-sampled_valid_point_num, replace=(N<num_point))
        sampled_point_inds = np.concatenate([sampled_valid_point_inds, sampled_other_point_inds])

    # --------- get point data (xyzrgbXYZ): only use xyzRGB information --------------------
    data = data[sampled_point_inds]
    xyz = data[:, 0:3]
    rgb = data[:, 3:6]

    xyz_min = np.amin(xyz, axis=0)
    xyz -= xyz_min
    if pc_augm:
        xyz = augment_pointcloud(xyz, pc_augm_config)
    if 'XYZ' in pc_attribs:
        xyz_min = np.amin(xyz, axis=0)
        XYZ = xyz - xyz_min
        xyz_max = np.amax(XYZ, axis=0)
        XYZ = XYZ/xyz_max

    ptcloud = []
    if 'xyz' in pc_attribs: ptcloud.append(xyz)
    if 'rgb' in pc_attribs: ptcloud.append(rgb/255.)
    if 'XYZ' in pc_attribs: ptcloud.append(XYZ)
    ptcloud = np.concatenate(ptcloud, axis=1) # (2048, 9)

    # -------- get POINT label  -----------
    if use_label_noise == False:
        labels = data[:, 6].astype(np.int)
    else:
        labels = data[:, 7].astype(np.int)

    if support:
        groundtruth = labels==sampled_class # binary label
    else:
        groundtruth = np.zeros_like(labels) # labels that only availabel in the given classes
        for i, label in enumerate(labels):
            if label in sampled_classes:
                groundtruth[i] = sampled_classes.index(label)+1

    return ptcloud, groundtruth


def sample_pointcloud_universal(data_path, num_point, pc_attribs, pc_augm, pc_augm_config, scan_name, sampled_classes, sampled_class=0,
                    support=False, random_sample=False, use_label_noise=False, NoiseInFold=-1, ReturnCluster=False, SamplePoints=True, partial_noise=False):
    '''
    data format: xyzrgb + clean + (NoiseFold0 + NoiseFold1)
    :param use_label_noise: whther use noisy label to sample indices and generate label
    :param NoiseInFold: only used on the condition that use_label_noise == True.
    :param ReturnCluster: whether use cluster label for epidose. if False, randomly use some value
    :param SamplePoints: whether sample 2048 pcd.
    :param partial_noise: whther generate partial noise. only used in the test.
    :return: pcd(2048,9), label(2048,), gt label(clean), cluster_label
    '''
    if use_label_noise == True:
        assert NoiseInFold != -1

    sampled_classes = list(sampled_classes)  # pre-train: train classes [1, 2, 5, 6, 7, 9].  train: 2-way classes name
    data = np.load(os.path.join(data_path, 'data', '%s.npy' % scan_name))
    N = data.shape[0]  # number of points in this scan (BLOACK)

    if SamplePoints == True:
        # ------------------- get 2048 points indices: when not random sample, should consisder noisy label in sampling ------------------
        if random_sample or partial_noise==True:
            sampled_point_inds = np.random.choice(np.arange(N), num_point, replace=(N < num_point))  # random sample 2048 points in this block
        elif random_sample == False and use_label_noise == False:  # clean meta-train dataset
            # If this point cloud is for support/query set, make sure that the sampled points contain target class
            valid_point_inds = np.nonzero(data[:, 6] == sampled_class)[0]  # indices of points belonging to the sampled class. Here!! should consider noise

            if N < num_point:
                sampled_valid_point_num = len(valid_point_inds)
            else:
                valid_ratio = len(valid_point_inds) / float(N)
                sampled_valid_point_num = int(valid_ratio * num_point)

            sampled_valid_point_inds = np.random.choice(valid_point_inds, sampled_valid_point_num, replace=False)
            sampled_other_point_inds = np.random.choice(np.arange(N), num_point - sampled_valid_point_num, replace=(N < num_point))
            sampled_point_inds = np.concatenate([sampled_valid_point_inds, sampled_other_point_inds])

        elif random_sample == False and use_label_noise == True:
            # If this point cloud is for support/query set, make sure that the sampled points contain target class
            if NoiseInFold == 0:
                valid_point_inds = np.nonzero(data[:, 7] == sampled_class)[0]  # indices of points belonging to the sampled class. Here!! should consider noise
            elif NoiseInFold == 1:
                valid_point_inds = np.nonzero(data[:, 8] == sampled_class)[0]
            else:
                print('should indicate NoiseInFold!')

            if N < num_point:
                sampled_valid_point_num = len(valid_point_inds)
            else:
                valid_ratio = len(valid_point_inds) / float(N)
                sampled_valid_point_num = int(valid_ratio * num_point)

            sampled_valid_point_inds = np.random.choice(valid_point_inds, sampled_valid_point_num, replace=False)
            sampled_other_point_inds = np.random.choice(np.arange(N), num_point - sampled_valid_point_num,
                                                        replace=(N < num_point))
            sampled_point_inds = np.concatenate([sampled_valid_point_inds, sampled_other_point_inds])
    else:
        # PRETRAIN inference, take a whole pcd into net.
        # sampled_point_inds = np.arange(N)
        if N < 2048:
            sampled_point_inds = np.arange(N)
        else:
            sampled_point_inds = np.random.choice(np.arange(N), 2048, replace=False)  # random sample 2048 points in this block
    # --------- get point data (xyzrgbXYZ): only use xyzRGB information --------------------
    data = data[sampled_point_inds]
    xyz = data[:, 0:3]
    rgb = data[:, 3:6]

    xyz_min = np.amin(xyz, axis=0)
    xyz -= xyz_min
    if pc_augm:
        xyz = augment_pointcloud(xyz, pc_augm_config)
    if 'XYZ' in pc_attribs:
        xyz_min = np.amin(xyz, axis=0)
        XYZ = xyz - xyz_min
        xyz_max = np.amax(XYZ, axis=0)
        XYZ = XYZ / xyz_max

    ptcloud = []
    if 'xyz' in pc_attribs: ptcloud.append(xyz)
    if 'rgb' in pc_attribs: ptcloud.append(rgb / 255.)
    if 'XYZ' in pc_attribs: ptcloud.append(XYZ)
    ptcloud = np.concatenate(ptcloud, axis=1)  # (2048, 9)

    # -------- get POINT label  -----------
    if use_label_noise == False:
        labels = data[:, 6].astype(np.int)
    elif NoiseInFold == 0:
        labels = data[:, 7].astype(np.int)
    elif NoiseInFold == 1:
        labels = data[:, 8].astype(np.int)  # noisy label in fold1
    else:
        print('should indicate NoiseInFold!!')

    if support:
        groundtruth = labels == sampled_class  # binary label
    else:
        groundtruth = np.zeros_like(labels)  # labels that only availabel in the given classes
        for i, label in enumerate(labels):
            if label in sampled_classes:
                groundtruth[i] = sampled_classes.index(label) + 1

    # generate partial noise inside 2048pcd. instance label: use [:,-1] to index!
    gt_fg_obj_list = np.unique(data[groundtruth][:,-1])
    if partial_noise == True:
        # # flip all fg to bg
        # groundtruth = np.zeros(data.shape[0], dtype=bool)

        obj_list = list(np.unique(data[:,-1])) # last dim is instance
        # only flip for shot containing multiple objects of different class for 2048pcd.
        if len(obj_list) > 1 and len(np.unique(data[:, 6])) > 1:
            select_obj = np.random.choice(obj_list, 1, replace=False)[0]
            obj_mask = data[:,-1] == select_obj
            obj_class = data[obj_mask][:,6][0] # get selected obj label
            # flip one bg without constrains
            while(obj_class == sampled_class):
                select_obj = np.random.choice(obj_list, 1, replace=False)[0]
                obj_mask = data[:, -1] == select_obj
                obj_class = data[obj_mask][:, 6][0]  # get selected obj label
            # flip its groundtruth to 1
            groundtruth[obj_mask] = True
            print('bg obj: {} pts'.format(np.sum(obj_mask)))

        # # constrain on the size of pcd.
        # print('obj in this shot are: {}'.format(obj_list))
        # if len(obj_list) > 1 and len(np.unique(data[:, 6])) > 1:
        #     select_obj = np.random.choice(obj_list, 1, replace=False)[0]
        #     obj_mask = data[:, -1] == select_obj
        #     obj_class = data[obj_mask][:, 6][0]  # get selected obj label
        #     bg_obj_size = []
        #     bg_obj_list = []
        #     # flip one bg without constrains
        #     while (obj_class == sampled_class) or np.sum(obj_mask) < 300:
        #         select_obj = np.random.choice(obj_list, 1, replace=False)[0]
        #         obj_mask = data[:, -1] == select_obj
        #         obj_class = data[obj_mask][:, 6][0]  # get selected obj label
        #
        #         # remove cur sample
        #         obj_list.remove(select_obj)
        #         # record
        #         if obj_class != sampled_class:
        #             bg_obj_list.append(select_obj)
        #             bg_obj_size.append(np.sum(obj_mask))
        #         # sample till the last obj
        #         if len(obj_list) == 0:
        #             break
        #
        #     if (obj_class == sampled_class) or np.sum(obj_mask) < 300:
        #         idx = np.argmax(bg_obj_size)
        #         print(idx)
        #         select_obj = bg_obj_list[idx]
        #         obj_mask = data[:, -1] == select_obj
        #         obj_class = data[obj_mask][:, 6][0]  # get selected obj label
        #     # flip its groundtruth to 1
        #     groundtruth[obj_mask] = True
        #     print('bg obj: {} pts'.format(np.sum(obj_mask)))




            # total_obj_mask = np.zeros(data.shape[0], dtype=bool)
            # while (np.sum(total_obj_mask) < 0.9 * np.sum(groundtruth)):
            #     select_obj = np.random.choice(obj_list, 1, replace=False)[0]
            #     obj_mask = data[:, -1] == select_obj
            #     obj_class = data[obj_mask][:, 6][0]  # get selected obj label
            #     while (obj_class == sampled_class):
            #         select_obj = np.random.choice(obj_list, 1, replace=False)[0]
            #         obj_mask = data[:, -1] == select_obj
            #         obj_class = data[obj_mask][:, 6][0]  # get selected obj label
            #     total_obj_mask = total_obj_mask | obj_mask
            #     obj_list.remove(select_obj)
            #     if np.sum(total_obj_mask) + np.sum(groundtruth) == data.shape[0]:
            #         break
            # # flip its groundtruth to 1
            # groundtruth[total_obj_mask] = True
            # print('bg obj: {} pts'.format(np.sum(total_obj_mask)))

        # randomly flip fg obj to bg: to reduce the number of fg points in the mask
        if random.uniform(0, 1) > 0.7:
            select_obj = np.random.choice(gt_fg_obj_list, 1)[0]
            # print(obj_list, select_obj)
            obj_mask = data[:, -1] == select_obj
            print('fg obj: {} pts'.format(np.sum(obj_mask)))
            groundtruth[obj_mask] = False
    # print(np.sum(groundtruth))
    assert np.sum(groundtruth) > 0






    # add gt labels for debug
    gt_labels = data[:, 6]
    if support:
        gt_groundtruth = gt_labels == sampled_class  # binary label
    else:
        gt_groundtruth = np.zeros_like(gt_labels)  # labels that only availabel in the given classes
        for i, gt_label in enumerate(gt_labels):
            if gt_label in sampled_classes:
                gt_groundtruth[i] = sampled_classes.index(gt_label) + 1

    # return cluster label: data[:,9]
    if ReturnCluster:
        if data.shape[1] == 10: # noisy data: xyzrgb+ clean+ NoiseINFold0 + NOiseInFold1 + cluster
            cluster_label = data[:, 9].astype(np.int)
        elif data.shape[1] == 8: # clean data: xyzrgb+ clean + cluster
            cluster_label = data[:,7].astype(np.int)
        elif data.shape[1] == 9: # xyzrgb + label + cluster + instance
            cluster_label = data[:,7].astype(np.int)

    else:
        cluster_label = np.zeros_like(gt_groundtruth, dtype=np.int) # creat dummy value

    return ptcloud, groundtruth, gt_groundtruth, cluster_label


def augment_pointcloud(P, pc_augm_config):
    """" Augmentation on XYZ and jittering of everything """
    M = transforms3d.zooms.zfdir2mat(1)
    if pc_augm_config['scale'] > 1:
        s = random.uniform(1 / pc_augm_config['scale'], pc_augm_config['scale'])
        M = np.dot(transforms3d.zooms.zfdir2mat(s), M)
    if pc_augm_config['rot'] == 1:
        angle = random.uniform(0, 2 * math.pi)
        M = np.dot(transforms3d.axangles.axangle2mat([0, 0, 1], angle), M)  # z=upright assumption
    if pc_augm_config['mirror_prob'] > 0:  # mirroring x&y, not z
        if random.random() < pc_augm_config['mirror_prob'] / 2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), M)
        if random.random() < pc_augm_config['mirror_prob'] / 2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 1, 0]), M)
    P[:, :3] = np.dot(P[:, :3], M.T)

    if pc_augm_config['jitter']:
        sigma, clip = 0.01, 0.05  # https://github.com/charlesq34/pointnet/blob/master/provider.py#L74
        P = P + np.clip(sigma * np.random.randn(*P.shape), -1 * clip, clip).astype(np.float32)
    return P


############################################### dataset for meta-training  ###############################################
class MyDataset(Dataset):
    # clean meta-train
    def __init__(self, data_path, dataset_name, cvfold=0, num_episode=50000, n_way=3, k_shot=5, n_queries=1,
                 phase=None, mode='train', num_point=4096, pc_attribs='xyz', pc_augm=False, pc_augm_config=None, ReturnCluster=True):
        '''
        :param data_path: use clean data path for meta-test! use clean class2scan !
        :param dataset_name:
        :param cvfold:
        :param num_episode:
        :param n_way:
        :param k_shot:
        :param n_queries:
        :param phase:
        :param mode:
        :param num_point:
        :param pc_attribs:
        :param pc_augm:
        :param pc_augm_config:
        :param ReturnCluster:
        '''
        super(MyDataset).__init__()
        self.data_path = data_path
        self.n_way = n_way # 2
        self.k_shot = k_shot # 1
        self.n_queries = n_queries # 1
        self.num_episode = num_episode # 40,000
        self.phase = phase
        self.mode = mode
        self.num_point = num_point # 2048
        self.pc_attribs = pc_attribs
        self.pc_augm = pc_augm
        self.pc_augm_config = pc_augm_config
        self.ReturnCluster = ReturnCluster

        if dataset_name == 's3dis':
            from dataloaders.s3dis import S3DISDataset
            self.dataset = S3DISDataset(cvfold, data_path)
        elif dataset_name == 'scannet':
            from dataloaders.scannet import ScanNetDataset
            self.dataset = ScanNetDataset(cvfold, data_path)
        else:
            raise NotImplementedError('Unknown dataset %s!' % dataset_name)

        if mode == 'train':
            self.classes = np.array(self.dataset.train_classes) # meta-train classes, ['door', 'floor', 'sofa', 'table', 'wall', 'window']
        elif mode == 'test':
            self.classes = np.array(self.dataset.test_classes)
        else:
            raise NotImplementedError('Unkown mode %s! [Options: train/test]' % mode)

        print('MODE: {0} | Classes: {1}'.format(mode, self.classes))
        self.class2scans = self.dataset.class2scans

    def __len__(self):
        return self.num_episode

    def __getitem__(self, index, n_way_classes=None):
        if n_way_classes is not None:
            sampled_classes = np.array(n_way_classes)
        else:
            sampled_classes = np.random.choice(self.classes, self.n_way, replace=False)

        support_ptclouds, support_masks, query_ptclouds, query_labels, support_clusters, query_clusters, gt_support_masks, gt_query_labels = self.generate_one_episode(sampled_classes)

        # if self.mode == 'train' and self.phase == 'metatrain':
        #     remain_classes = list(set(self.classes) - set(sampled_classes))
        #     try:
        #         sampled_valid_classes = np.random.choice(np.array(remain_classes), self.n_way, replace=False)
        #     except:
        #         raise NotImplementedError('Error! The number remaining classes is less than %d_way' %self.n_way)
        #
        #     valid_support_ptclouds, valid_support_masks, valid_query_ptclouds, \
        #                                     valid_query_labels = self.generate_one_episode(sampled_valid_classes)
        #
        #     return support_ptclouds.astype(np.float32), \
        #            support_masks.astype(np.int32), \
        #            query_ptclouds.astype(np.float32), \
        #            query_labels.astype(np.int64), \
        #            valid_support_ptclouds.astype(np.float32), \
        #            valid_support_masks.astype(np.int32), \
        #            valid_query_ptclouds.astype(np.float32), \
        #            valid_query_labels.astype(np.int64)
        # else:
        #     return support_ptclouds.astype(np.float32), \
        #            support_masks.astype(np.int32), \
        #            query_ptclouds.astype(np.float32), \
        #            query_labels.astype(np.int64), \
        #            sampled_classes.astype(np.int32)

        if self.mode == 'train': # addtionally return gt labels for support and query. for debug
            return support_ptclouds.astype(np.float32), \
                    support_masks.astype(np.int32), \
                   query_ptclouds.astype(np.float32), \
                   query_labels.astype(np.int64), \
                   sampled_classes.astype(np.int32),\
                    support_clusters.astype(np.int32),\
                    query_clusters.astype(np.int32),\
                    gt_support_masks.astype(np.int32),\
                    gt_query_labels.astype(np.int32)
        else:
            return support_ptclouds.astype(np.float32), \
                   support_masks.astype(np.int32), \
                   query_ptclouds.astype(np.float32), \
                   query_labels.astype(np.int64), \
                   sampled_classes.astype(np.int32), \
                   support_clusters.astype(np.int32), \
                   query_clusters.astype(np.int32), \
                   gt_support_masks.astype(np.int32)

    def generate_one_episode(self, sampled_classes):
        '''
        use clean class2scan to generate episode in the meta-test.
        :param sampled_classes:
        :return:
        '''
        support_ptclouds = []
        support_masks = []
        query_ptclouds = []
        query_labels = []
        # add
        gt_query_labels = []
        gt_support_masks = []
        support_clusters = []
        query_clusters = []


        black_list = []  # to store the sampled scan names, in order to prevent sampling one scan several times...
        for sampled_class in sampled_classes:
            all_scannames = self.class2scans[sampled_class].copy()
            if len(black_list) != 0:
                all_scannames = [x for x in all_scannames if x not in black_list]
            selected_scannames = np.random.choice(all_scannames, self.k_shot+self.n_queries, replace=False)
            black_list.extend(selected_scannames)
            query_scannames = selected_scannames[:self.n_queries]
            support_scannames = selected_scannames[self.n_queries:]

            query_ptclouds_one_way, query_labels_one_way, gt_query_labels_one_way, query_cluster_one_way = sample_K_pointclouds(self.data_path, self.num_point,
                                                                                self.pc_attribs, self.pc_augm,
                                                                                self.pc_augm_config,
                                                                                query_scannames,
                                                                                sampled_class,
                                                                                sampled_classes,
                                                                                is_support=False,
                                                                                use_label_noise=False,
                                                                                NoiseInFold=-1,
                                                                                ReturnCluster=self.ReturnCluster)

            support_ptclouds_one_way, support_masks_one_way, gt_support_masks_one_way, support_cluster_one_way = sample_K_pointclouds(self.data_path, self.num_point,
                                                                                self.pc_attribs, self.pc_augm,
                                                                                self.pc_augm_config,
                                                                                support_scannames,
                                                                                sampled_class,
                                                                                sampled_classes,
                                                                                is_support=True,
                                                                               use_label_noise=False,
                                                                               NoiseInFold=-1,
                                                                               ReturnCluster=self.ReturnCluster
                                                                               )

            query_ptclouds.append(query_ptclouds_one_way)
            query_labels.append(query_labels_one_way)
            support_ptclouds.append(support_ptclouds_one_way)
            support_masks.append(support_masks_one_way)
            # add cluster
            support_clusters.append(support_cluster_one_way)
            query_clusters.append(query_cluster_one_way)
            gt_support_masks.append(gt_support_masks_one_way)
            gt_query_labels.append(gt_query_labels_one_way)

        support_ptclouds = np.stack(support_ptclouds, axis=0) # (N_way, N_s, 2048, 9)
        support_masks = np.stack(support_masks, axis=0) # (N_way, N_s, 2048)
        query_ptclouds = np.concatenate(query_ptclouds, axis=0) # (N_way, 2048, 9)
        query_labels = np.concatenate(query_labels, axis=0) # (N_way, 2048)
        # gt
        gt_support_masks = np.stack(gt_support_masks, axis=0)
        gt_query_labels = np.concatenate(gt_query_labels, axis=0)
        # add cluster: if ReturnCluster, they are valid cluster label. Otherwise, all zeros
        support_clusters = np.stack(support_clusters, axis=0)  # (N-way, N_s, 2048)
        query_clusters = np.concatenate(query_clusters, axis=0)  # (N-way, 2048)
        if self.ReturnCluster:
            assert np.sum(support_clusters) != 0
            assert np.sum(query_clusters) != 0
        # print(support_ptclouds.shape, support_masks.shape, query_ptclouds.shape, query_labels.shape)
        return support_ptclouds, support_masks, query_ptclouds, query_labels, support_clusters, query_clusters, gt_support_masks, gt_query_labels

class NoiseInMetaTest(Dataset):
    # generate noise in k-shot meta-test. use clean class2scan to generate.
    def __init__(self, data_path, dataset_name, cvfold=0, num_episode=50000, n_way=3, k_shot=5, n_queries=1,
                 phase=None, mode='train', num_point=4096, pc_attribs='xyz', pc_augm=False, pc_augm_config=None,
                 ReturnCluster=False, noise_ratio=0.4, noise_type='sym'):
        super(NoiseInMetaTest).__init__()
        self.data_path = data_path # clean class2scan path. clean data_path
        self.n_way = n_way # 2
        self.k_shot = k_shot # 1
        self.n_queries = n_queries # 1
        self.num_episode = num_episode # 40,000
        # self.phase = phase
        self.mode = mode
        self.num_point = num_point # 2048
        self.pc_attribs = pc_attribs
        self.pc_augm = pc_augm
        self.pc_augm_config = pc_augm_config
        self.noise_ratio = noise_ratio
        self.noise_type = noise_type
        self.cvfold = cvfold
        self.ReturnCluster = ReturnCluster

        #### add noise in the train support set:
        if mode == 'train':
            self.noise_type = 'train'
            assert isinstance(self.noise_ratio, list) # should be a list

        if dataset_name == 's3dis':
            from dataloaders.s3dis import S3DISDataset
            self.dataset = S3DISDataset(cvfold, data_path) # clean data path. purpose: to get class2scan. so use S3DISDataset.
            # self.noise_pair_dict = {0:3, 1:1, 2:6, 3:0, 4:4, 5:11, 6:2, 7:7, 8:9, 9:8, 10:10, 11:5}
            # self.noise_pair_dict = {3:11,11:10, 10:0, 0:8, 8:4, 4:3}
        elif dataset_name == 'scannet':
            from dataloaders.scannet import ScanNetDataset
            self.dataset = ScanNetDataset(cvfold, data_path)
        else:
            raise NotImplementedError('Unknown dataset %s!' % dataset_name)

        if mode == 'train':
            self.classes = np.array(self.dataset.train_classes) # meta-train classes, ['door', 'floor', 'sofa', 'table', 'wall', 'window']
        elif mode == 'test':
            self.classes = np.array(self.dataset.test_classes)
        else:
            raise NotImplementedError('Unkown mode %s! [Options: train/test]' % mode)

        print('MODE: {0} | Classes: {1} | noise_ratio: {2}, noise_type: {3}'.format(mode, self.classes, self.noise_ratio, self.noise_type))
        self.class2scans = self.dataset.class2scans # clean. full class2scan
        print('noise type: {}'.format(self.noise_type))
    def __len__(self):
        return self.num_episode

    def __getitem__(self, index, n_way_classes=None):
        if n_way_classes is not None:
            sampled_classes = np.array(n_way_classes)
        else:
            sampled_classes = np.random.choice(self.classes, self.n_way, replace=False)

        support_ptclouds, support_masks, query_ptclouds, query_labels, support_clusters, query_clusters, gt_support_masks, gt_query_labels, bg_pcd_x, bg_pcd_y, support_flag = self.generate_one_episode(sampled_classes)

        # # add augmneted pcd
        # support_ptclouds, support_masks, query_ptclouds, query_labels, support_clusters, query_clusters, gt_support_masks, gt_query_labels, support_flag, pcd_1024, label_1024, pcd_cutout, label_cutout = self.generate_one_episode_Augment(sampled_classes)

        if self.mode == 'train': # addtionally return gt labels for support and query. for debug
            return support_ptclouds.astype(np.float32), \
                    support_masks.astype(np.int32), \
                   query_ptclouds.astype(np.float32), \
                   query_labels.astype(np.int64), \
                   sampled_classes.astype(np.int32),\
                    support_clusters.astype(np.int32),\
                    query_clusters.astype(np.int32),\
                    gt_support_masks.astype(np.int32),\
                    gt_query_labels.astype(np.int32),\
                    bg_pcd_x.astype(np.float32),\
                    bg_pcd_y.astype(np.int32),\
                    support_flag.astype(np.int32)
        else:
            return support_ptclouds.astype(np.float32), \
                   support_masks.astype(np.int32), \
                   query_ptclouds.astype(np.float32), \
                   query_labels.astype(np.int64), \
                   sampled_classes.astype(np.int32), \
                   support_clusters.astype(np.int32), \
                   query_clusters.astype(np.int32), \
                   gt_support_masks.astype(np.int32),\
                # support_flag.astype(np.int32)

    def generate_one_episode(self, sampled_classes):
        ''' for symmetric, noise can only come from its Nway!
        :param sampled_classes:
        :return: generate noist k-shot based on noise ratio and noise type (symmetric / assymetric)
        '''
        support_ptclouds = []
        support_masks = []
        query_ptclouds = []
        query_labels = []

        gt_query_labels = []
        support_clusters = []
        query_clusters = []
        # gt_support_cluster
        gt_support_masks = []
        support_flag = []
        black_list = []  # to store the sampled scan names, in order to prevent sampling one scan several times...


        # define num_noise_shot:
        if self.mode == 'train':
            tmp_noise = np.random.choice(self.noise_ratio) # should be a list
            num_noise_shot = int(round(self.k_shot * tmp_noise))
            print('noisy shots: {}'.format(num_noise_shot))
        else:
            num_noise_shot = int(round(self.k_shot * self.noise_ratio))

        # define noise_class_range
        if self.mode == 'test':
            if self.noise_type == 'sym' and self.mode == 'test':
                noise_class_range = sampled_classes
            elif self.noise_type == 'ood' and self.mode == 'test':
                noise_class_range = [cls for cls in self.classes if cls not in sampled_classes]
                print(noise_class_range, sampled_classes, self.classes)
            elif self.noise_type == 'partial':
                noise_class_range = None
            else:
                print('only implemeted symmetric noise and ood noise for meta-test support!')
        elif self.mode == 'train':
            noise_class_range = list(self.classes)

        for sampled_class in sampled_classes:
            all_scannames = self.class2scans[sampled_class].copy()
            if len(black_list) != 0:
                all_scannames = [x for x in all_scannames if x not in black_list]

            # get clean shot support + query
            clean_scanname = np.random.choice(all_scannames, self.k_shot-num_noise_shot+self.n_queries, replace=False)
            black_list.extend(clean_scanname)
            query_scannames = clean_scanname[:self.n_queries]
            clean_support_scannames = clean_scanname[self.n_queries:]
            # sample 2048pcd
            clean_support_ptclouds_one_way, clean_support_masks_one_way, clean_gt_support_masks_one_way, clean_support_cluster_one_way  = sample_K_pointclouds(self.data_path, self.num_point,
                                                                                    self.pc_attribs, self.pc_augm,
                                                                                   self.pc_augm_config,
                                                                                   clean_support_scannames,
                                                                                   sampled_class,
                                                                                   sampled_classes,
                                                                                   is_support=True,
                                                                                   use_label_noise=False,
                                                                                   NoiseInFold=-1,
                                                                                   ReturnCluster=self.ReturnCluster
                                                                                   )
            support_ptclouds_one_way = clean_support_ptclouds_one_way # (Ns, 2048, 9)
            support_masks_one_way = clean_support_masks_one_way # (Ns, 2048)
            gt_support_masks_one_way = clean_gt_support_masks_one_way # (Ns, 2048)
            support_cluster_one_way = clean_support_cluster_one_way # (Ns, 2048)

            query_ptclouds_one_way, query_labels_one_way, gt_query_labels_one_way, query_clutser_one_way = sample_K_pointclouds(self.data_path, self.num_point,
                                                                                self.pc_attribs, self.pc_augm,
                                                                                self.pc_augm_config,
                                                                                query_scannames,
                                                                                sampled_class,
                                                                                sampled_classes,
                                                                                is_support=False,
                                                                                use_label_noise=False,
                                                                                NoiseInFold=-1,
                                                                                ReturnCluster=self.ReturnCluster
                                                                                )


            # support_flag. record each shot's aboslute label
            way_support_flag = np.zeros(self.k_shot)
            way_support_flag[:len(clean_support_scannames)] = sampled_class

            # -------- get noisy shot: sample from way_noise_class_range --------
            if self.noise_type == 'pair':
                way_noise_class_range = [self.noise_pair_dict[sampled_class]]
            elif self.noise_type == 'partial':
                way_noise_class_range = [sampled_class] # only used in the testing stage
            else:
                way_noise_class_range = copy.deepcopy(noise_class_range) # including training stage

            for i in range(num_noise_shot):
                noise_class_dict = {cls: 0 for cls in way_noise_class_range}
                # get noisy_class
                if self.noise_type == 'pair' or self.noise_type == 'partial':
                    noisy_class = np.random.choice(way_noise_class_range, 1)[0]
                else:
                    noisy_class = sampled_class
                    while(noisy_class == sampled_class):
                        noisy_class = np.random.choice(way_noise_class_range, 1)[0]
                # get noisy scan
                cur_noisy_all_scannames = self.class2scans[noisy_class].copy()
                if len(black_list) != 0:
                    cur_noisy_all_scannames = [x for x in cur_noisy_all_scannames if x not in black_list]
                cur_noisy_scan = np.random.choice(cur_noisy_all_scannames, 1, replace=False)
                if self.noise_type == 'partial':
                    data = np.load(os.path.join(self.data_path, 'data', '%s.npy' % cur_noisy_scan[0]))
                    num_obj = len(np.unique(data[:, -1]))
                    num_cls = len(np.unique(data[:, 6]))
                    while (num_obj < 3) or (num_cls < 3):
                        cur_noisy_scan = np.random.choice(cur_noisy_all_scannames, 1, replace=False)
                        data = np.load(os.path.join(self.data_path, 'data', '%s.npy' % cur_noisy_scan[0]))
                        num_obj = len(np.unique(data[:, -1]))
                        num_cls = len(np.unique(data[:, 6]))


                black_list.extend(cur_noisy_scan)
                # sample 2048 pcd
                noise_support_ptclouds_one_way, noise_support_masks_one_way, noise_gt_support_masks_one_way, noise_support_cluster_one_way = sample_K_pointclouds(
                                                                                                    self.data_path, self.num_point,
                                                                                                    self.pc_attribs, self.pc_augm,
                                                                                                    self.pc_augm_config,
                                                                                                    cur_noisy_scan,
                                                                                                    noisy_class, # use noisy_class !!!! class for this block
                                                                                                    sampled_classes, # no use for support
                                                                                                    is_support=True,
                                                                                                    use_label_noise=False,
                                                                                                    NoiseInFold=-1,
                                                                                                    ReturnCluster=self.ReturnCluster,
                                                                                                    partial_noise=self.noise_type == 'partial'
                                                                                                    ) # (1, 2048, 9)
                support_ptclouds_one_way = np.concatenate([support_ptclouds_one_way, noise_support_ptclouds_one_way], axis=0)
                support_masks_one_way = np.concatenate([support_masks_one_way, noise_support_masks_one_way], axis=0)
                gt_support_masks_one_way = np.concatenate([gt_support_masks_one_way, noise_gt_support_masks_one_way], axis=0) # (k, 2048)
                support_cluster_one_way = np.concatenate([support_cluster_one_way, noise_support_cluster_one_way], axis=0)
                # check noise dict, should not outnumber clean
                noise_class_dict[noisy_class] += 1
                if noise_class_dict[noisy_class] == self.k_shot - num_noise_shot - 1:
                    way_noise_class_range.remove(noisy_class)
                    print('remove: {}, left: {}'.format(noisy_class, way_noise_class_range))

                # support flag
                way_support_flag[len(clean_support_scannames) + i] = noisy_class

            # set noisy support's gt_mask to 0
            # if num_noise_shot > 0 and noisy_class != sampled_class: # some pair noise don't have noisy class
            #     gt_support_masks_one_way[-num_noise_shot:] = 0
            if num_noise_shot > 0:
                if self.noise_type == 'pair' and noisy_class != sampled_class:
                    gt_support_masks_one_way[-num_noise_shot:] = 0
                else:
                    gt_support_masks_one_way[-num_noise_shot:] = 0

            assert len(support_ptclouds_one_way) == self.k_shot
            # puturbe order in the clean support scannames
            order = np.arange(self.k_shot)
            np.random.shuffle(order) # in-place operation
            support_ptclouds_one_way = support_ptclouds_one_way[order]
            support_masks_one_way = support_masks_one_way[order]
            gt_support_masks_one_way = gt_support_masks_one_way[order]
            support_cluster_one_way = support_cluster_one_way[order]
            # # add noise flag
            # way_support_noise_flag = np.ones(5)
            # way_support_noise_flag[-2:] = 0
            # print('before: {}'.format(way_support_noise_flag))
            # way_support_noise_flag = way_support_noise_flag[order]
            # print('after: {}, order: {}'.format(way_support_noise_flag, order))
            # support flag
            way_support_flag = way_support_flag[order] # (5, )





            query_ptclouds.append(query_ptclouds_one_way)
            query_labels.append(query_labels_one_way)
            support_ptclouds.append(support_ptclouds_one_way)
            support_masks.append(support_masks_one_way)
            # add cluster
            support_clusters.append(support_cluster_one_way)
            query_clusters.append(query_clutser_one_way)
            gt_support_masks.append(gt_support_masks_one_way)
            gt_query_labels.append(gt_query_labels_one_way)
            # add support flag
            support_flag.append(way_support_flag)

        support_ptclouds = np.stack(support_ptclouds, axis=0) # (N_way, N_s, 2048, 9)
        support_masks = np.stack(support_masks, axis=0) # (N_way, N_s, 2048)
        query_ptclouds = np.concatenate(query_ptclouds, axis=0) # (N_way, 2048, 9)
        query_labels = np.concatenate(query_labels, axis=0) # (N_way, 2048)
        # gt
        gt_support_masks = np.stack(gt_support_masks, axis=0)
        gt_query_labels = np.concatenate(gt_query_labels, axis=0)
        # add cluster: if ReturnCluster, they are valid cluster label. Otherwise, all zeros
        support_clusters = np.stack(support_clusters, axis=0)  # (N-way, N_s, 2048)
        query_clusters = np.concatenate(query_clusters, axis=0)  # (N-way, 2048)
        if self.ReturnCluster:
            assert np.sum(support_clusters) != 0
            assert np.sum(query_clusters) != 0

        # add support flag
        support_flag = np.stack(support_flag) # (nway, kshot). record each shot's absolute label

        # sample additional 60 bg pcd. 2 way=4*8, 3_way: 3*8
        bg_pcd_feat = []
        bg_pcd_label = []
        num_bg_cls = 4
        per_bg_sample = 1
        bg_cls_list = [cls for cls in self.classes if cls not in sampled_classes]
        num_bg_cls = np.minimum(num_bg_cls, len(bg_cls_list))
        print(num_bg_cls)
        for i in range(num_bg_cls):

            noisy_class = np.random.choice(bg_cls_list, 1)[0]
            bg_cls_list.remove(noisy_class)
            # get noisy scan
            cur_noisy_all_scannames = self.class2scans[noisy_class].copy()
            if len(black_list) != 0:
                cur_noisy_all_scannames = [x for x in cur_noisy_all_scannames if x not in black_list]
            cur_noisy_scan = np.random.choice(cur_noisy_all_scannames, per_bg_sample, replace=False)
            black_list.extend(cur_noisy_scan)
            # sample 2048 pcd
            noise_support_ptclouds_one_way, noise_support_masks_one_way, noise_gt_support_masks_one_way, noise_support_cluster_one_way = sample_K_pointclouds(
                self.data_path, self.num_point,
                self.pc_attribs, self.pc_augm,
                self.pc_augm_config,
                cur_noisy_scan,
                noisy_class,  # use noisy_class !!!! class for this block
                sampled_classes,  # no use for support
                is_support=True,
                use_label_noise=False,
                NoiseInFold=-1,
                ReturnCluster=self.ReturnCluster
            )  # (20, 2048, 9)
            bg_pcd_feat.append(noise_support_ptclouds_one_way)
            bg_pcd_label.append(noise_support_masks_one_way)
        bg_pcd_feat = np.concatenate(bg_pcd_feat, axis=0) # (n, 2048, 9)
        bg_pcd_label = np.concatenate(bg_pcd_label, axis=0) # (n, 2048)

        print(support_flag)
        return support_ptclouds, support_masks, query_ptclouds, query_labels, support_clusters, query_clusters, gt_support_masks, gt_query_labels, bg_pcd_feat, bg_pcd_label, support_flag

    def generate_one_episode_Augment(self, sampled_classes):
        ''' for symmetric, noise can only come from its Nway! generate with augmented samples.
        :param sampled_classes:
        :return: generate noist k-shot based on noise ratio and noise type (symmetric / assymetric)
        '''
        support_ptclouds = []
        support_masks = []
        query_ptclouds = []
        query_labels = []

        gt_query_labels = []
        support_clusters = []
        query_clusters = []
        # gt_support_cluster
        gt_support_masks = []
        support_flag = []
        black_list = []  # to store the sampled scan names, in order to prevent sampling one scan several times...

        # augmented samples: support + bg.
        pcd_1024 = []
        label_1024 = []
        pcd_cutout = []
        label_cutout = []
        cluster_cutout = []

        # define num_noise_shot:
        if self.mode == 'train':
            tmp_noise = np.random.choice(self.noise_ratio)  # should be a list
            num_noise_shot = int(round(self.k_shot * tmp_noise))
            print('noisy shots: {}'.format(num_noise_shot))
        else:
            num_noise_shot = int(round(self.k_shot * self.noise_ratio))

        # define noise_class_range
        if self.mode == 'test':
            if self.noise_type == 'sym' and self.mode == 'test':
                noise_class_range = sampled_classes
            elif self.noise_type == 'ood' and self.mode == 'test':
                noise_class_range = [cls for cls in self.classes if cls not in sampled_classes]
                print(noise_class_range, sampled_classes, self.classes)
            else:
                print('only implemeted symmetric noise and ood noise for meta-test support!')
        elif self.mode == 'train':
            noise_class_range = list(self.classes)

        for sampled_class in sampled_classes:
            all_scannames = self.class2scans[sampled_class].copy()
            if len(black_list) != 0:
                all_scannames = [x for x in all_scannames if x not in black_list]

            # get clean shot support + query
            clean_scanname = np.random.choice(all_scannames, self.k_shot - num_noise_shot + self.n_queries, replace=False)
            black_list.extend(clean_scanname)
            query_scannames = clean_scanname[:self.n_queries]
            clean_support_scannames = clean_scanname[self.n_queries:]
            # sample 2048pcd
            clean_support_ptclouds_one_way, clean_support_masks_one_way, clean_gt_support_masks_one_way, clean_support_cluster_one_way = sample_K_pointclouds(
                self.data_path, self.num_point,
                self.pc_attribs, self.pc_augm,
                self.pc_augm_config,
                clean_support_scannames,
                sampled_class,
                sampled_classes,
                is_support=True,
                use_label_noise=False,
                NoiseInFold=-1,
                ReturnCluster=self.ReturnCluster
                )
            support_ptclouds_one_way = clean_support_ptclouds_one_way
            support_masks_one_way = clean_support_masks_one_way
            gt_support_masks_one_way = clean_gt_support_masks_one_way
            support_cluster_one_way = clean_support_cluster_one_way

            # downsample 1024 pcd
            support1024_ptclouds_one_way, support1024_masks_one_way, _, _ = sample_K_pointclouds(
                self.data_path, 1024,
                self.pc_attribs, self.pc_augm,
                self.pc_augm_config,
                clean_support_scannames,
                sampled_class,
                sampled_classes,
                is_support=True,
                use_label_noise=False,
                NoiseInFold=-1,
                ReturnCluster=self.ReturnCluster
                )

            pcd_1024.append(support1024_ptclouds_one_way)
            label_1024.append(support1024_masks_one_way)
            pcd_cutout.append(clean_support_ptclouds_one_way)
            label_cutout.append(clean_support_masks_one_way) # (n, 2048)
            cluster_cutout.append(clean_support_cluster_one_way) # (n, 2048)


            # sample query
            query_ptclouds_one_way, query_labels_one_way, gt_query_labels_one_way, query_clutser_one_way = sample_K_pointclouds(
                self.data_path, self.num_point,
                self.pc_attribs, self.pc_augm,
                self.pc_augm_config,
                query_scannames,
                sampled_class,
                sampled_classes,
                is_support=False,
                use_label_noise=False,
                NoiseInFold=-1,
                ReturnCluster=self.ReturnCluster
                )

            # support_flag. record each shot's aboslute label
            way_support_flag = np.zeros(self.k_shot)
            way_support_flag[:len(clean_support_scannames)] = sampled_class

            # -------- get noisy shot: sample from way_noise_class_range --------
            if self.noise_type == 'pair':
                way_noise_class_range = [self.noise_pair_dict[sampled_class]]
            else:
                way_noise_class_range = copy.deepcopy(noise_class_range)

            for i in range(num_noise_shot):
                noise_class_dict = {cls: 0 for cls in way_noise_class_range}
                # get noisy_class
                if self.noise_type == 'pair':
                    noisy_class = np.random.choice(way_noise_class_range, 1)[0]
                else:
                    noisy_class = sampled_class
                    while (noisy_class == sampled_class):
                        noisy_class = np.random.choice(way_noise_class_range, 1)[0]
                # get noisy scan
                cur_noisy_all_scannames = self.class2scans[noisy_class].copy()
                if len(black_list) != 0:
                    cur_noisy_all_scannames = [x for x in cur_noisy_all_scannames if x not in black_list]
                cur_noisy_scan = np.random.choice(cur_noisy_all_scannames, 1, replace=False)
                black_list.extend(cur_noisy_scan)
                # sample 2048 pcd
                noise_support_ptclouds_one_way, noise_support_masks_one_way, noise_gt_support_masks_one_way, noise_support_cluster_one_way = sample_K_pointclouds(
                    self.data_path, self.num_point,
                    self.pc_attribs, self.pc_augm,
                    self.pc_augm_config,
                    cur_noisy_scan,
                    noisy_class,  # use noisy_class !!!! class for this block
                    sampled_classes,  # no use for support
                    is_support=True,
                    use_label_noise=False,
                    NoiseInFold=-1,
                    ReturnCluster=self.ReturnCluster
                )  # (1, 2048, 9)
                support_ptclouds_one_way = np.concatenate([support_ptclouds_one_way, noise_support_ptclouds_one_way], axis=0)
                support_masks_one_way = np.concatenate([support_masks_one_way, noise_support_masks_one_way], axis=0)
                gt_support_masks_one_way = np.concatenate([gt_support_masks_one_way, noise_gt_support_masks_one_way], axis=0)  # (k, 2048)
                support_cluster_one_way = np.concatenate([support_cluster_one_way, noise_support_cluster_one_way], axis=0)

                # sample 1024 pcd:
                noise_support1024_ptclouds_one_way, noise_support1024_masks_one_way, _, _ = sample_K_pointclouds(
                    self.data_path, 1024,
                    self.pc_attribs, self.pc_augm,
                    self.pc_augm_config,
                    cur_noisy_scan,
                    noisy_class,  # use noisy_class !!!! class for this block
                    sampled_classes,  # no use for support
                    is_support=True,
                    use_label_noise=False,
                    NoiseInFold=-1,
                    ReturnCluster=self.ReturnCluster
                    )  # (1, 1024, 9)
                pcd_1024.append(noise_support1024_ptclouds_one_way)
                label_1024.append(noise_support1024_masks_one_way)
                pcd_cutout.append(noise_support_ptclouds_one_way)
                label_cutout.append(noise_support_masks_one_way)
                cluster_cutout.append(noise_support_cluster_one_way)

                # check noise dict, should not outnumber clean
                noise_class_dict[noisy_class] += 1
                if noise_class_dict[noisy_class] == self.k_shot - num_noise_shot - 1:
                    way_noise_class_range.remove(noisy_class)
                    print('remove: {}, left: {}'.format(noisy_class, way_noise_class_range))

                # support flag
                way_support_flag[len(clean_support_scannames) + i] = noisy_class

            # set noisy support's gt_mask to 0
            if num_noise_shot > 0 and noisy_class != sampled_class:  # some pair noise don't have noisy class
                gt_support_masks_one_way[-num_noise_shot:] = 0

            assert len(support_ptclouds_one_way) == self.k_shot
            # puturbe order in the clean support scannames
            order = np.arange(self.k_shot)
            np.random.shuffle(order)  # in-place operation
            support_ptclouds_one_way = support_ptclouds_one_way[order]
            support_masks_one_way = support_masks_one_way[order]
            gt_support_masks_one_way = gt_support_masks_one_way[order]
            support_cluster_one_way = support_cluster_one_way[order]
            # # add noise flag
            # way_support_noise_flag = np.ones(5)
            # way_support_noise_flag[-2:] = 0
            # print('before: {}'.format(way_support_noise_flag))
            # way_support_noise_flag = way_support_noise_flag[order]
            # print('after: {}, order: {}'.format(way_support_noise_flag, order))
            # support flag
            way_support_flag = way_support_flag[order]  # (5, )




            query_ptclouds.append(query_ptclouds_one_way)
            query_labels.append(query_labels_one_way)
            support_ptclouds.append(support_ptclouds_one_way)
            support_masks.append(support_masks_one_way)
            # add cluster
            support_clusters.append(support_cluster_one_way)
            query_clusters.append(query_clutser_one_way)
            gt_support_masks.append(gt_support_masks_one_way)
            gt_query_labels.append(gt_query_labels_one_way)
            # add support flag
            support_flag.append(way_support_flag)


        support_ptclouds = np.stack(support_ptclouds, axis=0)  # (N_way, N_s, 2048, 9)
        support_masks = np.stack(support_masks, axis=0)  # (N_way, N_s, 2048)
        query_ptclouds = np.concatenate(query_ptclouds, axis=0)  # (N_way, 2048, 9)
        query_labels = np.concatenate(query_labels, axis=0)  # (N_way, 2048)
        # gt
        gt_support_masks = np.stack(gt_support_masks, axis=0)
        gt_query_labels = np.concatenate(gt_query_labels, axis=0)
        # add cluster: if ReturnCluster, they are valid cluster label. Otherwise, all zeros
        support_clusters = np.stack(support_clusters, axis=0)  # (N-way, N_s, 2048)
        query_clusters = np.concatenate(query_clusters, axis=0)  # (N-way, 2048)
        if self.ReturnCluster:
            assert np.sum(support_clusters) != 0
            assert np.sum(query_clusters) != 0

        # add support flag
        support_flag = np.stack(support_flag)  # (nway, kshot). record each shot's absolute label




        # sample additional 60 bg pcd. 2 way=4*8, 3_way: 3*8
        bg_pcd_feat = []
        bg_pcd_label = []
        num_bg_cls = 4
        per_bg_sample = 3
        bg_cls_list = [cls for cls in self.classes]
        # print(self.classes) # should always be the same.
        num_bg_cls = np.minimum(num_bg_cls, len(bg_cls_list))
        # print(num_bg_cls)
        for i in range(num_bg_cls):

            noisy_class = np.random.choice(bg_cls_list, 1)[0]
            bg_cls_list.remove(noisy_class)
            # get noisy scan
            cur_noisy_all_scannames = self.class2scans[noisy_class].copy()
            if len(black_list) != 0:
                cur_noisy_all_scannames = [x for x in cur_noisy_all_scannames if x not in black_list]
            cur_noisy_scan = np.random.choice(cur_noisy_all_scannames, per_bg_sample, replace=False)
            black_list.extend(cur_noisy_scan)
            # sample 2048 pcd
            noise_support_ptclouds_one_way, noise_support_masks_one_way, noise_gt_support_masks_one_way, noise_support_cluster_one_way = sample_K_pointclouds(
                self.data_path, self.num_point,
                self.pc_attribs, self.pc_augm,
                self.pc_augm_config,
                cur_noisy_scan,
                noisy_class,  # use noisy_class !!!! class for this block
                sampled_classes,  # no use for support
                is_support=True,
                use_label_noise=False,
                NoiseInFold=-1,
                ReturnCluster=self.ReturnCluster
            )  # (20, 2048, 9)
            pcd_cutout.append(noise_support_ptclouds_one_way)
            label_cutout.append(noise_support_masks_one_way)
            cluster_cutout.append(noise_support_cluster_one_way)

            # sample 1024 pcd
            noise_support1024_ptclouds_one_way, noise_support1024_masks_one_way, _, _ = sample_K_pointclouds(
                self.data_path, 1024,
                self.pc_attribs, self.pc_augm,
                self.pc_augm_config,
                cur_noisy_scan,
                noisy_class,  # use noisy_class !!!! class for this block
                sampled_classes,  # no use for support
                is_support=True,
                use_label_noise=False,
                NoiseInFold=-1,
                ReturnCluster=self.ReturnCluster
            )  # (20, 1024, 9)
            pcd_1024.append(noise_support1024_ptclouds_one_way)
            label_1024.append(noise_support1024_masks_one_way)




        # collect augmented samples
        pcd_1024 = np.concatenate(pcd_1024, axis=0) # (n,1024,9)
        label_1024 = np.concatenate(label_1024, axis=0) # (n, 1024)
        # cutcout
        pcd_cutout = np.concatenate(pcd_cutout, axis=0) # (n, 2048, 9)
        label_cutout = np.concatenate(label_cutout, axis=0) # (n, 2048)
        cluster_cutout = np.concatenate(cluster_cutout, axis=0) # (n, 2048)

        pcd_cutout, label_cutout = self.cut_out(pcd_cutout, label_cutout, cluster_cutout)

        # bg_pcd_feat = np.concatenate(bg_pcd_feat, axis=0)  # (n, 2048, 9)
        # bg_pcd_label = np.concatenate(bg_pcd_label, axis=0)  # (n, 2048)

        return support_ptclouds, support_masks, query_ptclouds, query_labels, support_clusters, query_clusters, gt_support_masks, gt_query_labels,\
               support_flag, pcd_1024, label_1024, pcd_cutout, label_cutout

    def cut_out(self, support_x, support_y, support_c):
        '''
        :param support_x: (n, 2048,d)
        :param support_y: (n, 2048)
        :param support_c: (n, 2048)
        :return:
        pcd: (n,2048,d)
        label: (n,2048)
        '''
        global_support_x = []
        global_support_y = []

        for i in range(support_c.shape[0]):

            cur_support_x = support_x[i, :, :]  # (2048, 9)
            cur_support_y = support_y[i, :]  # (2048, )
            cur_support_c = support_c[i, :]  # (2048, )
            #
            seg_ids, seg_counts = np.unique(cur_support_c[cur_support_y == 1], return_counts=True)
            # get largest seg
            # print(torch.max(seg_counts))
            if len(seg_ids) > 1:
                # random mask out a segment
                target_id = seg_ids[np.argmax(seg_counts)]
                tmp_support_y = copy.deepcopy(cur_support_y)
                tmp_support_y[cur_support_c == target_id] = 0  # set as bg
                tmp_support_x = copy.deepcopy(cur_support_x)
                tmp_support_x[cur_support_c == target_id, :] = 0.  # set value to 0
            else:
                tmp_support_y = copy.deepcopy(cur_support_y)
                tmp_support_x = copy.deepcopy(cur_support_x)

            global_support_x.append(tmp_support_x)
            global_support_y.append(tmp_support_y)


        global_support_y = np.stack(global_support_y, axis=0)
        global_support_x = np.stack(global_support_x, axis=0)

        return global_support_x, global_support_y


class MyDataset_NoiseTrain(Dataset):
    def __init__(self, data_path, dataset_name, cvfold=0, num_episode=50000, n_way=3, k_shot=5, n_queries=1,
                 phase=None, mode='train', num_point=4096, pc_attribs='xyz', pc_augm=False, pc_augm_config=None,
                 NoisySupport=True, NoisyQuery=True, NoiseInFold=None, clean_data_path=None, ReturnCluster=False):
        '''
        :param data_path: symmetric_0.5/blocks_bs1_s1/
        :param NoisySupport: whther support is noisy
        :param NoisyQuery: whther query is noisy
        :param NoiseInFold: label using which fold. Only effective when (NoisySupport or NoisyQuery == True)
        :param cvfold: decide test_classes.

        return: flexiblly generate (support, query) each can be noisy or not. also suitable for clean training. but only use noise.
        '''

        super(MyDataset_NoiseTrain).__init__()

        self.data_path = data_path
        self.n_way = n_way # 2
        self.k_shot = k_shot # 1
        self.n_queries = n_queries # 1
        self.num_episode = num_episode # 40,000
        self.phase = phase
        self.mode = mode
        self.num_point = num_point # 2048
        self.pc_attribs = pc_attribs
        self.pc_augm = pc_augm
        self.pc_augm_config = pc_augm_config

        self.clean_data_path = clean_data_path # only effective when support/query is clean.-> clean class2scan.
        self.ReturnCluster = ReturnCluster
        self.NoisySupport = NoisySupport
        self.NoisyQuery = NoisyQuery
        self.NoiseInFold = NoiseInFold # decide which label to use
        print('---- NoisySupport={}, NoisyQuery={}, NoiseInFold={} -----'.format(NoisySupport, NoisyQuery, NoiseInFold))
        # self.use_noisy_label = True # when feeding data to network, use noisy label!

        if dataset_name == 's3dis':
            from dataloaders.s3dis import S3DISDataset, S3DISDataset_NoiseInMetaTrain_Universal
            # cvfold decide which fold is test.
            if NoisySupport:
                self.support_dataset = S3DISDataset_NoiseInMetaTrain_Universal(cvfold, data_path, NoiseInFold=self.NoiseInFold)
            else:
                self.support_dataset = S3DISDataset(cvfold, self.clean_data_path) # clean
            if NoisyQuery:
                self.query_dataset = S3DISDataset_NoiseInMetaTrain_Universal(cvfold, data_path, NoiseInFold=self.NoiseInFold)
            else:
                self.query_dataset = S3DISDataset(cvfold, self.clean_data_path)


            # self.dataset = S3DISDataset_NoiseInMetaTrain(cvfold, data_path) # noisy data_path: datasets/S3DIS/scenes/symmetric_0.5_fold1/blocks_bs1_s1


        elif dataset_name == 'scannet':
            from dataloaders.scannet import ScanNetDataset
            self.dataset = ScanNetDataset(cvfold, data_path)
        else:
            raise NotImplementedError('Unknown dataset %s!' % dataset_name)

        if mode == 'train':
            self.classes = np.array(self.support_dataset.train_classes) # meta-train classes, ['door', 'floor', 'sofa', 'table', 'wall', 'window']
        elif mode == 'test':
            self.classes = np.array(self.support_dataset.test_classes) # support_dataset and query_dataset should have same class
        else:
            raise NotImplementedError('Unkown mode %s! [Options: train/test]' % mode)

        # load class2scan for support and query
        print('MODE: {0} | Classes: {1}'.format(mode, self.classes))
        self.support_class2scans = self.support_dataset.class2scans
        self.query_class2scans = self.query_dataset.class2scans
        # record for debug
        self.support_noise_outlier = 0 # support set noise ratio =1
        self.query_noise_outlier = 0
        self.support_noise = 0.
        self.query_noise = 0.
        self.class_count = {cls:0 for cls in self.classes}
        self.class_freq = {cls:0 for cls in self.classes}
        self.class_support_noise = {cls:0 for cls in self.classes}


    def __len__(self):
        return self.num_episode

    def __getitem__(self, index, n_way_classes=None):
        if n_way_classes is not None:
            sampled_classes = np.array(n_way_classes)
        else:
            sampled_classes = np.random.choice(self.classes, self.n_way, replace=False) # self.classes: [1,2,5,6,7,9]

        support_ptclouds, support_masks, query_ptclouds, query_labels, support_clusters, query_clusters, gt_support_masks, gt_query_labels = self.generate_one_episode(sampled_classes)

        # if self.mode == 'train' and self.phase == 'metatrain':
        #     remain_classes = list(set(self.classes) - set(sampled_classes))
        #     try:
        #         sampled_valid_classes = np.random.choice(np.array(remain_classes), self.n_way, replace=False)
        #     except:
        #         raise NotImplementedError('Error! The number remaining classes is less than %d_way' %self.n_way)
        #
        #     valid_support_ptclouds, valid_support_masks, valid_query_ptclouds, \
        #                                     valid_query_labels = self.generate_one_episode(sampled_valid_classes, self.use_noisy_label)
        #
        #     return support_ptclouds.astype(np.float32), \
        #            support_masks.astype(np.int32), \
        #            query_ptclouds.astype(np.float32), \
        #            query_labels.astype(np.int64), \
        #            valid_support_ptclouds.astype(np.float32), \
        #            valid_support_masks.astype(np.int32), \
        #            valid_query_ptclouds.astype(np.float32), \
        #            valid_query_labels.astype(np.int64)
        # else:
        #     return support_ptclouds.astype(np.float32), \
        #            support_masks.astype(np.int32), \
        #            query_ptclouds.astype(np.float32), \
        #            query_labels.astype(np.int64), \
        #            sampled_classes.astype(np.int32),\
        #             support_clusters.astype(np.int32),\
        #             query_clusters.astype(np.int32),\
        #             gt_support_masks.astype(np.int32),\
        #             gt_query_labels.astype(np.int32)

        if self.mode == 'train': # addtionally return gt labels for support and query. for debug
            return support_ptclouds.astype(np.float32), \
                    support_masks.astype(np.int32), \
                   query_ptclouds.astype(np.float32), \
                   query_labels.astype(np.int64), \
                   sampled_classes.astype(np.int32),\
                    support_clusters.astype(np.int32),\
                    query_clusters.astype(np.int32),\
                    gt_support_masks.astype(np.int32),\
                    gt_query_labels.astype(np.int32)
        else:
            return support_ptclouds.astype(np.float32), \
                   support_masks.astype(np.int32), \
                   query_ptclouds.astype(np.float32), \
                   query_labels.astype(np.int64), \
                   sampled_classes.astype(np.int32), \
                   support_clusters.astype(np.int32), \
                   query_clusters.astype(np.int32)

    def generate_one_episode(self, sampled_classes):
        support_ptclouds = []
        support_masks = []
        query_ptclouds = []
        query_labels = []
        gt_query_labels = []
        support_clusters = []
        query_clusters = []
        # gt_support_cluster
        gt_support_masks = []

        black_list = []  # to store the sampled scan names, in order to prevent sampling one scan several times...
        for sampled_class in sampled_classes:
            support_noise_ratio = 1.2  # re-define noise ratio for each class
            # sample support block name from support_class2scans
            # support_all_scannames = list((self.support_class2scans[sampled_class].copy()).keys()) # {'blk1': 0, 'block2': 1, }
            support_all_scannames = self.support_class2scans[sampled_class].copy()  # still a dictionary
            # thr 1: check block noise : {'blk1': 0, 'block2': 1, }. 0 is noise, 1 is clean

            while (support_noise_ratio > 1):
                # exclude bloaklist
                support_all_scannames = [x for x in support_all_scannames if x not in black_list]
                    # support_all_scannames = [x for x in support_all_scannames if x not in black_list]  # exclude those selected in the current episode
                select_support = np.random.choice(support_all_scannames, self.k_shot, replace=False)
                support_scannames = select_support[:]

                # sample points
                support_ptclouds_one_way, support_masks_one_way, gt_support_masks_one_way, support_cluster_one_way = sample_K_pointclouds(
                    self.data_path, self.num_point,
                    self.pc_attribs, self.pc_augm,
                    self.pc_augm_config,
                    support_scannames,
                    sampled_class,
                    sampled_classes,
                    is_support=True,
                    use_label_noise=self.NoisySupport,
                    NoiseInFold=self.NoiseInFold,
                    ReturnCluster=self.ReturnCluster)

                # check support mask noise ratio: for each class, mask: (k,2048), binary
                assert gt_support_masks_one_way.shape == support_masks_one_way.shape
                total_cls_points = np.count_nonzero(support_masks_one_way == 1)
                mask1 = gt_support_masks_one_way == 1
                mask2 = support_masks_one_way == 1
                total_gt_cls_points = np.count_nonzero(mask1 * mask2)
                support_noise_ratio = 1. - total_gt_cls_points / total_cls_points
                # print('class: {}, support mask noise: {}'.format(sampled_class, support_noise_ratio))

            black_list.extend(select_support)  # truly selected support
            # print('class: {}, support mask noise: {}'.format(sampled_class, support_noise_ratio))
            if support_noise_ratio >0.5 :
                self.support_noise_outlier += 1
                self.class_count[sampled_class] += 1
            self.class_freq[sampled_class] += 1
            self.class_support_noise[sampled_class] += support_noise_ratio
            self.support_noise += support_noise_ratio

            # ---------------------------------- sample query block name from query_class2scans ----------------------------
            query_all_scannames = self.query_class2scans[sampled_class].copy() # still a dictionary
            query_noise_ratio = 1.2
            while (query_noise_ratio > 1): # no constrain for query
                if len(black_list) != 0:
                    query_all_scannames = [x for x in query_all_scannames if x not in black_list] # if query_all_scannames is a dict, x is the keys. so it will become a list again.! # exclude those selected in the current episode
                select_query = np.random.choice(query_all_scannames, self.n_queries, replace=False)
                # black_list.extend(select_query)
                query_scannames = select_query[:]

                query_ptclouds_one_way, query_labels_one_way, gt_query_labels_one_way, query_clutser_one_way = sample_K_pointclouds(
                    self.data_path, self.num_point,
                    self.pc_attribs, self.pc_augm,
                    self.pc_augm_config,
                    query_scannames,
                    sampled_class,
                    sampled_classes,
                    is_support=False,
                    use_label_noise=self.NoisyQuery,
                    NoiseInFold=self.NoiseInFold,
                    ReturnCluster=self.ReturnCluster)

                assert gt_query_labels_one_way.shape == query_labels_one_way.shape
                cls_id = list(sampled_classes).index(sampled_class) + 1
                total_cls_points = np.count_nonzero(query_labels_one_way == cls_id)
                mask1 = gt_query_labels_one_way == cls_id
                mask2 = query_labels_one_way == cls_id
                total_gt_cls_points = np.count_nonzero(mask1 * mask2)
                query_noise_ratio = 1. - total_gt_cls_points / total_cls_points
                # print('class: {}, query label noise: {}'.format(sampled_class, noise_ratio))
            black_list.extend(select_query)
            # print('class: {}, query label noise: {}'.format(sampled_class, query_noise_ratio))

            query_ptclouds.append(query_ptclouds_one_way)
            query_labels.append(query_labels_one_way)
            support_ptclouds.append(support_ptclouds_one_way)
            support_masks.append(support_masks_one_way)
            # add cluster
            support_clusters.append(support_cluster_one_way)
            query_clusters.append(query_clutser_one_way)
            gt_support_masks.append(gt_support_masks_one_way)
            gt_query_labels.append(gt_query_labels_one_way)

        support_ptclouds = np.stack(support_ptclouds, axis=0)  # (N_way, N_s, 2048, 9)
        support_masks = np.stack(support_masks, axis=0)  # (N_way, N_s, 2048)
        query_ptclouds = np.concatenate(query_ptclouds, axis=0)  # (N_way, 2048, 9)
        query_labels = np.concatenate(query_labels, axis=0)  # (N_way, 2048)
        # gt
        gt_support_masks = np.stack(gt_support_masks, axis=0)
        gt_query_labels = np.concatenate(gt_query_labels, axis=0)
        # add cluster: if ReturnCluster, they are valid cluster label. Otherwise, all zeros
        support_clusters = np.stack(support_clusters, axis=0) # (N-way, N_s, 2048)
        query_clusters = np.concatenate(query_clusters, axis=0) # (N-way, 2048)
        if self.ReturnCluster:
            assert np.sum(support_clusters) != 0
            assert np.sum(query_clusters) !=0

        # # check query noise within episode: using all query in the episode. we have gt querylabel
        # for cls in sampled_classes:
        #     cls_id = list(sampled_classes).index(cls) + 1
        #     total_cls_points = np.count_nonzero(query_labels == cls_id)
        #     mask1 = gt_query_labels == cls_id
        #     mask2 = query_labels == cls_id
        #     total_gt_cls_points = np.count_nonzero(mask1 * mask2)
        #     # noise_ratio = 1. - total_gt_cls_points / total_cls_points # class-aware noise ratio
        #     noise_ratio = 1. - (gt_query_labels == query_labels).sum().astype(np.float64) / (self.n_way* self.num_point) # how many label are correct
        #     print('---all queries noise ratio for class {} is {}---'.format(cls, noise_ratio))
        #     # record
        #     if noise_ratio > 0.5:
        #         self.query_noise_outlier += 1
        #     self.query_noise += noise_ratio

        return support_ptclouds, support_masks, query_ptclouds, query_labels, support_clusters, query_clusters, gt_support_masks, gt_query_labels



def batch_train_task_collate(batch):
    task_train_support_ptclouds, task_train_support_masks, task_train_query_ptclouds, task_train_query_labels, \
    task_valid_support_ptclouds, task_valid_support_masks, task_valid_query_ptclouds, task_valid_query_labels = list(zip(*batch))

    task_train_support_ptclouds = np.stack(task_train_support_ptclouds)
    task_train_support_masks = np.stack(task_train_support_masks)
    task_train_query_ptclouds = np.stack(task_train_query_ptclouds)
    task_train_query_labels = np.stack(task_train_query_labels)
    task_valid_support_ptclouds = np.stack(task_valid_support_ptclouds)
    task_valid_support_masks = np.stack(task_valid_support_masks)
    task_valid_query_ptclouds = np.array(task_valid_query_ptclouds)
    task_valid_query_labels = np.stack(task_valid_query_labels)

    data = [torch.from_numpy(task_train_support_ptclouds).transpose(3,4), torch.from_numpy(task_train_support_masks),
            torch.from_numpy(task_train_query_ptclouds).transpose(2,3), torch.from_numpy(task_train_query_labels),
            torch.from_numpy(task_valid_support_ptclouds).transpose(3,4), torch.from_numpy(task_valid_support_masks),
            torch.from_numpy(task_valid_query_ptclouds).transpose(2,3), torch.from_numpy(task_valid_query_labels)]

    return data


################################################ Static Testing Dataset ################################################
# clean meta-test
class MyTestDataset(Dataset):
    def __init__(self, data_path, dataset_name, cvfold=0, num_episode_per_comb=100, n_way=3, k_shot=5, n_queries=1,
                       num_point=4096, pc_attribs='xyz', mode='valid', ReturnCluster=True):
        '''
        :param data_path: clean data path. save the episode file in the clean_data_path
        :param dataset_name:
        :param cvfold:
        :param num_episode_per_comb:
        :param n_way:
        :param k_shot:
        :param n_queries:
        :param num_point:
        :param pc_attribs:
        :param mode:
        Return: support_ptclouds, support_masks, query_ptclouds, query_labels, support_clusters, query_clusters,
        '''
        super(MyTestDataset).__init__()

        dataset = MyDataset(data_path, dataset_name, cvfold=cvfold, n_way=n_way, k_shot=k_shot, n_queries=n_queries,
                            mode='test', num_point=num_point, pc_attribs=pc_attribs, pc_augm=False, ReturnCluster=ReturnCluster)
        self.classes = dataset.classes # meta train classes

        if mode == 'valid':
            test_data_path = os.path.join(data_path, 'S_%d_N_%d_K_%d_episodes_%d_pts_%d' % (
                                                    cvfold, n_way, k_shot, num_episode_per_comb, num_point))
        elif mode == 'test':
            test_data_path = os.path.join(data_path, 'S_%d_N_%d_K_%d_test_episodes_%d_pts_%d' % (
                                                    cvfold, n_way, k_shot, num_episode_per_comb, num_point))
            # test_data_path = os.path.join(data_path, 'S_1_N_2_K_5_episodes_100_pts_2048')
        else:
            raise NotImplementedError('Mode (%s) is unknown!' %mode)

        print(test_data_path)
        if os.path.exists(test_data_path):
            self.file_names = glob.glob(os.path.join(test_data_path, '*.h5'))
            self.num_episode = len(self.file_names)
        else:
            print('Test dataset (%s) does not exist...\n Constructing...' %test_data_path)
            os.mkdir(test_data_path)

            class_comb = list(combinations(self.classes, n_way))  # [(),(),(),...]
            self.num_episode = len(class_comb) * num_episode_per_comb # total episode in the test

            episode_ind = 0
            self.file_names = [] # record each episode name
            for sampled_classes in class_comb:
                sampled_classes = list(sampled_classes)
                for i in range(num_episode_per_comb):
                    data = dataset.__getitem__(episode_ind, sampled_classes)
                    out_filename = os.path.join(test_data_path, '%d.h5' % episode_ind)
                    write_episode(out_filename, data)
                    self.file_names.append(out_filename)
                    episode_ind += 1
        print('--------------- meta-test use clean_class2scan, episode save in the clean_data_path !!!!! -------------------')

    def __len__(self):
        return self.num_episode

    def __getitem__(self, index):
        file_name = self.file_names[index]
        return read_episode(file_name)


# only for Noise in meta test.
class MyTestDataset_NoiseInMetaTest(Dataset):
    def __init__(self, data_path, dataset_name, cvfold=0, num_episode_per_comb=100, n_way=3, k_shot=5, n_queries=1,
                       num_point=4096, pc_attribs='xyz', mode='valid', ReturnCluster=False, noise_ratio=None, noise_type=None):
        '''
        :param data_path: clean
        :param dataset_name:
        :param cvfold:
        :param num_episode_per_comb:
        :param n_way:
        :param k_shot:
        :param n_queries:
        :param num_point:
        :param pc_attribs:
        :param mode:
        :param clean_class2scan_path: path to the clean class2scan. for episode training, we only use clean class2scan to generate episode.
        :param noise_ratio: noise ratio in episode.
        :param noise_type: noise type in episode
        '''
        super(MyTestDataset_NoiseInMetaTest).__init__()

        dataset = NoiseInMetaTest(data_path, dataset_name, cvfold=cvfold, n_way=n_way, k_shot=k_shot, n_queries=n_queries,
                            mode='test', num_point=num_point, pc_attribs=pc_attribs, pc_augm=False,
                            ReturnCluster=ReturnCluster, noise_ratio=noise_ratio, noise_type=noise_type)
        self.classes = dataset.classes # meta train classes

        if mode == 'valid':
            test_data_path = os.path.join(data_path, 'NoiseTest_%s_%f_S_%d_N_%d_K_%d_episodes_%d_pts_%d' % (
                                                    noise_type, noise_ratio, cvfold, n_way, k_shot, num_episode_per_comb, num_point))
        elif mode == 'test':
            test_data_path = os.path.join(data_path, 'NoiseTest_%s_%f_S_%d_N_%d_K_%d_test_episodes_%d_pts_%d' % (
                                                    noise_type, noise_ratio, cvfold, n_way, k_shot, num_episode_per_comb, num_point))
        else:
            raise NotImplementedError('Mode (%s) is unknown!' %mode)
        print(test_data_path)
        if os.path.exists(test_data_path):
            self.file_names = glob.glob(os.path.join(test_data_path, '*.h5'))
            self.num_episode = len(self.file_names)
        else:
            print('Test dataset (%s) does not exist...\n Constructing...' %test_data_path)
            os.mkdir(test_data_path)

            class_comb = list(combinations(self.classes, n_way))  # [(),(),(),...]
            self.num_episode = len(class_comb) * num_episode_per_comb # total episode in the test

            episode_ind = 0
            self.file_names = [] # record each episode name
            for sampled_classes in class_comb:
                sampled_classes = list(sampled_classes)
                for i in range(num_episode_per_comb):
                    data = dataset.__getitem__(episode_ind, sampled_classes)
                    out_filename = os.path.join(test_data_path, '%d.h5' % episode_ind)
                    write_episode(out_filename, data)
                    self.file_names.append(out_filename)
                    episode_ind += 1

    def __len__(self):
        return self.num_episode

    def __getitem__(self, index):
        file_name = self.file_names[index]
        return read_episode(file_name)


def batch_test_task_collate(batch):
    # additionally add cluster label. for train
    batch_support_ptclouds, batch_support_masks, batch_query_ptclouds, batch_query_labels, batch_sampled_classes, batch_support_cluster, batch_query_cluster, batch_gt_support_masks, batch_gt_query_labels, bg_pcd_feat, bg_pcd_label, support_flag = batch[0]

    data = [torch.from_numpy(batch_support_ptclouds).transpose(2,3), torch.from_numpy(batch_support_masks),
            torch.from_numpy(batch_query_ptclouds).transpose(1,2), torch.from_numpy(batch_query_labels.astype(np.int64)),
            torch.from_numpy(batch_support_cluster), torch.from_numpy(batch_query_cluster),
            torch.from_numpy(batch_gt_support_masks), torch.from_numpy(batch_gt_query_labels),
            torch.from_numpy(bg_pcd_feat).transpose(1,2), torch.from_numpy(bg_pcd_label),
            torch.from_numpy(support_flag)]

    return data, batch_sampled_classes


def batch_test_task_collate_test(batch):
    # additionally add cluster label. for vaidation and test
    batch_support_ptclouds, batch_support_masks, batch_query_ptclouds, batch_query_labels, batch_sampled_classes, batch_support_cluster, batch_query_cluster, batch_gt_support_masks = batch[0]
    # batch_support_ptclouds, batch_support_masks, batch_query_ptclouds, batch_query_labels, batch_sampled_classes, batch_support_cluster, batch_query_cluster = batch[0]
    data = [torch.from_numpy(batch_support_ptclouds).transpose(2,3), torch.from_numpy(batch_support_masks),
            torch.from_numpy(batch_query_ptclouds).transpose(1,2), torch.from_numpy(batch_query_labels.astype(np.int64)),
            torch.from_numpy(batch_support_cluster), torch.from_numpy(batch_query_cluster), torch.from_numpy(batch_gt_support_masks)]

    return data, batch_sampled_classes


def write_episode(out_filename, data):
    support_ptclouds, support_masks, query_ptclouds, query_labels, sampled_classes, support_clusters, query_clusters, gt_support_masks = data # add cluster label

    data_file = h5.File(out_filename, 'w')
    data_file.create_dataset('support_ptclouds', data=support_ptclouds, dtype='float32')
    data_file.create_dataset('support_masks', data=support_masks, dtype='int32')
    data_file.create_dataset('query_ptclouds', data=query_ptclouds, dtype='float32')
    data_file.create_dataset('query_labels', data=query_labels, dtype='int64')
    data_file.create_dataset('sampled_classes', data=sampled_classes, dtype='int32')
    # add cluster label
    data_file.create_dataset('support_clusters', data=support_clusters, dtype='int32')
    data_file.create_dataset('query_clusters', data=query_clusters, dtype='int32')
    # add gt support_mask
    data_file.create_dataset('gt_support_masks', data=gt_support_masks, dtype='int32')
    # # add support flag
    # data_file.create_dataset('support_flag', data=support_flag, dtype='int32')
    data_file.close()

    print('\t {0} saved! | classes: {1}'.format(out_filename, sampled_classes))


def read_episode(file_name):
    data_file = h5.File(file_name, 'r')
    support_ptclouds = data_file['support_ptclouds'][:]
    support_masks = data_file['support_masks'][:]
    query_ptclouds = data_file['query_ptclouds'][:]
    query_labels = data_file['query_labels'][:]
    sampled_classes = data_file['sampled_classes'][:]
    support_clusters = data_file['support_clusters'][:]
    query_clusters = data_file['query_clusters'][:]
    # add gt_support_mask
    gt_support_masks = data_file['gt_support_masks'][:]
    # support_flag = data_file['support_flag'][:] # only for qualitative analysis

    return support_ptclouds, support_masks, query_ptclouds, query_labels, sampled_classes, support_clusters, query_clusters, gt_support_masks


