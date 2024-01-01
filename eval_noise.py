#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 15/7/22 10:04 AM
# @Author  : Yating
# @File    : eval_noise.py

import os
import numpy as np
from datetime import datetime
import ast
import argparse

import torch
from torch.utils.data import DataLoader

from dataloaders.loader import MyTestDataset, batch_test_task_collate_test, MyTestDataset_NoiseInMetaTest
from models.mpti_learner import MPTILearner_V3
from utils.cuda_util import cast_cuda
from utils.logger import init_logger



def evaluate_metric(logger, pred_labels_list, gt_labels_list, label2class_list, test_classes):
    """
    :param pred_labels_list: a list of np array, each entry with shape (n_queries*n_way, num_points).
    :param gt_labels_list: a list of np array, each entry with shape (n_queries*n_way, num_points).
    :param test_classes: a list of np array, each entry with shape (n_way,)
    :return: iou: scaler
    """
    assert len(pred_labels_list) == len(gt_labels_list) == len(label2class_list)

    logger.cprint('*****Test Classes: {0}*****'.format(test_classes))

    NUM_CLASS = len(test_classes) + 1 # add 1 to consider background class
    gt_classes = [0 for _ in range(NUM_CLASS)]
    positive_classes = [0 for _ in range(NUM_CLASS)]
    true_positive_classes = [0 for _ in range(NUM_CLASS)]

    for i, batch_gt_labels in enumerate(gt_labels_list):
        batch_pred_labels = pred_labels_list[i] #(n_queries*n_way, num_points)
        label2class = label2class_list[i] #(n_way,)

        for j in range(batch_pred_labels.shape[0]):
            for k in range(batch_pred_labels.shape[1]):
                gt = int(batch_gt_labels[j, k])
                pred = int(batch_pred_labels[j,k])

                if gt == 0: # 0 indicate background class
                    gt_index = 0
                else:
                    gt_class = label2class[gt-1] # the ground truth class in the dataset
                    gt_index = test_classes.index(gt_class) + 1
                gt_classes[gt_index] += 1

                if pred == 0:
                    pred_index = 0
                else:
                    pred_class = label2class[pred-1]
                    pred_index = test_classes.index(pred_class) + 1
                positive_classes[pred_index] += 1

                true_positive_classes[gt_index] += int(gt == pred)

    iou_list = []
    for c in range(NUM_CLASS):
        iou = true_positive_classes[c] / float(gt_classes[c] + positive_classes[c] - true_positive_classes[c])
        logger.cprint('----- [class %d]  IoU: %f -----'% (c, iou))
        iou_list.append(iou)

    mean_IoU = np.array(iou_list[1:]).mean()

    return mean_IoU


def test_few_shot(test_loader, learner, logger, test_classes, path=None, eval=False):

    total_loss = 0

    predicted_label_total = []
    gt_label_total = []
    label2class_total = []
    # add
    clean_flag_list = []

    for batch_idx, (data, sampled_classes) in enumerate(test_loader):
        query_label = data[3]

        if torch.cuda.is_available():
            data = cast_cuda(data)

        query_pred, loss, accuracy = learner.test(data, sampled_classes, batch_idx, path=path, eval=eval) # path to save test record
        # query_pred, loss, accuracy = learner.test(data, sampled_classes)
        total_loss += loss.detach().item()

        if (batch_idx+1) % 50 == 0:
            logger.cprint('[Eval] Iter: %d | Loss: %.4f | %s' % ( batch_idx+1, loss.detach().item(), str(datetime.now())))
            # debug
            # print('----------------- cluster acc: {} -------------'.format(learner.model.acc))
            # print('----------------- original acc: {} ------------'.format(learner.model.original_acc))
            # print('----------------- clean count: {} -------------'.format(learner.model.clean_count))
            # print('----------------- acc=0: {} -------------'.format(learner.model.acc_0))
            # print('-------------------- shot-level clean ratio: {}'.format(learner.model.shot_level_clean_ratio))
        #compute metric for predictions
        predicted_label_total.append(query_pred.cpu().detach().numpy())
        gt_label_total.append(query_label.numpy())
        label2class_total.append(sampled_classes)
        # clean_flag_list.append(clean_flag)

    mean_loss = total_loss/len(test_loader)
    mean_IoU = evaluate_metric(logger, predicted_label_total, gt_label_total, label2class_total, test_classes)
    print(mean_IoU)

    return mean_loss, mean_IoU



def eval(args, test_data_path, clean_data_path, ReturnCluster, noise_ratio, noise_type):
    logger = init_logger(args.log_dir, args)
    logger.cprint('\n------------------- noise ratio= {}, noise type={} --------------------\n'.format(noise_ratio, noise_type))
    if args.phase == 'protoeval':
        learner = ProtoNet_learner(args, mode='test') # protonet + CCNS
    elif args.phase == 'mptieval':
        learner = MPTILearner(args, mode='test')
    elif args.phase == 'mptinoise_eval': # my method
        learner = MPTILearner_V3(args, mode='test')
    elif args.phase == 'transformereval':
        learner = ProtoNet_transformer_learner(args, mode='test')

    if noise_ratio > 0:
        # testdataset, assume noise in episode. use clean data path.
        TEST_DATASET = MyTestDataset_NoiseInMetaTest(clean_data_path, args.dataset, cvfold=args.cvfold,
                                                              num_episode_per_comb=args.n_episode_test,
                                                              n_way=args.n_way, k_shot=args.k_shot, n_queries=args.n_queries,
                                                              num_point=args.pc_npts, pc_attribs=args.pc_attribs,
                                                              mode='test', ReturnCluster=False,
                                                              noise_ratio=noise_ratio, noise_type=noise_type)
    else:
        # test_data_path: clean data path. clean meta-test.
        TEST_DATASET = MyTestDataset(clean_data_path, args.dataset, cvfold=args.cvfold,
                                      num_episode_per_comb=args.n_episode_test,
                                      n_way=args.n_way, k_shot=args.k_shot, n_queries=args.n_queries,
                                      num_point=args.pc_npts, pc_attribs=args.pc_attribs, mode='test', ReturnCluster=False)

    TEST_CLASSES = list(TEST_DATASET.classes)
    TEST_LOADER = DataLoader(TEST_DATASET, batch_size=1, shuffle=False, collate_fn=batch_test_task_collate_test)

    # path
    if args.save_path:
        path = os.path.join(args.model_checkpoint_path, '{}_{:.3f}_test_record'.format(noise_type, noise_ratio))
    else:
        path = None
    test_loss, mean_IoU = test_few_shot(TEST_LOADER, learner, logger, TEST_CLASSES, path=path, eval=True)

    logger.cprint('\n=====[TEST] Loss: %.4f | Mean IoU: %f =====\n' %(test_loss, mean_IoU))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--phase', type=str, default='mptinoise_eval', choices=['pretrain', 'finetune',
                                                                           'prototrain', 'protoeval',
                                                                           'mptitrain', 'mptieval', 'mptinoise_eval', 'transformereval'])
    parser.add_argument('--dataset', type=str, default='scannet', help='Dataset name: s3dis|scannet')
    parser.add_argument('--cvfold', type=int, default=0, help='Fold left-out for testing in leave-one-out setting '
                                                              'Options:{0,1}')
    parser.add_argument('--data_path', type=str, default='/home/yating/Documents/3d_segmentation/attMPTI-main/datasets/S3DIS/blocks_bs1_s1',
                        help='Directory to the noisy source data')
    parser.add_argument('--model_checkpoint_path', type=str, default='log_s3dis/Cleantrain/S0_N2_K5_Att1/mpti_WayContrast=0.1+FPS=4_V2_[0,0.2,0.4]',
                        help='Path to the checkpoint of model for resuming')
    parser.add_argument('--eval_interval', type=int, default=2000,
                        help='iteration/epoch inverval to evaluate model')


    # few-shot episode setting
    parser.add_argument('--n_way', type=int, default=2, help='Number of classes for each episode: 1|3')
    parser.add_argument('--k_shot', type=int, default=5, help='Number of samples/shots for each class: 1|5')
    parser.add_argument('--n_queries', type=int, default=1, help='Number of queries for each class')
    parser.add_argument('--n_episode_test', type=int, default=100,
                        help='Number of episode per configuration during testing')

    # Point cloud processing
    parser.add_argument('--pc_npts', type=int, default=2048, help='Number of input points for PointNet.')
    parser.add_argument('--pc_attribs', default='xyzrgbXYZ',
                        help='Point attributes fed to PointNets, if empty then all possible. '
                             'xyz = coordinates, rgb = color, XYZ = normalized xyz')
    parser.add_argument('--pc_augm', action='store_true', help='Training augmentation for points in each superpoint')
    parser.add_argument('--pc_augm_scale', type=float, default=0,
                        help='Training augmentation: Uniformly random scaling in [1/scale, scale]')
    parser.add_argument('--pc_augm_rot', type=int, default=1,
                        help='Training augmentation: Bool, random rotation around z-axis')
    parser.add_argument('--pc_augm_mirror_prob', type=float, default=0,
                        help='Training augmentation: Probability of mirroring about x or y axes')
    parser.add_argument('--pc_augm_jitter', type=int, default=1,
                        help='Training augmentation: Bool, Gaussian jittering of all attributes')

    # feature extraction network configuration
    parser.add_argument('--dgcnn_k', type=int, default=20, help='Number of nearest neighbors in Edgeconv')
    parser.add_argument('--edgeconv_widths', default='[[64,64], [64, 64], [64, 64]]', help='DGCNN Edgeconv widths')
    parser.add_argument('--dgcnn_mlp_widths', default='[512, 256]',
                        help='DGCNN MLP (following stacked Edgeconv) widths')
    parser.add_argument('--base_widths', default='[128, 64]', help='BaseLearner widths')
    parser.add_argument('--output_dim', type=int, default=64,
                        help='The dimension of the final output of attention learner or linear mapper')
    parser.add_argument('--use_attention', action='store_true', help='it incorporate attention learner')
    parser.add_argument('--dg_atten_dim', type=int, default=128, help='attention layer output dim.')

    # protoNet configuration
    parser.add_argument('--dist_method', default='gaussian',
                        help='Method to compute distance between query feature maps and prototypes.[Option: cosine|euclidean]')

    # MPTI configuration
    parser.add_argument('--n_subprototypes', type=int, default=100,
                        help='Number of prototypes for each class in support set')
    parser.add_argument('--k_connect', type=int, default=200,
                        help='Number of nearest neighbors to construct local-constrained affinity matrix')
    parser.add_argument('--sigma', type=float, default=1., help='hyeprparameter in gaussian similarity function')

    # noise config
    parser.add_argument('--noise_ratio', type=float, default=0.4, help='noise ratio in the support set')
    parser.add_argument('--clean_data_path', type=str,
                        default='/mnt/6202BA0A02B9E369/FewShot_Seg_datasets/datasets/ScanNet/blocks_bs1_s1',
                        help='in meta-test, always use clean data. thus, path is fixed')
    parser.add_argument('--ReturnCluster', default=True, help='whether return cluster indx')
    parser.add_argument('--noise_type', default='partial', type=str, help='noise type: sym, ood')
    parser.add_argument('--shot_seed', type=int, default=1, help='number of seed for each shot')
    parser.add_argument('--save_path', action='store_true', help='whether save test statis')

    # transformer confg:
    parser.add_argument('--d_model', type=int, default=192, help='transformer d_model')
    parser.add_argument('--n_head', type=int, default=4, help='transformer n head')
    parser.add_argument('--d_feed', type=int, default=128, help='transformer d_feed')
    parser.add_argument('--n_layers', type=int, default=1, help='number of transformer layers')

    args = parser.parse_args()

    args.edgeconv_widths = ast.literal_eval(args.edgeconv_widths)
    args.dgcnn_mlp_widths = ast.literal_eval(args.dgcnn_mlp_widths)
    args.base_widths = ast.literal_eval(args.base_widths)
    args.pc_in_dim = len(args.pc_attribs)

    args.log_dir = args.model_checkpoint_path

    eval(args, test_data_path=args.data_path, clean_data_path=args.clean_data_path, ReturnCluster=args.ReturnCluster, noise_ratio=args.noise_ratio, noise_type=args.noise_type)