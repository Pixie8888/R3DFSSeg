#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 13/7/22 4:46 PM
# @Author  : Yating
# @File    : mpti_train_noise.py
import os
import ast
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from eval_noise import test_few_shot
from dataloaders.loader import MyTestDataset, batch_test_task_collate, batch_test_task_collate_test, NoiseInMetaTest
from models.mpti_learner import MPTILearner_V3
from utils.cuda_util import cast_cuda
from utils.logger import init_logger





def train(args, clean_data_path):
    logger = init_logger(args.log_dir, args)


    #Init datasets, dataloaders, and writer
    PC_AUGMENT_CONFIG = {'scale': args.pc_augm_scale,
                         'rot': args.pc_augm_rot,
                         'mirror_prob': args.pc_augm_mirror_prob,
                         'jitter': args.pc_augm_jitter
                         }


    # train: artifically add noise to the noise shots in meta train:
    TRAIN_DATASET = NoiseInMetaTest(data_path=clean_data_path, dataset_name=args.dataset, cvfold=args.cvfold, num_episode=args.n_iters,
                                    n_way=args.n_way, k_shot=args.k_shot, n_queries=args.n_queries,
                                    phase=None, mode='train', num_point=args.pc_npts, pc_attribs=args.pc_attribs,
                                    pc_augm=args.pc_augm, pc_augm_config=PC_AUGMENT_CONFIG,
                                    ReturnCluster=False, noise_ratio=args.train_noise_ratio, noise_type='train')


    # test_data_path: clean data path. clean meta-test.
    VALID_DATASET = MyTestDataset(clean_data_path, args.dataset, cvfold=args.cvfold,
                                  num_episode_per_comb=args.n_episode_test,
                                  n_way=args.n_way, k_shot=args.k_shot, n_queries=args.n_queries,
                                  num_point=args.pc_npts, pc_attribs=args.pc_attribs, ReturnCluster=False)



    logger.cprint('--------- cvfold={}, train class: {}, test class: {} --------'.format(args.cvfold, TRAIN_DATASET.classes, VALID_DATASET.classes))

    VALID_CLASSES = list(VALID_DATASET.classes)
    training_class_order = np.sort(TRAIN_DATASET.classes)  # [1,2,5,6,7,9] from small to big

    TRAIN_LOADER = DataLoader(TRAIN_DATASET, batch_size=1, collate_fn=batch_test_task_collate)
    VALID_LOADER = DataLoader(VALID_DATASET, batch_size=1, collate_fn=batch_test_task_collate_test)
    # VALID_LOADER_noise = DataLoader(VALID_DATASET_noise, batch_size=1, collate_fn=batch_test_task_collate_test)

    WRITER = SummaryWriter(log_dir=args.log_dir)



    ## init model and optimizer
    MPTI = MPTILearner_V3(args, mode='train') # my method. mpti + ours

    # train
    best_iou = 0
    noise_best_iou = 0

    for batch_idx, (data, sampled_classes) in enumerate(TRAIN_LOADER):
        # for debug, check clean accuracy
        if batch_idx % 100 == 0:

            support_LP_clean_ratio = 0
            support_original_clean_ratio = 0

            Is_acc_record = 0.

            query_acc_LP_record = 0
            query_acc_original_record = 0
            query_acc_refine_record = 0.
            Iq_acc_record = 0.
            refine_fg_acc_record = 0.
            given_fg_acc_record = 0.
            gt_fg_ratio_record = 0.
            refine_fg_ratio_record = 0.
            given_fg_ratio_record = 0.

        if torch.cuda.is_available():
            data = cast_cuda(data)

        episode_class_idx = []
        for cls in sampled_classes:
            episode_class_idx.append(np.argwhere(training_class_order == cls)[0][0])

        out = MPTI.train(data, logger)

        loss, lp_loss, contrastive_loss, accuracy, query_acc_LP, query_acc_original, clean_ratio_LP_avg, original_clean_ratio = out

        query_acc_LP_record += query_acc_LP
        query_acc_original_record += query_acc_original
        support_LP_clean_ratio += clean_ratio_LP_avg
        support_original_clean_ratio += original_clean_ratio

        logger.cprint('==[Train] Iter: %d | Loss: %.4f |  lp_loss: %.4f | contrast_loss: %.4f | Accuracy: %f  ==' % (batch_idx, loss, lp_loss, contrastive_loss, accuracy))
        WRITER.add_scalar('Train/loss', loss, batch_idx)
        WRITER.add_scalar('Train/lp_loss', lp_loss, batch_idx)
        WRITER.add_scalar('Train/edge_loss', contrastive_loss, batch_idx)
        WRITER.add_scalar('Train/accuracy', accuracy, batch_idx)
        # record clean detection acc
        if (batch_idx+1) % 100 == 0:
            WRITER.add_scalar('Train/support_LP_clean_ratio', support_LP_clean_ratio / 100, batch_idx // 100)  # estimate on the point level
            WRITER.add_scalar('Train/support_original_clean_ratio', support_original_clean_ratio / 100, batch_idx // 100)  # estimate on the point level
            WRITER.add_scalar('Train/Is_acc', Is_acc_record/100, batch_idx//100)

            # query
            WRITER.add_scalar('Train/query_acc_LP', query_acc_LP_record / 100, batch_idx // 100)
            WRITER.add_scalar('Train/query_acc_original', query_acc_original_record / 100, batch_idx // 100)
            WRITER.add_scalar('Train/query_acc_refine', query_acc_refine_record / 100, batch_idx //100)
            WRITER.add_scalar('Train/Iq_acc', Iq_acc_record / 100, batch_idx //100)
            WRITER.add_scalar('Train/refine_fg_acc', refine_fg_acc_record / 100, batch_idx //100)
            WRITER.add_scalar('Train/given_fg_acc', given_fg_acc_record / 100, batch_idx // 100)
            WRITER.add_scalar('Train/gt_fg_ratio', gt_fg_ratio_record / 100, batch_idx //100)
            WRITER.add_scalar('Train/refine_fg_ratio', refine_fg_ratio_record/100, batch_idx // 100)
            WRITER.add_scalar('Train/given_fg_ratio', given_fg_ratio_record/100, batch_idx //100)

        if (batch_idx+1) % args.eval_interval == 0:

            valid_loss, mean_IoU = test_few_shot(VALID_LOADER, MPTI, logger, VALID_CLASSES)
            logger.cprint('\n=====[VALID] Loss: %.4f | Mean IoU: %f  =====\n' % (valid_loss, mean_IoU))
            WRITER.add_scalar('Valid/loss', valid_loss, batch_idx)
            WRITER.add_scalar('Valid/meanIoU', mean_IoU, batch_idx)
            if mean_IoU > best_iou:
                best_iou = mean_IoU
                logger.cprint('*******************Model Saved*******************')
                save_dict = {'iteration': batch_idx + 1,
                             'model_state_dict': MPTI.model.state_dict(),
                             'optimizer_state_dict': MPTI.optimizer.state_dict(),
                             'loss': valid_loss,
                             'IoU': best_iou
                             }
                torch.save(save_dict, os.path.join(args.log_dir, 'checkpoint.tar'))

            save_dict = {'iteration': batch_idx + 1,
                         'model_state_dict': MPTI.model.state_dict(),
                         'optimizer_state_dict': MPTI.optimizer.state_dict(),
                         'loss': valid_loss,
                         'IoU': best_iou
                         }
            torch.save(save_dict, os.path.join(args.log_dir, 'checkpoint_{}.tar'.format(batch_idx+1)))

    WRITER.close()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--phase', type=str, default='mptitrain', choices=['pretrain', 'finetune',
                                                                           'prototrain', 'protoeval',
                                                                           'mptitrain', 'mptieval'])
    parser.add_argument('--dataset', type=str, default='s3dis', help='Dataset name: s3dis|scannet')
    parser.add_argument('--cvfold', type=int, default=0, help='Fold left-out for testing in leave-one-out setting '
                                                              'Options:{0,1}')

    parser.add_argument('--pretrain_checkpoint_path', type=str, default='/home/yating/Documents/3d_segmentation/attMPTI-main/log_s3dis/log_pretrain_s3dis_S0/checkpoint.tar',
                        help='Path to the checkpoint of pre model for resuming')
    parser.add_argument('--model_checkpoint_path', type=str, default=None,
                        help='Path to the checkpoint of model for resuming')
    parser.add_argument('--save_path', type=str, default='./log_s3dis/',
                        help='Directory to the save log and checkpoints')
    parser.add_argument('--eval_interval', type=int, default=2000,
                        help='iteration/epoch inverval to evaluate model')

    # optimization
    parser.add_argument('--batch_size', type=int, default=1, help='Number of samples/tasks in one batch')
    parser.add_argument('--n_workers', type=int, default=16, help='number of workers to load data')
    parser.add_argument('--n_iters', type=int, default=40000, help='number of iterations/epochs to train')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Model (eg. protoNet or MPTI) learning rate [default: 0.001]')
    parser.add_argument('--step_size', type=int, default=5000, help='Iterations of learning rate decay')
    parser.add_argument('--gamma', type=float, default=0.5, help='Multiplicative factor of learning rate decay')

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
    # parser.add_argument('--noise_ratio', type=float, default=0.4, help='noise ratio in the support set')
    parser.add_argument('--clean_data_path', type=str, default='/home/yating/Documents/3d_segmentation/attMPTI-main/datasets/S3DIS/blocks_bs1_s1_new_sp',
                        help='path to the clean_data: up to blocks_bs1_s1')
    parser.add_argument('--log_dir', type=str, default='debug')
    parser.add_argument('--ReturnCluster', default=False, help='whether use cluster label in epidoe')
    parser.add_argument('--seed', default=123, type=int, help='seed')
    parser.add_argument('--proto_path', default='log_s3dis/SymmetricNoise_0.4/pretrain_S0/pretrain_MixBlock+Atten=128/initial_proto_epoch10_4089.pkl', type=str, help='path to the intial global protoes')
    parser.add_argument('--num_spectra_group', default=4, type=int, help='number of spectra clusters.')
    parser.add_argument('--train_noise_ratio', default='[0.2]', help='noise ratio in support set of meta train')
    parser.add_argument('--shot_seed', type=int, default=1, help='number of seed points per shot in clean detection')


    args = parser.parse_args()

    args.edgeconv_widths = ast.literal_eval(args.edgeconv_widths)
    args.dgcnn_mlp_widths = ast.literal_eval(args.dgcnn_mlp_widths)
    args.base_widths = ast.literal_eval(args.base_widths)
    args.pc_in_dim = len(args.pc_attribs)
    args.train_noise_ratio = ast.literal_eval(args.train_noise_ratio) # should be a list

    args.log_dir = os.path.join(args.save_path, 'Cleantrain',
                                'S%d_N%d_K%d_Att%d' % (args.cvfold, args.n_way, args.k_shot, args.use_attention),
                                args.log_dir)


    # seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    # random.seed(args.seed)

    train(args, clean_data_path=args.clean_data_path)
