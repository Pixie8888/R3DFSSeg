""" Multi-prototype transductive inference

Author: Zhao Na, 2020
"""
import numpy as np
import faiss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import fps

from models.dgcnn import DGCNN
from models.attention import SelfAttention
from torch_scatter import scatter_mean, scatter_add, scatter_max


class BaseLearner(nn.Module):
    """The class for inner loop."""
    def __init__(self, in_channels, params):
        super(BaseLearner, self).__init__()

        self.num_convs = len(params)
        self.convs = nn.ModuleList()

        for i in range(self.num_convs):
            if i == 0:
                in_dim = in_channels
            else:
                in_dim = params[i-1]
            self.convs.append(nn.Sequential(
                              nn.Conv1d(in_dim, params[i], 1),
                              nn.BatchNorm1d(params[i])))

    def forward(self, x):
        for i in range(self.num_convs):
            x = self.convs[i](x)
            if i != self.num_convs-1:
                x = F.relu(x)
        return x




class MPTI_SelfAtten(nn.Module):
    def __init__(self, args):
        super(MPTI_SelfAtten, self).__init__()
        # self.gpu_id = args.gpu_id
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.in_channels = args.pc_in_dim
        self.n_points = args.pc_npts
        self.use_attention = args.use_attention
        self.n_subprototypes = args.n_subprototypes
        self.k_connect = args.k_connect
        self.sigma = args.sigma

        self.n_classes = self.n_way+1

        self.encoder = DGCNN(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k)
        self.base_learner = BaseLearner(args.dgcnn_mlp_widths[-1], args.base_widths)

        if self.use_attention:
            self.att_learner = SelfAttention(args.dgcnn_mlp_widths[-1], args.output_dim)
        else:
            self.linear_mapper = nn.Conv1d(args.dgcnn_mlp_widths[-1], args.output_dim, 1, bias=False)

        self.feat_dim = args.edgeconv_widths[0][-1] + args.output_dim + args.base_widths[-1]
        # # # debug:
        # self.classes = [3, 11, 10, 0, 8, 4]
        self.acc = {1: 0., 0: 0.}
        self.original_acc = {1: 0., 0: 0.} # 1 is orignal_clean ratio>50%
        self.clean_count = {1: 0, 0: 0}
        self.acc_0 = 0
        self.shot_level_clean_ratio = 0

        # clean detection
        self.shot_seed = args.shot_seed
        # self.Self_Atten = SelfAttention_residual(in_channels=self.feat_dim)
        # self.binary_head = nn.Sequential(nn.Linear(self.feat_dim+3, 16), nn.ReLU(), nn.Linear(16, 2))

        #
        self.proj = nn.Linear(self.feat_dim, 128)
        # self.proj_2 = nn.Linear(self.feat_dim, 128)


    def Mean_pl_support_y(self, support_feat, support_y, gt_support_y, support_x, n_x=1, n_y=1, n_z=1):
        ''' nway clean detection
        :param support_feat: (nway, kshot, d, 2048)
        :param support_y: (nway, kshot, 2048)
        :param gt_support_y: for debug. not use in clean detection
        :param support_x: (nway, kshot, 9, 2048)
        :return: pl_support_y: [num_fg_in_way1, num_fg_in_way2] new mask for the fg of each way
        '''

        # voting
        pl_support_y = []
        # add flag
        flag = torch.zeros((self.n_way, self.k_shot), device=support_y.device)  # (nway, kshot) # record each shot is clean(1) or not(0)

        # for debug
        cosine_map_list = []
        cosine_sum_list = []
        mask_list = []
        gt_mask_list = []
        seed_len_list = []

        for way in range(self.n_way):
            seed_point_list = []
            point_assign = []  # per shot
            seed_len = []  # per shot
            for k in range(self.k_shot):
                fg_mask = support_y[way, k, :] == 1
                cur_support_feat = support_feat[way, k, :, :][:, fg_mask]  # (d, num_fg)
                cur_support_feat = cur_support_feat.transpose(1, 0)
                # grid sampling:
                spatial_feat = support_x[way, k, :, :][:, fg_mask].transpose(1, 0) # (num_fg, 9)
                seed_proto, assignments, num_seed = self.grid_sampling(spatial_feat, cur_support_feat, n_x=n_x, n_y=n_y, n_z=n_z)

                point_assign.append(assignments)  # (num_fg)
                seed_len.append(num_seed)
                seed_point_list.append(seed_proto)  # (num_seed, d)

            seed_point_list = F.normalize(torch.cat(seed_point_list, dim=0), p=2, dim=1)  # (num_seed, d)

            # mask out self-connect
            logits_mask = 1. - torch.eye(seed_point_list.shape[0], device=support_y.device, dtype=torch.float)

            # cosine_similarity
            cosine_map = torch.mm(seed_point_list, seed_point_list.transpose(1, 0))  # (num_seed, num_seed)
            # mask out self-connection
            cosine_map = cosine_map * logits_mask

            # add temperature
            if n_x == 1 and n_y==1 and n_z == 1:
                cosine_map = cosine_map.pow(3)

            cosine_sum = torch.sum(cosine_map, dim=1)  # (num_seed)
            mask = cosine_sum > torch.mean(cosine_sum)  # (num_seed,)

            # add gt
            gt_mask = torch.sum(gt_support_y[way], dim=-1) > 0
            # print(cosine_map)
            # print(cosine_sum)
            # print('mask: {}'.format(mask), 'gt_mask: {}'.format(gt_mask))
            # print('gt_mask: {}'.format(gt_mask))
            # debug
            cosine_map_list.append(cosine_map.detach().cpu().numpy())
            cosine_sum_list.append(cosine_sum.detach().cpu().numpy())
            mask_list.append(mask.detach().cpu().numpy())
            gt_mask_list.append(gt_mask.detach().cpu().numpy())
            seed_len_list.append(seed_len)

            way_pl_support_y = []
            count = 0

            for k in range(self.k_shot):
                cur_seed_mask = mask[count: count + seed_len[k]]
                # print(cur_seed_mask.float(), torch.mean(cur_seed_mask.float()))
                # majority voting to refine the mask for each shot:
                if torch.mean(cur_seed_mask.float()) > 0.5:
                    cur_seed_mask = torch.ones_like(cur_seed_mask)
                    flag[way, k] = 1
                else:
                    cur_seed_mask = torch.zeros_like(cur_seed_mask)
                    flag[way, k] = 0

                count += seed_len[k]
                cur_assign = point_assign[k]  # ()
                way_pl_support_y.append(cur_seed_mask[cur_assign])
            way_pl_support_y = torch.cat(way_pl_support_y, dim=0)  # (num_fg, )
            # print(way_pl_support_y.requires_grad)
            pl_support_y.append(way_pl_support_y)


        return pl_support_y, flag, cosine_map_list, cosine_sum_list, mask_list, gt_mask_list, seed_len_list

    def Mean_pl_support_y_multi_scale(self, support_feat, support_y, gt_support_y, support_x):
        '''
        :param support_feat: (nway, kshot, d, 2048)
        :param support_y: (nway, kshot, 2048)
        :param gt_support_y: for debug. not use in clean detection
        :param support_x: (nway, kshot, 9, 2048)
        :return: pl_support_y: [num_fg_in_way1, num_fg_in_way2]
        '''
        # average over 3 views.
        x_list = [1,2]
        y_list = [1,2]
        z_list = [1,1]
        num_scale = len(x_list)
        total_flag = []
        for i in range(num_scale):
            _, flag, _,_,_,_,_ = self.Mean_pl_support_y(support_feat, support_y, gt_support_y, support_x, n_x=x_list[i], n_y=y_list[i], n_z=z_list[i])

            total_flag.append(flag) # flag: (n,k)

        # multi-scale:
        # print(total_flag)
        total_flag = torch.stack(total_flag, dim=0) # (num_scale, nway, kshot)
        total_flag = torch.mean(total_flag, dim=0) # (nway, kshot)

        pl_support_y = []
        clean_flag = torch.ones((self.n_way, self.k_shot), device=support_y.device) # (n_way, k_shot)

        for way in range(self.n_way):
            way_total_flag = total_flag[way] # (k_shots)
            way_pl_support_y = []
            for k in range(self.k_shot):
                shot_pl_support_y = support_y[way,k][support_y[way,k]>0] # (fg, )
                if way_total_flag[k] < 0.5:
                    shot_pl_support_y = torch.zeros_like(shot_pl_support_y) # set to 0
                    clean_flag[way,k] = 0 # set to 0
                way_pl_support_y.append(shot_pl_support_y)
            way_pl_support_y = torch.cat(way_pl_support_y, dim=0) # (num_fg, )
            # check if it's all zero:
            if torch.sum(way_pl_support_y) == 0:
                # reset to all 1
                way_pl_support_y = torch.ones_like(way_pl_support_y) # (num_fg, )
                clean_flag[way] = torch.ones(self.k_shot, device=support_x.device)

            pl_support_y.append(way_pl_support_y)

        return pl_support_y, clean_flag


    def per_way_contrast_loss(self, support_feat, support_y, gt_support_y,
                          support_flag, fps_k=1, temp=0.1):
        ''' WayContrast: per way supervised contrative loss
        :param support_feat: (nway, kshot, d, 2048)
        :param gt_support_y: (nway, kshot, 2048)
        :param support_flag: (nway, kshot). abosulte label for each shots
        :param temp: scalar, temperature
        :param query_feat: (n_way*2048, feat_dim)
        :param query_y: (nway, 2048)
        :return: label: 1: cls1, 2: cls2, -1: bg/noise
        '''

        ele = support_flag[0, 0]
        total_loss = []
        if ele * self.k_shot == torch.sum(support_flag[0]):
            # clean support set.
            clean_flag = 1
        else:
            clean_flag = 0

        for way in range(self.n_way):
            # 1. get feat_list and label-list
            feat_list = []
            label_list = []
            for k in range(self.k_shot):
                fg_mask = support_y[way, k, :] == 1
                # print(fg_mask.shape)
                cur_support_feat = support_feat[way, k, :, :][:, fg_mask]  # (d, num_fg)
                # print(cur_support_feat.shape[1])
                cur_support_feat = cur_support_feat.transpose(1, 0)
                seed_proto, assignments, num_seed, seed_feat = self.getMutiplePrototypes(cur_support_feat, k=fps_k)
                feat_list.append(F.normalize(self.proj(seed_proto), p=2, dim=1)) # l2 norm !!(1, d)
                # gt label: is their absolute label.
                cur_label = torch.zeros(seed_proto.shape[0], dtype=torch.float, device=support_y.device) + support_flag[way,k] # (n, )
                label_list.append(cur_label)

            # check if all shots are clean, sample additional negative to make contrastive work
            if clean_flag == 1:
                # find another way
                if way < self.n_way - 1:
                    another_way = way + 1
                else:
                    another_way = 0
                # sample another 2 shots as negative
                for k in range(2):
                    fg_mask = support_y[another_way, k, :] == 1
                    # print(fg_mask.shape)
                    cur_support_feat = support_feat[another_way, k, :, :][:, fg_mask]  # (d, num_fg)
                    # print(cur_support_feat.shape[1])
                    cur_support_feat = cur_support_feat.transpose(1, 0)
                    seed_proto, assignments, num_seed, seed_feat = self.getMutiplePrototypes(cur_support_feat, k=fps_k)
                    feat_list.append(F.normalize(self.proj(seed_proto), p=2, dim=1))  # l2 norm !!(1, d)
                    # gt label: -1.
                    cur_label = torch.zeros(seed_proto.shape[0], dtype=torch.float, device=support_y.device) - 1.  # (n, )
                    label_list.append(cur_label)

            # stack
            feat_list = torch.cat(feat_list, dim=0)  # (K, d)
            label_list = torch.cat(label_list, dim=0)  # (K, )


            # 2. mask_out self-connect
            logits_mask = 1. - torch.eye(label_list.shape[0], device=support_y.device, dtype=torch.float)

            # get gt_mask (n*k, n*k)
            gt_mask = torch.eq(label_list.unsqueeze(1), label_list.unsqueeze(0)).float().cuda() # (n*k, n*k)
            # gt_mask[label_list == -1] = 0 # connection with noise is all 0
            gt_mask = gt_mask * logits_mask # remove self-connection

            # compute logits
            logits = torch.div(torch.matmul(feat_list, feat_list.T), temp) # (n*k, n*k)

            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask # (n*k, n*k)
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

            # # compute mean of log-likelihood over positive
            mean_log_prob_pos = (gt_mask * log_prob).sum(1) / gt_mask.sum(1) # (n*k,)
            # mean_log_prob_pos = mean_log_prob_pos[new_mask] # remove noise in contrative loss

            # loss
            loss = - mean_log_prob_pos
            loss = loss.mean()
            total_loss.append(loss)

        total_loss = sum(total_loss) / len(total_loss)

        return total_loss


    def grid_sampling(self, spatial_feat, cur_support_feat, n_x=2, n_y=2, n_z=1):
        '''
        :param spatial_feat: (n_fg, 9) xyzrgbXYZ
        :param cur_support_feat: (n_fg, d)
        :param n_x: how many grid along x axis
        :param n_y: how many grid along y axis
        :param n_z: how many grid along z
        :return:
            seed_proto: (n_seed, d): local aggregation
            assigmentL: (n_fg,) point 2 proto assignments
            num_seed: number of seed.
        '''
        # get bbox
        x_min = torch.min(spatial_feat[:,0])
        x_max = torch.max(spatial_feat[:,0])
        y_min = torch.min(spatial_feat[:,1])
        y_max = torch.max(spatial_feat[:,1])
        z_min = torch.min(spatial_feat[:,2])
        z_max = torch.max(spatial_feat[:,2])

        # compute stride
        d_x = (x_max - x_min) / n_x
        d_y = (y_max - y_min) / n_y
        d_z = (z_max - z_min) / n_z



        # compute start
        x_start = [x_min+i*d_x for i in range(n_x)]
        y_start = [y_min+i*d_y for i in range(n_y)]
        z_start = [z_min+i*d_z for i in range(n_z)]
        # print(x_start, y_start, z_start)

        seed_proto = []
        assignments = torch.zeros(spatial_feat.shape[0], dtype=torch.long, device=spatial_feat.device)

        # grid sampling
        count = 0
        for x in x_start:
            x_mask = (spatial_feat[:,0]>= x) * (spatial_feat[:,0] <= x+d_x)
            for y in y_start:
                y_mask = (spatial_feat[:,1] >= y) * (spatial_feat[:,1] <= y+d_y)
                for z in z_start:
                    z_mask = (spatial_feat[:,2] >= z) * (spatial_feat[:,2] <= z+d_z)
                    mask = x_mask * y_mask * z_mask # (n_fg, )

                    if torch.sum(mask) > 0:
                        # get feat
                        assert torch.sum(mask) > 0
                        selected = cur_support_feat[mask] # (n, d)
                        seed_proto.append(torch.mean(selected, dim=0, keepdim=True))
                        assignments[mask] = count
                        count += 1 # record how many protoes
        # print(len(seed_proto))
        seed_proto = torch.cat(seed_proto, dim=0) # (n, d)
        return seed_proto, assignments, seed_proto.shape[0]

    # check prototype cleanes
    def Check_Proto_Cleanness(self, gt_support_y, support_y, fg_assign):
        '''
        :param gt_support_y: [nway, k, 2048]
        :param support_y: [nway, k, 2048]
        :return: gt_Y: record the cleanness of each multi-proto. [multi-proto in way1, multi-proto in way 2]
        count_1: how many proto' cleaness = 1
        orignal_acc: orignal cleanness in support set. it is the cleaness of single proto.
        '''
        gt_Y = []
        count_1 = []
        orignal_acc = []
        # # for bg
        # bg_gt_Y = torch.ones(bg_proto_number, device=support_y.device, dtype=torch.int32) # bg is considered clean.
        # gt_Y.append(bg_gt_Y)

        for way in range(self.n_way):
            way_gt_support_y = gt_support_y[way]
            way_support_y = support_y[way]
            point_clean_indicator = way_gt_support_y[way_support_y==1].view(-1)
            way_fg_assign = fg_assign[way] # point2proto assignment. start from 0 to max
            assert way_fg_assign.shape[0] == point_clean_indicator.shape[0]
            way_gt_Y = scatter_mean(point_clean_indicator.float(), way_fg_assign.long())
            print(way_gt_Y)
            gt_Y.append(way_gt_Y)

            # how many proto is 1:
            way_count = torch.sum(way_gt_Y == 1)
            count_1.append(way_count)

            # orignal acc:
            way_orig_acc = gt_support_y[way][support_y[way]==1] == 1
            way_orig_acc = torch.sum(way_orig_acc) / torch.sum(support_y[way]==1)
            orignal_acc.append(way_orig_acc)

            # way_gt_Y = torch.where(way_gt_Y > 0.5,
            #             torch.tensor(1, dtype=torch.int32, device=support_y.device),
            #             torch.tensor(0, dtype=torch.int32, device=support_y.device))

        return gt_Y, count_1, orignal_acc

    def forward(self, support_x, support_y, query_x, query_y, gt_support_y=None, gt_query_y=None, train=False, logger=None, step=None, path=None, sampled_classes=None,
                bg_pcd_x=None, bg_pcd_y=None, support_c=None, support_flag=None, pcd_1024=None, label_1024=None, pcd_cutout=None, label_cutout=None, eval=False):
        """
        Args:
            support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points)
            support_y: support masks (foreground) with shape (n_way, k_shot, num_points)
            query_x: query point clouds with shape (n_queries, in_channels, num_points)
            query_y: query labels with shape (n_queries, num_points), each point \in {0,..., n_way}
            bg_pcd_x: (n, 9, 2048)
            bg_pcd_y: (n, 2048). binary mask to indicate object
            support_c: (nway, kshot, 2048). segment label
            support_flag: (nway, kshot) indicate each shot's absolute label
            pcd_1024: (n,9, 1024)
            label_1024: (n, 1024)
            pcd_cutout: (n, 9, 2048)
            label_cutout: (n, 2048)
        Return:
            query_pred: query point clouds predicted similarity, shape: (n_queries, n_way+1, num_points)
        """
        support_x = support_x.view(self.n_way*self.k_shot, self.in_channels, self.n_points) #
        support_feat = self.getFeatures(support_x)
        support_feat = support_feat.view(self.n_way, self.k_shot, self.feat_dim, self.n_points) # (n_way, k_shot, d, 2048)
        query_feat = self.getFeatures(query_x) #(n_queries, feat_dim, num_points)
        query_feat = query_feat.transpose(1,2).contiguous().view(-1, self.feat_dim) #(n_queries*num_points, feat_dim)

        pl_support_y = None
        if train == False and eval == True: # only use pl_support_y in the evaluation!
            # pl_support_y, _, cosine_map_list, cosine_sum_list, mask_list, gt_mask_list, seed_len_list = self.Mean_pl_support_y(support_feat, support_y, gt_support_y, support_x.view(self.n_way, self.k_shot, self.in_channels, self.n_points), n_x=2, n_y=1, n_z=1, shot_seed=self.shot_seed)
            pl_support_y, clean_flag = self.Mean_pl_support_y_multi_scale(support_feat, support_y, gt_support_y, support_x.view(self.n_way, self.k_shot, self.in_channels, self.n_points))

            # check pl_support_y accuracy. how many clean support in the pl_support_y=1.
            for way in range(self.n_way):
                way_pl_support_y = pl_support_y[way]  # (way_num_fg,) clean is 1
                mask = way_pl_support_y == 1
                point_acc = way_pl_support_y[mask] == gt_support_y[way][support_y[way] == 1].view(-1)[mask]
                point_acc = torch.sum(point_acc) / torch.sum(mask)
                # given fg mask acc
                orig_acc = gt_support_y[way][support_y[way] == 1] == 1
                orig_acc = torch.sum(orig_acc) / torch.sum(support_y[way] == 1)
                print('acc: {}, original acc: {}'.format(point_acc, orig_acc))

                # compute gt_flag
                gt_flag = torch.sum(gt_support_y[way], dim=-1) # (k, )
                gt_flag = torch.where(gt_flag > 0,
                                      torch.tensor(1, dtype=torch.float, device=support_y.device),
                                      torch.tensor(0, dtype=torch.float, device=support_y.device)) # (k, )
                # get shot-level clean ratio
                acc = torch.sum(clean_flag[way] * gt_flag) / torch.sum(clean_flag[way])
                print('shot level clean ratio: {}'.format(acc))
                self.shot_level_clean_ratio += acc

                # if orig_acc > 0.5:
                #     self.acc[1] += acc
                #     self.original_acc[1] += orig_acc
                #     self.clean_count[1] += 1
                # else:
                #     self.acc[0] += acc
                #     self.original_acc[0] += orig_acc
                #     self.clean_count[0] += 1
                # if acc == 0:
                #     self.acc_0 += 1 # how many shots become totally noise after cd..

        # add contrastive loss
        if train == True:

            # # bg samples with fps
            fps_k = 4
            # per way contrast: each class forms a cluster
            contrast_loss_2 = self.per_way_contrast_loss(support_feat, support_y, gt_support_y, support_flag, fps_k=fps_k, temp=0.1)
            pl_support_y = None

        fg_mask = support_y
        bg_mask = torch.logical_not(support_y)

        fg_prototypes, fg_labels, fg_assignments, fg_proto_number = self.getForegroundPrototypes(support_feat, fg_mask, k=self.n_subprototypes, pl_support_y=pl_support_y) # (n_way*k, feat_dim), (n_way*k, n_way+1). gt_mask only available in train
        bg_prototype, bg_labels, bg_assignments, bg_proto_number = self.getBackgroundPrototypes(support_feat, bg_mask, k=self.n_subprototypes) # (k, fea_dim), (k, n_way+1)

        # prototype learning
        if bg_prototype is not None and bg_labels is not None:
            prototypes = torch.cat((bg_prototype, fg_prototypes), dim=0) #(*, feat_dim)
            prototype_labels = torch.cat((bg_labels, fg_labels), dim=0) #(*,n_classes)
        else:
            prototypes = fg_prototypes
            prototype_labels = fg_labels
        self.num_prototypes = prototypes.shape[0]
        # print(prototypes.shape)


        # construct label matrix Y, with Y_ij = 1 if x_i is from the support set and labeled as y_i = j, otherwise Y_ij = 0.
        self.num_nodes = self.num_prototypes + query_feat.shape[0] # number of node of partial observed graph
        Y = torch.zeros(self.num_nodes, self.n_classes).cuda()
        Y[:self.num_prototypes] = prototype_labels

        # construct feat matrix F
        node_feat = torch.cat((prototypes, query_feat), dim=0)  # (num_nodes, feat_dim)


        A = self.calculateLocalConstrainedAffinity(node_feat, k=self.k_connect)
        Z = self.label_propagate(A, Y) #(num_nodes, n_way+1)

        # --------------------------- debug: check support label after label propagation ------------------
        if train == True:
            clean_ratio_LP_avg = 0.
            clean_ratio_original_avg = 0.
            begin_idx = 0
            proto_loss = []
            for i in range(self.n_way):
                # get point label from label propagation
                assert len(fg_assignments) == self.n_way
                # assert self.num_prototypes == (self.n_way+1) * self.n_subprototypes
                cls_assignment = fg_assignments[i]  # (|Ms==1|, )

                proto_pred_logits = Z[bg_proto_number:, :][begin_idx: begin_idx + fg_proto_number[i], :]  # (n_subprototypes, 3)
                begin_idx += fg_proto_number[i]

                proto_pred = torch.argmax(torch.softmax(proto_pred_logits, dim=1), dim=1)  # (self.n_subprototypes, )
                proto_pred = torch.where(proto_pred == i + 1,
                                         torch.tensor(1, dtype=support_y.dtype, device=support_y.device),
                                         torch.tensor(0, dtype=support_y.dtype, device=support_y.device))
                # diffuse prototype label to point label (only for point in Ms==1)
                point_pred = proto_pred[cls_assignment]
                # ------------- get gt label for Ms==1 point --------------#
                gt_label = gt_support_y[i].view(-1)  # (k-shot * 2048, )
                given_label = fg_mask[i].view(-1)  # support points that participate in the label propagation. (k-shot*2048, )
                # only get the Ms==1 point. should aligned with the point index in cls_assignments
                gt_label = gt_label[given_label == 1] # (num_fg, ). 1: true fg. 0: false fg.
                given_label = given_label[given_label == 1]
                assert len(gt_label) == len(cls_assignment)  # length is the number of |Ms==1|

                clean_ratio_LP = (point_pred == gt_label).sum().float() / len(gt_label)  # how many point label is correct after LP
                clean_ratio_original = (given_label == gt_label).sum().float() / len(gt_label)

                logger.cprint('after label propagation: class {}, clean_ratio_LP: {:.3f}, clean_ratio_original: {:.3f}'.format(i, clean_ratio_LP, clean_ratio_original))
                clean_ratio_LP_avg += clean_ratio_LP
                clean_ratio_original_avg += clean_ratio_original

            clean_ratio_LP_avg /= self.n_way
            clean_ratio_original_avg /= self.n_way

        # -------------------------------- end debug --------------------------------------------------------
        # clean_ratio_LP_avg = 0
        # clean_ratio_original_avg = 0


        query_pred = Z[self.num_prototypes:, :] #(n_queries*num_points, n_way+1)
        query_pred = query_pred.view(-1, query_y.shape[1], self.n_classes).transpose(1,2) #(n_queries, n_way+1, num_points)


        # ------------------- debug: check query prediction accuracy after label propagation -----------------------
        if train == True:
            query_pred_label = torch.softmax(query_pred, dim=1)
            query_pred_label = torch.argmax(query_pred_label, dim=1)  # (n-way, 2048)
            query_acc_LP = (query_pred_label == gt_query_y).sum().float() / (self.n_way * self.n_points)  # how many query prediction is truly correct.
            query_acc_original = (query_y == gt_query_y).sum().float() / (self.n_way * self.n_points)
            logger.cprint('after label propagation: QUERY prediction acc: {:.3f}, original_acc: {:.3f}'.format(query_acc_LP, query_acc_original))
        # -------------------- end debug ---------------------------------------------------------------------

        lp_loss = self.computeCrossEntropyLoss(query_pred, query_y)

        if train == True:
            contrast_loss = contrast_loss_2
            return query_pred, lp_loss, contrast_loss, query_acc_LP, query_acc_original, clean_ratio_LP_avg, clean_ratio_original_avg
        else:
            return query_pred, lp_loss

    def getFeatures(self, x):
        """
        Forward the input data to network and generate features
        :param x: input data with shape (B, C_in, L)
        :return: features with shape (B, C_out, L)
        """
        if self.use_attention:
            feat_level1, feat_level2 = self.encoder(x)
            feat_level3 = self.base_learner(feat_level2)
            att_feat = self.att_learner(feat_level2)
            return torch.cat((feat_level1, att_feat, feat_level3), dim=1)
        else:
            # return self.base_learner(self.encoder(x))
            feat_level1, feat_level2 = self.encoder(x)
            feat_level3 = self.base_learner(feat_level2)
            map_feat = self.linear_mapper(feat_level2)
            return torch.cat((feat_level1, map_feat, feat_level3), dim=1)

    def getMutiplePrototypes(self, feat, k):
        """
        Extract multiple prototypes by points separation and assembly

        Args:
            feat: input point features, shape:(n_points, feat_dim)
        Return:
            prototypes: output prototypes, shape: (n_prototypes, feat_dim)
            assignments: (num_points) assignment to the prototype
            num_proto:
        """
        # sample k seeds as initial centers with Farthest Point Sampling (FPS)
        n = feat.shape[0]
        assert n > 0
        ratio = k / n
        if ratio < 1:
            fps_index = fps(feat, None, ratio=ratio, random_start=False).unique()
            num_prototypes = len(fps_index)
            farthest_seeds = feat[fps_index]

            # compute the point-to-seed distance
            distances = F.pairwise_distance(feat[..., None], farthest_seeds.transpose(0, 1)[None, ...],
                                            p=2)  # (n_points, n_prototypes)

            # hard assignment for each point
            assignments = torch.argmin(distances, dim=1)  # (n_points,)

            # aggregating each cluster to form prototype
            prototypes = torch.zeros((num_prototypes, feat.shape[1])).cuda()
            for i in range(num_prototypes):
                selected = torch.nonzero(assignments == i).squeeze(1)
                selected = feat[selected, :]
                prototypes[i] = selected.mean(0)
            return prototypes, assignments, num_prototypes, farthest_seeds
        else:
            assignments = torch.arange(n, device=feat.device)
            num_prototypes = n
            return feat, assignments, num_prototypes, feat # last one is the seed feature

    def getForegroundPrototypes(self, feats, masks, k=100, gt_masks=None, pl_support_y=None):
        """
        Extract foreground prototypes for each class via clustering point features within that class

        Args:
            feats: input support features, shape: (n_way, k_shot, feat_dim, num_points)
            masks: foreground binary masks, shape: (n_way, k_shot, num_points)
            gt_masks: gt_support_y. only use the clean fg points to generate prototype.
        Return:
            prototypes: foreground prototypes, shape: (n_way*k, feat_dim)
            labels: foreground prototype labels (one-hot), shape: (n_way*k, n_way+1)
            assignment: [cls1_fg_points_assignment, cls2_fg_points_assignment, ...]
            prototype_number: [num_proto_in_cls1, num_proto_in_cls2, ...]
        """
        prototypes = []
        labels = []
        assignment = []
        prototype_number = []
        for i in range(self.n_way):
            # extract point features belonging to current foreground class
            feat = feats[i, ...].transpose(1,2).contiguous().view(-1, self.feat_dim) #(k_shot*num_points, feat_dim)

            if gt_masks != None: # only use clean support fg points. for Ys_gt
                index = torch.nonzero((masks[i] * gt_masks[i]).view(-1)).squeeze(1) #(num_fg_points,)
                print('after gt_mask: {}'.format(index.shape[0]))
            else:
                index = torch.nonzero(masks[i, ...].view(-1)).squeeze(1) #(num_fg_points,)

            # index = torch.nonzero(masks[i, ...].view(-1)).squeeze(1) #(num_fg_points,)

            feat = feat[index] # get fg features (num_fg_points, d)
            print(feat.shape[0])

            if pl_support_y != None: # only use clean fg
                assert feat.shape[0] == pl_support_y[i].shape[0]
                feat = feat[pl_support_y[i]==1]
                print('after clustering: {}'.format(feat.shape[0]))

            class_prototypes, cls_assignment, num_prototypes, _ = self.getMutiplePrototypes(feat, k) # (k, d)

            assignment.append(cls_assignment)
            prototypes.append(class_prototypes)
            prototype_number.append(num_prototypes)

            # construct label matrix
            class_labels = torch.zeros(class_prototypes.shape[0], self.n_classes).cuda()
            class_labels[:, i+1] = 1
            labels.append(class_labels)

        prototypes = torch.cat(prototypes, dim=0) # ()
        labels = torch.cat(labels, dim=0)

        return prototypes, labels, assignment, prototype_number

    def getBackgroundPrototypes(self, feats, masks, k=100):
        """
        Extract background prototypes via clustering point features within background class

        Args:
            feats: input support features, shape: (n_way, k_shot, feat_dim, num_points)
            masks: background binary masks, shape: (n_way, k_shot, num_points)
        Return:
            prototypes: background prototypes, shape: (k, feat_dim)
            labels: background prototype labels (one-hot), shape: (k, n_way+1)
            num_proto:
            assignment: point2proto assignment. (num_points, )
        """
        feats = feats.transpose(2,3).contiguous().view(-1, self.feat_dim)
        index = torch.nonzero(masks.view(-1)).squeeze(1)
        feat = feats[index]
        # in case this support set does not contain background points..
        if feat.shape[0] != 0:
            prototypes, assignment, num_prototypes, _ = self.getMutiplePrototypes(feat, k)

            labels = torch.zeros(prototypes.shape[0], self.n_classes).cuda()
            labels[:, 0] = 1

            return prototypes, labels, assignment, num_prototypes
        else:
            return None, None

    def calculateLocalConstrainedAffinity(self, node_feat, k=200, method='gaussian'):
        """
        Calculate the Affinity matrix of the nearest neighbor graph constructed by prototypes and query points,
        It is a efficient way when the number of nodes in the graph is too large.

        Args:
            node_feat: input node features
                  shape: (num_nodes, feat_dim)
            k: the number of nearest neighbors for each node to compute the similarity
            method: 'cosine' or 'gaussian', different similarity function
        Return:
            A: Affinity matrix with zero diagonal, shape: (num_nodes, num_nodes)
        """
        # kNN search for the graph
        X = node_feat.detach().cpu().numpy()
        # build the index with cpu version
        index = faiss.IndexFlatL2(self.feat_dim)
        index.add(X)
        _, I = index.search(X, k + 1)
        I = torch.from_numpy(I[:, 1:]).cuda() #(num_nodes, k)

        # create the affinity matrix
        knn_idx = I.unsqueeze(2).expand(-1, -1, self.feat_dim).contiguous().view(-1, self.feat_dim)
        knn_feat = torch.gather(node_feat, dim=0, index=knn_idx).contiguous().view(self.num_nodes, k, self.feat_dim)

        if method == 'cosine':
            knn_similarity = F.cosine_similarity(node_feat[:,None,:], knn_feat, dim=2)
        elif method == 'gaussian':
            dist = F.pairwise_distance(node_feat[:,:,None], knn_feat.transpose(1,2), p=2)
            knn_similarity = torch.exp(-0.5*(dist/self.sigma)**2)
        else:
            raise NotImplementedError('Error! Distance computation method (%s) is unknown!' %method)

        A = torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.float).cuda()
        A = A.scatter_(1, I, knn_similarity)
        A = A + A.transpose(0,1)

        identity_matrix = torch.eye(self.num_nodes, requires_grad=False).cuda()
        A = A * (1 - identity_matrix)
        return A

    def label_propagate(self, A, Y, alpha=0.99):
        """ Label Propagation, refer to "Learning with Local and Global Consistency" NeurIPs 2003
        Args:
            A: Affinity matrix with zero diagonal, shape: (num_nodes, num_nodes)
            Y: initial label matrix, shape: (num_nodes, n_way+1)
            alpha: a parameter to control the amount of propagated info.
        Return:
            Z: label predictions, shape: (num_nodes, n_way+1)
        """
        #compute symmetrically normalized matrix S
        eps = np.finfo(float).eps
        D = A.sum(1) #(num_nodes,)
        D_sqrt_inv = torch.sqrt(1.0/(D+eps))
        D_sqrt_inv = torch.diag_embed(D_sqrt_inv).cuda()
        S = D_sqrt_inv @ A @ D_sqrt_inv

        #close form solution
        Z = torch.inverse(torch.eye(self.num_nodes).cuda() - alpha*S + eps) @ Y
        return Z

    def computeCrossEntropyLoss(self, query_logits, query_labels):
        """ Calculate the CrossEntropy Loss for query set
        """
        return F.cross_entropy(query_logits, query_labels)
