""" Prototypical Network 

Author: Zhao Na, 2020
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dgcnn import DGCNN
from models.attention import SelfAttention
from torch_scatter import scatter_mean, scatter_add
from torch_cluster import fps

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


class ProtoNet(nn.Module):
    def __init__(self, args):
        super(ProtoNet, self).__init__()
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.dist_method = args.dist_method
        self.in_channels = args.pc_in_dim
        self.n_points = args.pc_npts
        self.use_attention = args.use_attention

        self.encoder = DGCNN(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k)
        self.base_learner = BaseLearner(args.dgcnn_mlp_widths[-1], args.base_widths)

        if self.use_attention:
            self.att_learner = SelfAttention(args.dgcnn_mlp_widths[-1], args.output_dim)
        else:
            self.linear_mapper = nn.Conv1d(args.dgcnn_mlp_widths[-1], args.output_dim, 1, bias=False)

        self.begin_use_global_proto = 1000000000
    def aggregate_cluster_one_class(self, feat, label, cluster_label):
        '''
        :param feat: (K, d, 2048)
        :param label: (K, 2048). binary label
        :param cluster_label: (N, 2048)
        :return: only return clusters belong to this class using given label!! [clsuters in pcd1, clusters in pcd2, ...]
        total_cls_cluster_list: [(d,m1), (d,m2), ...] only store class clusters. if pcd doesn't contain this class, skip
        total_cls_cluster_idx_list: [(cluster idx, len=m1, int64), [], ..] indicate cluster idx in the shunxu clusters in 2048 pcd.
        total_cluster_size: [[n1,n2,..], [n3, n4..], ...] record how many points in each cluster. each is a 1d tensor
        Note: scatter_mean: return distribution in [0, 1, 2, ..., max_id]!!
        '''
        total_cls_cluster_list = []
        total_cls_cluster_idx_list = []
        total_cluster_size = []
        for i in range(feat.shape[0]):
            tmp_feat = feat[i] # (d, 2048)
            tmp_cluster_label = cluster_label[i].long() # (2048)
            tmp_label = label[i].clone().float() # int32/bool -> float32 (2048,)

            # get cluster
            tmp_cluster_name = torch.unique(tmp_cluster_label, sorted=True) # (2,5,7,..)
            # record number of points in teach cluster
            aa = torch.ones_like(tmp_label) # (2048,)
            tmp_cluster_size = scatter_add(aa, tmp_cluster_label)
            tmp_cluster_size = tmp_cluster_size[tmp_cluster_name]

            # aggregate cluster feature
            tmp_cluster_feat = scatter_mean(tmp_feat, tmp_cluster_label, dim=1) # (d, max_cluster_id+1)
            tmp_cluster_feat = tmp_cluster_feat[:, tmp_cluster_name] # (d, number_cluster_in_2048pcd). only get clusters that truly exist in this 2048 pcd.

            # aggregate label within each cluster
            tmp_cluster_label = scatter_mean(tmp_label, tmp_cluster_label)
            tmp_cluster_label = tmp_cluster_label[tmp_cluster_name] # (number_cluster_in_2048pcd, ).
            # if label > 0.5, then this clsuter belongs to this class.
            tmp_cluster_label = torch.where(tmp_cluster_label > 0.5,
                                        torch.tensor(1, dtype=torch.int32, device=cluster_label.device),
                                        torch.tensor(0, dtype=torch.int32, device=cluster_label.device))
            # get cluster for this class: only have n clusters for this class
            tmp_cls_cluster_idx = torch.nonzero(tmp_cluster_label).squeeze(-1) # (n,), int64

            tmp_cls_cluster_feat = tmp_cluster_feat[:, tmp_cls_cluster_idx] # (d,n)
            total_cls_cluster_list.append(tmp_cls_cluster_feat) # (d, number_valid_cluster)
            total_cls_cluster_idx_list.append(tmp_cls_cluster_idx)
            # cluster size
            tmp_cluster_size = tmp_cluster_size[tmp_cls_cluster_idx]
            total_cluster_size.append(tmp_cluster_size)

        return total_cls_cluster_list, total_cls_cluster_idx_list, total_cluster_size

    def detect_clean_cluster_one_class(self, cls_cluster, cluster_size, support_len=None, step=-1, cls=None, train=False):
        '''
        :param cls_cluster: feature for clusters in one class. (d, num_cluster)
        :param cluster_size: size for each cluster. cluster order is the same as cls_cluster
        :param support_len: how many support clusters in this mtx
        :param step: current iteration
        :param cls: cls idx in the train classes
        :return: find clean mask. shape [num_cluster,], value=[0,1,0,0,...].
        make sure some support cluster is clean!!
        only use global prototype during meta-training!
        '''
        num_cluster = cls_cluster.shape[1]
        # l2 norm first
        cls_cluster_l2 = F.normalize(cls_cluster, p=2, dim=0)

        if train == True and step > self.begin_use_global_proto:
            # append global prototype at the end for the similarity check
            global_prototype_l2 = F.normalize(self.global_prototypes[cls].unsqueeze(1), p=2, dim=0)
            cls_cluster_l2 = torch.cat([cls_cluster_l2, global_prototype_l2], dim=1) # global prototype also need l2 norm
        # cosin similarity
        sim_mtx = torch.mm(cls_cluster_l2.transpose(0,1), cls_cluster_l2) # (num_cluster, num_cluster)
        # add weight for similarity
        weight = cluster_size.float() / cluster_size.max() # normalize to (0,1)
        if train == True and step > self.begin_use_global_proto:
            weight = torch.cat([weight, torch.ones(1, dtype=weight.dtype, device=weight.device)]) # set global prototype weight =1
            weight = weight.unsqueeze(1).transpose(0, 1).repeat(num_cluster+1, 1)
        else:
            weight = weight.unsqueeze(1).transpose(0,1).repeat(num_cluster, 1) # copy for each row. (num_cluster, num_cluster)
        assert (weight[0] == weight[1]).all()

        sim_mtx = sim_mtx * weight

        sim_score = torch.sum(sim_mtx, dim=1) # (num_cluster,)
        clean_mask = sim_score[:num_cluster] > torch.mean(sim_score) # exclude global prototype here
        print('sim mtx statistic: max={}, min={}, mean={}, above_mean={}, total_cluster={}'.format(torch.max(sim_score), torch.min(sim_score), torch.mean(sim_score),
                                                                                                   torch.sum(clean_mask), sim_score.shape[0]))
        # clean_mask = sim_score[:num_cluster] > torch.median(sim_score)  # exclude global prototype here
        # print('sim mtx statistic: max={}, min={}, median={}, above_median={}, total_cluster={}'.format(torch.max(sim_score),
        #                                                                                            torch.min(sim_score),
        #                                                                                            torch.median(sim_score),
        #                                                                                            torch.sum(clean_mask),
        #                                                                                            sim_score.shape[0]))

        # # print(torch.max(sim_score), torch.mean(sim_score))
        # if train == True:
        #     # update global prototype
        #     index = torch.argmax(sim_score[:support_len])
        #     current_feat = cls_cluster[:,index] # (d,) no l2 norm!
        #     self.update_global_proto(cls, current_feat, step)

        return clean_mask

    def check_clean_detection(self, gt_label, noisy_label):
        '''
        :param gt_label:  [k-shot. cluster-level binary label, k-shot cluster-level label, ...]
        :param noisy_label: [k-shot. cluster-level binary label, k-shot cluster-level label, ...]
        :return: precision and recall
        '''
        assert len(gt_label) == len(noisy_label)
        TP = 0.
        FN = 0.
        FP = 0.
        for i in range(len(gt_label)):
            TP += torch.sum(gt_label[i] * noisy_label[i]).float()
            FN += torch.sum(gt_label[i] * (1 - noisy_label[i])).float()
            FP += torch.sum((1 - gt_label[i]) * noisy_label[i]).float()

        # assert gt_label.shape == noisy_label.shape
        # # recall
        # TP = torch.sum(gt_label * noisy_label).float()
        # FN = torch.sum(gt_label * (1 - noisy_label)).float()
        # FP = torch.sum((1 - gt_label) * noisy_label).float()
        #
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        return precision, recall

    def check_query_label(self, gt_label, noisy_label, logger):
        '''
        :param gt_label: (n_way, 2048)
        :param noisy_label: (n_way, 2048)
        :return: check precision, recall for each class (include bg)
        '''
        n_class = self.n_way + 1
        avg_precision = 0.
        avg_recall = 0.
        for i in range(n_class):
            gt_label_cls = gt_label == i # (binary mask)
            noisy_label_cls = noisy_label == i

            TP = torch.sum(gt_label_cls * noisy_label_cls).float()
            FN = torch.sum(gt_label_cls * (~noisy_label_cls)).float()
            FP = torch.sum((~gt_label_cls) * noisy_label_cls).float()

            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            if i == 0:
                logger.cprint('query label: bg class, precision: {:.3f}, recall: {:.3f}'.format(precision, recall))
            else:
                logger.cprint('query label, class {}, precision: {:.3f}, recall: {:.3f}'.format(i-1, precision, recall))

            avg_precision += precision
            avg_recall += recall
        avg_precision /= n_class
        avg_recall /= n_class
        return avg_precision, avg_recall

    def check_support_mask_accuracy(self, estimate_mask, original_mask, gt_mask, logger):
        ''' point-level accuracy.
        :param esitmate_mask: (n-way, k-shot, 2048). input_mask * clean_detection_mask
        :param original_mask: input support mask. without any clean_detection.
        :param gt_mask: (n-way, k-shot, 2048). gt_support_mask.
        :return:
        '''
        clean_ratio = 0. # in the estimate clean mask
        size_ratio = 0. # how many 1_points in the estimate_mask vs how many 1_points in the original_mask. <1
        original_clean_ratio = 0. # (clean ratio in the original mask)
        for i in range(self.n_way):
            cls_estimate_mask = estimate_mask[i] # input mask * clean_detetcion_mask
            cls_gt_mask = gt_mask[i]
            cls_original_mask = original_mask[i] # input mask. without clean detection.
            # how many 1 points left:
            estimate_1 = torch.sum(cls_estimate_mask)
            original_1 = torch.sum(cls_original_mask)
            # how many clean in the 1 points
            original_clean = torch.sum(cls_original_mask * cls_gt_mask)
            estimate_clean = torch.sum(cls_estimate_mask * cls_gt_mask)


            logger.cprint('cls {}: original has {} 1_ponts and {} clean, ratio={:.3f}. After clean_detection has {} 1_points and {} clean, ratio={:.3f}'.format(i, original_1, original_clean, original_clean.float()/original_1,
                                                                                                                                               estimate_1, estimate_clean, estimate_clean.float()/estimate_1))
            clean_ratio += estimate_clean.float() / estimate_1
            size_ratio += estimate_1.float() / original_1
            # original_clean_ratio += original_clean.float()/original_1

        return clean_ratio/self.n_way, size_ratio/self.n_way


    def forward(self, support_x, support_y, query_x, query_y, support_c=None, query_c=None, train=False, gt_support_y=None, gt_query_y=None, logger=None):
        """
        Args:
            support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points)
            support_y: support masks (foreground) with shape (n_way, k_shot, num_points)
            query_x: query point clouds with shape (n_queries, in_channels, num_points)
            query_y: query labels with shape (n_queries, num_points), each point \in {0,..., n_way}
        Return:
            query_pred: query point clouds predicted similarity, shape: (n_queries, n_way+1, num_points)
        """
        support_x = support_x.view(self.n_way * self.k_shot, self.in_channels, self.n_points)
        support_feat = self.getFeatures(support_x)
        support_feat = support_feat.view(self.n_way, self.k_shot, -1, self.n_points)
        query_feat = self.getFeatures(query_x)  # (n_queries, feat_dim, num_points)

        fg_mask = support_y
        bg_mask = torch.logical_not(support_y)

        support_fg_feat = self.getMaskedFeatures(support_feat, fg_mask)
        suppoer_bg_feat = self.getMaskedFeatures(support_feat, bg_mask)

        # prototype learning
        fg_prototypes, bg_prototype = self.getPrototype(support_fg_feat, suppoer_bg_feat)
        prototypes = [bg_prototype] + fg_prototypes

        # non-parametric metric learning
        similarity = [self.calculateSimilarity(query_feat, prototype, self.dist_method) for prototype in prototypes]

        query_pred = torch.stack(similarity, dim=1)  # (n_queries, n_way+1, num_points)
        loss = self.computeCrossEntropyLoss(query_pred, query_y)
        return query_pred, loss

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

    def getMaskedFeatures(self, feat, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            feat: input features, shape: (n_way, k_shot, feat_dim, num_points)
            mask: binary mask, shape: (n_way, k_shot, num_points)
        Return:
            masked_feat: masked features, shape: (n_way, k_shot, feat_dim)
        """
        mask = mask.unsqueeze(2) # (n-way, k-shot, 1, 2048)
        masked_feat = torch.sum(feat * mask, dim=3) / (mask.sum(dim=3) + 1e-5)
        return masked_feat

    def getPrototype(self, fg_feat, bg_feat):
        """
        Average the features to obtain the prototype

        Args:
            fg_feat: foreground features for each way/shot, shape: (n_way, k_shot, feat_dim)
            bg_feat: background features for each way/shot, shape: (n_way, k_shot, feat_dim)
        Returns:
            fg_prototypes: a list of n_way foreground prototypes, each prototype is a vector with shape (feat_dim,)
            bg_prototype: background prototype, a vector with shape (feat_dim,)
        """
        # print('manually remove noise')
        fg_prototypes = [fg_feat[way, ...].sum(dim=0) / self.k_shot for way in range(self.n_way)]
        # fg_prototypes = [fg_feat[way, :3, :].sum(dim=0) / 3. for way in range(self.n_way)]
        bg_prototype =  bg_feat.sum(dim=(0,1)) / (self.n_way * self.k_shot)
        return fg_prototypes, bg_prototype

    def calculateSimilarity(self, feat,  prototype, method='cosine', scaler=10):
        """
        Calculate the Similarity between query point-level features and prototypes

        Args:
            feat: input query point-level features
                  shape: (n_queries, feat_dim, num_points)
            prototype: prototype of one semantic class
                       shape: (feat_dim,)
            method: 'cosine' or 'euclidean', different ways to calculate similarity
            scaler: used when 'cosine' distance is computed.
                    By multiplying the factor with cosine distance can achieve comparable performance
                    as using squared Euclidean distance (refer to PANet [ICCV2019])
        Return:
            similarity: similarity between query point to prototype
                        shape: (n_queries, num_points)
        """
        if method == 'cosine':
            similarity = F.cosine_similarity(feat, prototype[None, ..., None], dim=1) * scaler
        elif method == 'euclidean':
            similarity = - F.pairwise_distance(feat, prototype[None, ..., None], p=2)**2
        else:
            raise NotImplementedError('Error! Distance computation method (%s) is unknown!' %method)
        return similarity

    def computeCrossEntropyLoss(self, query_logits, query_labels):
        """ Calculate the CrossEntropy Loss for query set
        """
        return F.cross_entropy(query_logits, query_labels)


class ProtoNet_Contrast(nn.Module):
    def __init__(self, args):
        super(ProtoNet_Contrast, self).__init__()
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.dist_method = args.dist_method
        self.in_channels = args.pc_in_dim
        self.n_points = args.pc_npts
        self.use_attention = args.use_attention

        self.encoder = DGCNN(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k)
        self.base_learner = BaseLearner(args.dgcnn_mlp_widths[-1], args.base_widths)

        if self.use_attention:
            self.att_learner = SelfAttention(args.dgcnn_mlp_widths[-1], args.output_dim)
            print('use attention!')
        else:
            self.linear_mapper = nn.Conv1d(args.dgcnn_mlp_widths[-1], args.output_dim, 1, bias=False)

        # self.classes = [3, 11, 10, 0, 8, 4]
        self.acc = {1: 0., 0: 0.}
        self.original_acc = {1: 0., 0: 0.}  # 1 is orignal_clean ratio>50%
        self.clean_count = {1: 0, 0: 0}
        self.acc_0 = 0
        self.shot_level_clean_ratio = 0
        self.feat_dim = 192
        self.proj = nn.Linear(self.feat_dim, 128)

        print('using protonet+CCNS+MDNS')
    def per_way_contrast_loss(self, support_feat, support_y, gt_support_y,
                          support_flag,
                          query_feat=None, query_y=None, fps_k=1, temp=0.1):
        ''' WayContrast: per way supervised contrative loss
        :param support_feat: (nway, kshot, d, 2048)
        :param gt_support_y: (nway, kshot, 2048)
        :param support_flag: (nway, kshot). abosulte label for each shots
        :param temp: scalar, temperature
        :param query_feat: (n_way*2048, feat_dim)
        :param query_y: (nway, 2048)
        :return: label: 1: cls1, 2: cls2, -1: bg/noise
        '''
        # ele = support_flag[0,0]
        # if ele*self.k_shot == torch.sum(support_flag[0]):
        #     # clean support set.
        #     total_loss = -100
        # else:
        #     total_loss = []

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

            # # for fps=1:
            # # compute mean of log-likelihood over positive
            # new_mask = torch.sum(gt_mask, dim=1) > 0
            # mean_log_prob_pos = (gt_mask * log_prob).sum(1)[new_mask]  # (n*k,)
            # mean_log_prob_pos = mean_log_prob_pos / gt_mask[new_mask].sum(1)
            # mean_log_prob_pos = mean_log_prob_pos[new_mask] # remove noise in contrative loss

            # loss
            loss = - mean_log_prob_pos
            loss = loss.mean()
            total_loss.append(loss)

        total_loss = sum(total_loss) / len(total_loss)

        return total_loss

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

        # # shot-level as main view.
        # x_list = [1,2]
        # y_list = [1,2]
        # z_list = [1,1]
        # num_scale = len(x_list)
        # total_flag = []
        # for i in range(num_scale):
        #     _, flag, _,_,_,_,_ = self.Mean_pl_support_y(support_feat, support_y, gt_support_y, support_x, n_x=x_list[i], n_y=y_list[i], n_z=z_list[i])
        #
        #     total_flag.append(flag) # flag: (n,k)
        #
        # print(total_flag)
        # total_flag = torch.stack(total_flag, dim=0) # (3, nway, kshot)
        # anchor_view = total_flag[0] # (nway, kshot)
        # side_view = total_flag[1:] # (2, nway,kshot)
        # side_view = torch.mean(side_view, dim=0) # (nway, kshot)
        #
        # pl_support_y = []
        # for way in range(self.n_way):
        #     way_pl_support_y = []
        #     for k in range(self.k_shot):
        #         shot_pl_support_y = support_y[way,k][support_y[way,k]>0] # (fg, )
        #         if anchor_view[way,k] == 0 and side_view[way,k] < 0.6:
        #             # if both views are noise, set this view to noise.
        #             shot_pl_support_y = torch.zeros_like(shot_pl_support_y) # set to 0
        #         way_pl_support_y.append(shot_pl_support_y)
        #     way_pl_support_y = torch.cat(way_pl_support_y, dim=0) # (num_fg, )
        #     pl_support_y.append(way_pl_support_y)

        return pl_support_y, clean_flag

    def Mean_pl_support_y(self, support_feat, support_y, gt_support_y, support_x, n_x=1, n_y=1, n_z=1, shot_seed=1):
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
                # print(fg_mask.shape)
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
            # # add std check:
            # cosine_sum = torch.sum(cosine_map, dim=1)
            # if torch.std(cosine_sum) < 0.1:
            #     print('skip cd!')
            #     pl_support_y = []
            #     for way in range(self.n_way):
            #         way_pl_support_y = []
            #         for k in range(self.k_shot):
            #             shot_pl_support_y = support_y[way, k][support_y[way, k] > 0]  # (fg, )
            #             way_pl_support_y.append(shot_pl_support_y)
            #         way_pl_support_y = torch.cat(way_pl_support_y, dim=0)  # (num_fg, )
            #         pl_support_y.append(way_pl_support_y)
            #     return pl_support_y, flag, cosine_map_list, cosine_sum_list, mask_list, gt_mask_list, seed_len_list


            # add temperature
            if n_x == 1 and n_y==1 and n_z == 1:
                cosine_map = cosine_map.pow(3)
                # mask = (cosine_map < 0.9) * (cosine_map > -0.9)
                # cosine_map[mask] = (1/0.9**4)*cosine_map[mask].pow(5)
                # cosine_map[~mask] = cosine_map[~mask].pow(0.3)

            # #
            # cosine_map = cosine_map + 1
            # # a = 0.5
            # # cosine_map = 5*(a**4)*(torch.sqrt(1+(cosine_map / a).pow(4)) - 1)
            # cosine_map = cosine_map.pow(3)



            cosine_sum = torch.sum(cosine_map, dim=1)  # (num_seed)
            mask = cosine_sum > torch.mean(cosine_sum)  # (num_seed,)


            # add gt
            gt_mask = torch.sum(gt_support_y[way], dim=-1) > 0
            # print(cosine_map)
            # print(cosine_sum)
            print('mask: {}'.format(mask), 'gt_mask: {}'.format(gt_mask))
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

    def forward(self, support_x, support_y, query_x, query_y, gt_support_y=None, gt_query_y=None, train=False, logger=None, step=None, path=None, sampled_classes=None,
                bg_pcd_x=None, bg_pcd_y=None, support_c=None, support_flag=None, pcd_1024=None, label_1024=None, pcd_cutout=None, label_cutout=None):
        """
        Args:
            support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points)
            support_y: support masks (foreground) with shape (n_way, k_shot, num_points)
            query_x: query point clouds with shape (n_queries, in_channels, num_points)
            query_y: query labels with shape (n_queries, num_points), each point \in {0,..., n_way}
        Return:
            query_pred: query point clouds predicted similarity, shape: (n_queries, n_way+1, num_points)
        """
        support_x = support_x.view(self.n_way*self.k_shot, self.in_channels, self.n_points)
        support_feat = self.getFeatures(support_x)
        support_feat = support_feat.view(self.n_way, self.k_shot, -1, self.n_points)
        query_feat = self.getFeatures(query_x) #(n_queries, feat_dim, num_points)

        # add contrastive loss
        if train == True:
            clean_flag=None
            # # bg samples with fps
            fps_k = 4
            # per way contrast: each class forms a cluster
            contrast_loss_2 = self.per_way_contrast_loss(support_feat, support_y, gt_support_y, support_flag,
                                                         fps_k=fps_k, temp=0.1)

        if train == False:
            # pl_support_y, _, cosine_map_list, cosine_sum_list, mask_list, gt_mask_list, seed_len_list = self.Mean_pl_support_y(support_feat, support_y, gt_support_y, support_x.view(self.n_way, self.k_shot, self.in_channels, self.n_points), n_x=2, n_y=1, n_z=1, shot_seed=self.shot_seed)
            pl_support_y, clean_flag = self.Mean_pl_support_y_multi_scale(support_feat, support_y, gt_support_y, support_x.view(self.n_way, self.k_shot, self.in_channels, self.n_points))

            # pl_support_y = self.spectra_clustering_pl_support_y(support_feat, support_y, gt_support_y)
            # check pl_support_y accuracy. how many clean support in the pl_support_y=1.
            for way in range(self.n_way):
                way_pl_support_y = pl_support_y[way]  # (way_num_fg,) clean is 1
                mask = way_pl_support_y == 1
                acc = way_pl_support_y[mask] == gt_support_y[way][support_y[way] == 1].view(-1)[mask]
                acc = torch.sum(acc) / torch.sum(mask)
                # given fg mask acc
                orig_acc = gt_support_y[way][support_y[way] == 1] == 1
                orig_acc = torch.sum(orig_acc) / torch.sum(support_y[way] == 1)
                print('acc: {}, original acc: {}'.format(acc, orig_acc))

                # compute gt_flag
                gt_flag = torch.sum(gt_support_y[way], dim=-1) # (k, )
                gt_flag = torch.where(gt_flag > 0,
                                      torch.tensor(1, dtype=torch.float, device=support_y.device),
                                      torch.tensor(0, dtype=torch.float, device=support_y.device)) # (k, )
                # get shot-level clean ratio
                acc = torch.sum(clean_flag[way] * gt_flag) / torch.sum(clean_flag[way])
                print('shot level clean ratio: {}'.format(acc))
                self.shot_level_clean_ratio += acc




        fg_mask = support_y
        bg_mask = torch.logical_not(support_y)

        support_fg_feat = self.getMaskedFeatures(support_feat, fg_mask)
        suppoer_bg_feat = self.getMaskedFeatures(support_feat, bg_mask)

        # prototype learning
        fg_prototypes, bg_prototype = self.getPrototype(support_fg_feat, suppoer_bg_feat, clean_flag=clean_flag) # set clean_flag = clean_flag to enable MDNS
        prototypes = [bg_prototype] + fg_prototypes

        # non-parametric metric learning
        similarity = [self.calculateSimilarity(query_feat, prototype, self.dist_method) for prototype in prototypes]

        query_pred = torch.stack(similarity, dim=1) #(n_queries, n_way+1, num_points)
        loss = self.computeCrossEntropyLoss(query_pred, query_y)

        if train == True:
            contrast_loss = contrast_loss_2
            query_acc_LP = 0.5
            query_acc_original = 0.5
            clean_ratio_LP_avg = 0.5
            clean_ratio_original_avg = 0.5
            return query_pred, loss, contrast_loss, query_acc_LP, query_acc_original, clean_ratio_LP_avg, clean_ratio_original_avg
        else:
            return query_pred, loss

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

    def getMaskedFeatures(self, feat, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            feat: input features, shape: (n_way, k_shot, feat_dim, num_points)
            mask: binary mask, shape: (n_way, k_shot, num_points)
        Return:
            masked_feat: masked features, shape: (n_way, k_shot, feat_dim)
        """
        mask = mask.unsqueeze(2)
        masked_feat = torch.sum(feat * mask, dim=3) / (mask.sum(dim=3) + 1e-5)
        return masked_feat

    def getPrototype(self, fg_feat, bg_feat, clean_flag=None):
        """
        Average the features to obtain the prototype

        Args:
            fg_feat: foreground features for each way/shot, shape: (n_way, k_shot, feat_dim)
            bg_feat: background features for each way/shot, shape: (n_way, k_shot, feat_dim)
            clean_flag: (nway, kshot) clean is 1, noise is 0.
        Returns:
            fg_prototypes: a list of n_way foreground prototypes, each prototype is a vector with shape (feat_dim,)
            bg_prototype: background prototype, a vector with shape (feat_dim,)
        """
        if clean_flag != None:
            fg_prototypes = []
            for way in range(self.n_way):
                mask = clean_flag[way].unsqueeze(-1).repeat(1, self.feat_dim) # (k, d)
                num_clean = torch.sum(clean_flag[way])
                print(num_clean)
                tmp_proto = torch.sum(fg_feat[way] * mask, dim=0) / num_clean
                fg_prototypes.append(tmp_proto)
        else:
            fg_prototypes = [fg_feat[way, ...].sum(dim=0) / self.k_shot for way in range(self.n_way)]
        bg_prototype =  bg_feat.sum(dim=(0,1)) / (self.n_way * self.k_shot)
        return fg_prototypes, bg_prototype

    def calculateSimilarity(self, feat,  prototype, method='cosine', scaler=10):
        """
        Calculate the Similarity between query point-level features and prototypes

        Args:
            feat: input query point-level features
                  shape: (n_queries, feat_dim, num_points)
            prototype: prototype of one semantic class
                       shape: (feat_dim,)
            method: 'cosine' or 'euclidean', different ways to calculate similarity
            scaler: used when 'cosine' distance is computed.
                    By multiplying the factor with cosine distance can achieve comparable performance
                    as using squared Euclidean distance (refer to PANet [ICCV2019])
        Return:
            similarity: similarity between query point to prototype
                        shape: (n_queries, 1, num_points)
        """
        if method == 'cosine':
            similarity = F.cosine_similarity(feat, prototype[None, ..., None], dim=1) * scaler
        elif method == 'euclidean':
            similarity = - F.pairwise_distance(feat, prototype[None, ..., None], p=2)**2
        else:
            raise NotImplementedError('Error! Distance computation method (%s) is unknown!' %method)
        return similarity

    def computeCrossEntropyLoss(self, query_logits, query_labels):
        """ Calculate the CrossEntropy Loss for query set
        """
        return F.cross_entropy(query_logits, query_labels)


def gen_prototypes(embeddings, ways, shots, agg_method="mean"):
    assert (
        embeddings.size(0) == ways * shots
    ), "# of embeddings ({}) doesn't match ways ({}) and shots ({})".format(
        embeddings.size(0), ways, shots
    )

    embeddings = embeddings.reshape(ways, shots, -1)
    mean_embeddings = embeddings.mean(dim=1) # (nway, d)

    if agg_method == "mean":
        return mean_embeddings

    elif agg_method == "median":
        # Init median as mean
        median_embeddings = torch.unsqueeze(mean_embeddings, dim=1)
        c = 0.5
        for i in range(5):
            errors = median_embeddings - embeddings
            # Poor man's Newton's method
            denom = torch.sqrt(torch.sum(errors ** 2, axis=2, keepdims=True) + c ** 2)
            dw = -torch.sum(errors / denom, axis=1, keepdims=True) / torch.sum(
                1.0 / denom, axis=1, keepdims=True
            )
            median_embeddings += dw
        return torch.squeeze(median_embeddings, dim=1)

    elif (
        agg_method.startswith("cosine")
        or agg_method.startswith("euclidean")
        or agg_method.startswith("abs")
    ):
        epsilon = 1e-6

        if agg_method.startswith("cosine"):
            # Normalize all embeddings to unit vectors
            norm_embeddings = embeddings / (
                torch.norm(embeddings, dim=2, keepdim=True) + epsilon
            )
            # Calculate cosine angle between all support samples in each class: ways x shots x shots
            # Make negative, as higher cosine angle means greater correlation
            cos = torch.bmm(norm_embeddings, norm_embeddings.permute(0, 2, 1))
            attn = (torch.sum(cos, dim=1) - 1) / (shots - 1)
        elif agg_method.startswith("euclidean"):
            # dist: ways x shots x shots
            dist = (
                (embeddings.unsqueeze(dim=2) - embeddings.unsqueeze(dim=1)) ** 2
            ).sum(dim=-1)
            attn = -torch.sum(dist, dim=1) / (shots - 1)
        elif agg_method.startswith("abs"):
            # dist: ways x shots x shots
            dist = (
                torch.abs(embeddings.unsqueeze(dim=2) - embeddings.unsqueeze(dim=1))
            ).sum(dim=-1)
            attn = -torch.sum(dist, dim=1) / (shots - 1)

        # Parse softmax temperature (default=1)
        T = float(agg_method.split("_")[-1]) if "_" in agg_method else 1
        weights = F.softmax(attn / T, dim=1).unsqueeze(dim=2)
        weighted_embeddings = embeddings * weights
        return weighted_embeddings.sum(dim=1)

    else:
        raise NotImplementedError


class BinaryOutlierDetector(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        self.fc = nn.Linear(self.dim, 1)

    def forward(self, x):
        return self.fc(x)

class Transformer(nn.Module):
    def __init__(
        self,
        ways,
        shot,
        num_layers,
        nhead,
        d_model,
        dim_feedforward,
        device,
        cls_type="cls_learn",
        pos_type="pos_learn",
        agg_method="mean",
        transformer_metric="dot_prod",
    ):
        super().__init__()
        self.ways = ways
        self.shot = shot

        self.cls_type = cls_type
        self.pos_type = pos_type
        self.agg_method = agg_method

        if self.cls_type == "cls_learn":
            self.cls_embeddings = nn.Embedding(
                ways, dim_feedforward
            )
        elif self.cls_type == "rand_const":
            self.cls_embeddings = nn.Embedding(
                ways, dim_feedforward
            ).requires_grad_(False)

        if self.pos_type == "pos_learn":
            self.pos_embeddings = nn.Embedding(
                ways, dim_feedforward
            )
        elif self.pos_type == "rand_const":
            self.pos_embeddings = nn.Embedding(
                ways, dim_feedforward
            ).requires_grad_(False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout=0.1,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.device = device

    def forward(self, x): # support feat. (n, d)
        ways = self.ways
        shot = self.shot

        n_arng = torch.arange(ways, device=self.device)

        # Concatenate cls tokens with support embeddings
        if self.cls_type in ["cls_learn", "rand_const"]:
            cls_tokens = self.cls_embeddings(n_arng)  # (ways, dim)
        elif self.cls_type == "proto":
            cls_tokens = gen_prototypes(x, ways, shot, self.agg_method)  # (ways, dim)
        else:
            raise NotImplementedError

        cls_sup_embeds = torch.cat((cls_tokens, x), dim=0)  # (ways*(shot+1), dim)
        cls_sup_embeds = torch.unsqueeze(
            cls_sup_embeds, dim=1
        )  # (ways*(shot+1), BS, dim)

        # Position embeddings based on class ID
        pos_idx = torch.cat((n_arng, torch.repeat_interleave(n_arng, shot)))
        pos_tokens = torch.unsqueeze(
            self.pos_embeddings(pos_idx), dim=1
        )  # (ways*(shot+1), BS, dim)

        # Inputs combined with position encoding
        transformer_input = cls_sup_embeds + pos_tokens

        return self.encoder(transformer_input) # (n, b, d)


class ProtoNet_transformer(nn.Module):
    def __init__(self, args):
        super(ProtoNet_transformer, self).__init__()
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.dist_method = args.dist_method
        self.in_channels = args.pc_in_dim
        self.n_points = args.pc_npts
        self.use_attention = args.use_attention

        self.encoder = DGCNN(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k)
        self.base_learner = BaseLearner(args.dgcnn_mlp_widths[-1], args.base_widths)

        if self.use_attention:
            self.att_learner = SelfAttention(args.dgcnn_mlp_widths[-1], args.output_dim)
        else:
            self.linear_mapper = nn.Conv1d(args.dgcnn_mlp_widths[-1], args.output_dim, 1, bias=False)

        # self.classes = [3, 11, 10, 0, 8, 4]
        self.acc = {1: 0., 0: 0.}
        self.original_acc = {1: 0., 0: 0.}  # 1 is orignal_clean ratio>50%
        self.clean_count = {1: 0, 0: 0}
        self.acc_0 = 0
        self.shot_level_clean_ratio = 0
        self.feat_dim = 192


        self.proj_trans_in = nn.Parameter(
            torch.nn.init.orthogonal_(
                torch.empty(self.feat_dim, 128)
            )
        )
        self.proj_trans_out = nn.Parameter(self.proj_trans_in.data.detach())

        self.transformer = Transformer(
            ways=self.n_way,
            shot=self.k_shot,
            num_layers=3,
            nhead=8,
            d_model=128,
            dim_feedforward=128,
            device='cuda',
            cls_type='rand_const',
            pos_type='pos_learn',
            transformer_metric='dot_prod',
        )

        self.binary_outlier_detector = BinaryOutlierDetector(self.feat_dim)

        self.binary_outlier_loss_weight = 0.5
        self.clean_proto_loss_weight = 1


    def forward(self, support_x, support_y, query_x, query_y, gt_support_y=None, gt_query_y=None, train=False, logger=None, step=None, path=None, sampled_classes=None,
                bg_pcd_x=None, bg_pcd_y=None, support_c=None, support_flag=None, pcd_1024=None, label_1024=None, pcd_cutout=None, label_cutout=None):
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
        support_x = support_x.view(self.n_way*self.k_shot, self.in_channels, self.n_points)
        support_feat = self.getFeatures(support_x)
        support_feat = support_feat.view(self.n_way, self.k_shot, -1, self.n_points)
        query_feat = self.getFeatures(query_x) #(n_queries, feat_dim, num_points)

        fg_mask = support_y
        bg_mask = torch.logical_not(support_y)

        support_fg_feat = self.getMaskedFeatures(support_feat, fg_mask) # (n_way, k_shot, feat_dim)
        suppoer_bg_feat = self.getMaskedFeatures(support_feat, bg_mask)

        # prototype learning
        # get fg proto from transformer
        transformer_input = support_fg_feat.view(-1, self.feat_dim) # (n, d)
        transformer_input = transformer_input @ self.proj_trans_in

        transformer_output = self.transformer(transformer_input).squeeze(1) # (n, d)
        transformer_output = transformer_output @ self.proj_trans_out.T

        # collect fg proto
        fg_prototypes = []
        for way in range(self.n_way):
            fg_prototypes.append(transformer_output[way])

        # --------- compute binary classification loss -----------
        if train == True:
            binary_logits = self.binary_outlier_detector(transformer_output[self.n_way:]) # logits (n, 1)

            binary_label = torch.sum(gt_support_y, dim=-1) # (N, K)
            binary_label = torch.where(binary_label > 0,
                                       torch.tensor(1, dtype=torch.float32, device=support_y.device),
                                       torch.tensor(0, dtype=torch.float32, device=support_y.device)).view(-1, 1)

            binary_loss = self.binary_outlier_loss_weight * F.binary_cross_entropy_with_logits(
                binary_logits, binary_label)
        # --------------------------------------------------------

        # ------------------- clean proto loss -------------------
        if train == True:
            # get clean proto
            binary_label = torch.sum(gt_support_y, dim=-1)  # (N, K)
            binary_label = torch.where(binary_label > 0,
                                       torch.tensor(1, dtype=torch.float32, device=support_y.device),
                                       torch.tensor(0, dtype=torch.float32, device=support_y.device)) # (n, k). 1 is clean, 0 is noise
            # binary_label = binary_label.unsqueeze(-1).repeat(1,1,self.feat_dim) # (n, k, d)

            clean_proto_list = []
            for way in range(self.n_way):
                mask = binary_label[way] # (k,)
                clean_proto = support_fg_feat[way][mask.bool()] # (k, d)
                print(clean_proto.shape)
                clean_proto = torch.mean(clean_proto, dim=0) # (k,)
                clean_proto_list.append(clean_proto)
            clean_proto_list = torch.stack(clean_proto_list, dim=0) # (n, d)
            fg_proto_list = torch.stack(fg_prototypes, dim=0)

            clean_proto_loss = self.clean_proto_loss_weight *((fg_proto_list - clean_proto_list) ** 2).sum() / self.n_way
        # ----------------------------------------------------------









        _, bg_prototype = self.getPrototype(support_fg_feat, suppoer_bg_feat) # fg_proto: [proto1, proto2,]
        prototypes = [bg_prototype] + fg_prototypes

        # non-parametric metric learning
        similarity = [self.calculateSimilarity(query_feat, prototype, self.dist_method) for prototype in prototypes]

        query_pred = torch.stack(similarity, dim=1) #(n_queries, n_way+1, num_points)
        loss = self.computeCrossEntropyLoss(query_pred, query_y)

        if train == True:
            query_acc_LP = 0.5
            query_acc_original = 0.5
            clean_ratio_LP_avg = 0.5
            clean_ratio_original_avg = 0.5
            return query_pred, loss, binary_loss,clean_proto_loss, query_acc_LP, query_acc_original, clean_ratio_LP_avg, clean_ratio_original_avg
        else:
            return query_pred, loss

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

    def getMaskedFeatures(self, feat, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            feat: input features, shape: (n_way, k_shot, feat_dim, num_points)
            mask: binary mask, shape: (n_way, k_shot, num_points)
        Return:
            masked_feat: masked features, shape: (n_way, k_shot, feat_dim)
        """
        mask = mask.unsqueeze(2)
        masked_feat = torch.sum(feat * mask, dim=3) / (mask.sum(dim=3) + 1e-5)
        return masked_feat

    def getPrototype(self, fg_feat, bg_feat, clean_flag=None):
        """
        Average the features to obtain the prototype

        Args:
            fg_feat: foreground features for each way/shot, shape: (n_way, k_shot, feat_dim)
            bg_feat: background features for each way/shot, shape: (n_way, k_shot, feat_dim)
            clean_flag: (nway, kshot) clean is 1, noise is 0.
        Returns:
            fg_prototypes: a list of n_way foreground prototypes, each prototype is a vector with shape (feat_dim,)
            bg_prototype: background prototype, a vector with shape (feat_dim,)
        """
        if clean_flag != None:
            fg_prototypes = []
            for way in range(self.n_way):
                mask = clean_flag[way].unsqueeze(-1).repeat(1, self.feat_dim) # (k, d)
                num_clean = torch.sum(clean_flag[way])
                print(num_clean)
                tmp_proto = torch.sum(fg_feat[way] * mask, dim=0) / num_clean
                fg_prototypes.append(tmp_proto)
        else:
            fg_prototypes = [fg_feat[way, ...].sum(dim=0) / self.k_shot for way in range(self.n_way)]
        bg_prototype =  bg_feat.sum(dim=(0,1)) / (self.n_way * self.k_shot)
        return fg_prototypes, bg_prototype

    def calculateSimilarity(self, feat,  prototype, method='cosine', scaler=10):
        """
        Calculate the Similarity between query point-level features and prototypes

        Args:
            feat: input query point-level features
                  shape: (n_queries, feat_dim, num_points)
            prototype: prototype of one semantic class
                       shape: (feat_dim,)
            method: 'cosine' or 'euclidean', different ways to calculate similarity
            scaler: used when 'cosine' distance is computed.
                    By multiplying the factor with cosine distance can achieve comparable performance
                    as using squared Euclidean distance (refer to PANet [ICCV2019])
        Return:
            similarity: similarity between query point to prototype
                        shape: (n_queries, 1, num_points)
        """
        if method == 'cosine':
            similarity = F.cosine_similarity(feat, prototype[None, ..., None], dim=1) * scaler
        elif method == 'euclidean':
            similarity = - F.pairwise_distance(feat, prototype[None, ..., None], p=2)**2
        else:
            raise NotImplementedError('Error! Distance computation method (%s) is unknown!' %method)
        return similarity

    def computeCrossEntropyLoss(self, query_logits, query_labels):
        """ Calculate the CrossEntropy Loss for query set
        """
        return F.cross_entropy(query_logits, query_labels)