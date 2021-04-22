import torch
import torch.nn as nn
import numpy as np

from ...utils import box_utils
from .point_head_template import PointHeadTemplate
from ...ops.roiaware_pool3d import roiaware_pool3d_utils

class SoftmaxFocalClassificationLoss(nn.Module):
    """
    Softmax focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SoftmaxFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        ce_loss = nn.functional.cross_entropy(input, target, reduction='none')#target.to(torch.int64)
        pt = torch.exp(-ce_loss)
        #loss = self.alpha * (1-pt)**self.gamma * ce_loss
        loss = ce_loss
        print('loss shape:')
        print(loss.shape, weights.shape)

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        loss *= weights
        loss = loss.mean()

        return loss


class PointSegHead(PointHeadTemplate):
    """
    A point segmentation head.
    Reference Paper: https://arxiv.org/abs/1706.02413
    """
    def __init__(self, num_class, input_channels, model_cfg, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=input_channels,
            output_channels=num_class+1
        )

    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
        """
        point_coords = input_dict['point_coords']
        gt_boxes = input_dict['gt_boxes']
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)

        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]) # extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        targets_dict = self.assign_stack_targets(
            points=point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=False
        )

        return targets_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                point_features_before_fusion: (N1 + N2 + N3 + ..., C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_labels (optional): (N1 + N2 + N3 + ...)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
        """
        if self.model_cfg.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            point_features = batch_dict['point_features_before_fusion']
        else:
            point_features = batch_dict['point_features']
        point_cls_preds = self.cls_layers(point_features)  # (total_points, num_class)

        ret_dict = {
            'point_cls_preds': point_cls_preds
        }

        softmax = nn.Softmax(dim=1)
        point_cls_scores = softmax(point_cls_preds)
        _, batch_dict['point_cls_scores'] = torch.max(point_cls_scores, 1)
        # print('detected scores:')
        # print(batch_dict['point_cls_scores'][:200])


        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
        else:##############################else branch only for showing debugging info
            targets_dict = self.assign_targets(batch_dict)
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
        self.forward_ret_dict = ret_dict

        return batch_dict

    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            SoftmaxFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict_1 = self.get_cls_layer_loss()

        point_loss = point_loss_cls
        tb_dict.update(tb_dict_1)
        return point_loss, tb_dict

    def get_cls_layer_loss(self, tb_dict=None):
        point_cls_labels = self.forward_ret_dict['point_cls_labels'].view(-1)
        point_cls_preds = self.forward_ret_dict['point_cls_preds'].view(-1, self.num_class+1)

        positives = (point_cls_labels > 0)
        negative_cls_weights = (point_cls_labels == 0) * 1.0
        cls_weights = (negative_cls_weights + 10.0 * positives).float()
        pos_normalizer = positives.sum(dim=0).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        ####################################### Weights Info
        print('Weights Info')
        print(cls_weights.shape)
        #print(point_cls_labels[:200])
        #print(point_cls_preds[:200])
        label = point_cls_labels.cpu().numpy()
        u,count = np.unique(label, return_counts=True)
        print(u)
        print(count)
        #print(cls_weights[:200])
        weight = cls_weights.tolist()
        print(len(set(weight)))
        ########################################

        # one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 2)
        # one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
        # one_hot_targets = one_hot_targets[..., 1:]
        # print(one_hot_targets.shape)
        # print(one_hot_targets[:3, :])
        # cls_loss_src = self.cls_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights)
        # point_loss_cls = cls_loss_src.sum()

        point_loss_cls = self.cls_loss_func(point_cls_preds, point_cls_labels, cls_weights)

        # loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        # point_loss_cls = point_loss_cls * loss_weights_dict['point_cls_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({
            'point_loss_cls': point_loss_cls.item(),
            'point_pos_num': pos_normalizer.item()
        })
        return point_loss_cls, tb_dict
