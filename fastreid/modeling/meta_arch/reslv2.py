# encoding: utf-8
import copy

import torch
from torch import nn

from fastreid.config import configurable
from fastreid.layers import get_norm
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.backbones.resnet import Bottleneck
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY
from dropblock import DropBlock2D

@META_ARCH_REGISTRY.register()
class RESLv2(nn.Module):
    @configurable
    def __init__(
            self,
            *,
            backbone,
            neck1,
            neck2,
            neck3,
            b1_head,
            b2_head,
            b3_head,
            pixel_mean,
            pixel_std,
            loss_kwargs=None
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone:
            neck1:
            neck2:
            neck3:
            b1_head:
            b2_head:
            b3_head:
            pixel_mean:
            pixel_std:
            loss_kwargs:
        """
        super().__init__()
        self.backbone = backbone

        # branch1
        self.b1 = neck1
        self.drop_block = DropBlock2D(block_size=3, drop_prob=0.1)
        # branch2
        self.b2 = neck2
        # branch3
        self.b3 = neck3
        self.b1_head = b1_head
        self.b2_head = b2_head
        self.b3_head = b3_head
        self.loss_kwargs = loss_kwargs
        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(1, -1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        bn_norm = cfg.MODEL.BACKBONE.NORM
        with_se = cfg.MODEL.BACKBONE.WITH_SE

        all_blocks = build_backbone(cfg)

        # backbone
        backbone = nn.Sequential(
            all_blocks.conv1,
            all_blocks.bn1,
            all_blocks.relu,
            all_blocks.maxpool,
            all_blocks.layer1,
            all_blocks.layer2,
            all_blocks.layer3
        )
        res_g_conv5 = all_blocks.layer4
        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, bn_norm, False, with_se, downsample=nn.Sequential(
                nn.Conv2d(1024, 2048, 1, bias=False), get_norm(bn_norm, 2048))),
            Bottleneck(2048, 512, bn_norm, False, with_se),
            Bottleneck(2048, 512, bn_norm, False, with_se))
        res_p_conv5.load_state_dict(all_blocks.layer4.state_dict())
        # branch
        neck1 = nn.Sequential(
            # copy.deepcopy(res_conv4),
            copy.deepcopy(res_g_conv5)
        )
        # branch2
        neck2 = nn.Sequential(
            # copy.deepcopy(res_conv4),
            copy.deepcopy (res_p_conv5),
            # Bottleneck(2048, 512, bn_norm, False, with_se)
        )
        # branch3
        neck3 = nn.Sequential(
            # copy.deepcopy(res_conv4),
            copy.deepcopy(res_p_conv5),
            # Bottleneck(2048, 512, bn_norm, False, with_se)#,
            # Bottleneck(2048, 512, bn_norm, False, with_se)
        )
        b1_head = build_heads(cfg)
        b2_head = build_heads(cfg)
        b3_head = build_heads(cfg)

        return {
            'backbone': backbone,
            'neck1': neck1,
            'neck2': neck2,
            'neck3': neck3,
            'b1_head': b1_head,
            'b2_head': b2_head,
            'b3_head': b3_head,
            'pixel_mean': cfg.MODEL.PIXEL_MEAN,
            'pixel_std': cfg.MODEL.PIXEL_STD,
            'loss_kwargs':
                {
                    # loss name
                    'loss_names': cfg.MODEL.LOSSES.NAME,

                    # loss hyperparameters
                    'ce': {
                        'eps': cfg.MODEL.LOSSES.CE.EPSILON,
                        'alpha': cfg.MODEL.LOSSES.CE.ALPHA,
                        'scale': cfg.MODEL.LOSSES.CE.SCALE
                    },
                    'tri': {
                        'margin': cfg.MODEL.LOSSES.TRI.MARGIN,
                        'norm_feat': cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                        'hard_mining': cfg.MODEL.LOSSES.TRI.HARD_MINING,
                        'scale': cfg.MODEL.LOSSES.TRI.SCALE
                    },
                    'circle': {
                        'margin': cfg.MODEL.LOSSES.CIRCLE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.CIRCLE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.CIRCLE.SCALE
                    },
                    'cosface': {
                        'margin': cfg.MODEL.LOSSES.COSFACE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.COSFACE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.COSFACE.SCALE
                    }
                }
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images) 

        # branch1
        b1_feat = self.b1(self.drop_block(features))

        # branch2
        b2_feat = self.b2(self.drop_block(features))

        # branch3
        b3_feat = self.b3(self.drop_block(features))

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"]

            if targets.sum() < 0: targets.zero_()

            b1_outputs = self.b1_head(b1_feat, targets)
            b2_outputs = self.b2_head(b2_feat, targets)
            b3_outputs = self.b3_head(b3_feat, targets)

            losses = self.losses(b1_outputs,
                                 b2_outputs, 
                                 b3_outputs,
                                 targets)
            return losses
        else:
            b1_pool_feat = self.b1_head(b1_feat)
            b2_pool_feat = self.b2_head(b2_feat)
            b3_pool_feat = self.b3_head(b3_feat)

            pred_feat = torch.cat([b1_pool_feat, b2_pool_feat, b3_pool_feat], dim=1)
            return pred_feat

    def preprocess_image(self, batched_inputs):
        r"""
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs["images"].to(self.device)
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs.to(self.device)
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))
        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self,
               b1_outputs,
               b2_outputs,
               b3_outputs, gt_labels):
        # model predictions
        # fmt: off
        pred_class_logits = b1_outputs['pred_class_logits'].detach()
        b1_logits         = b1_outputs['cls_outputs']
        b2_logits         = b2_outputs['cls_outputs']
        b3_logits         = b3_outputs['cls_outputs']
        b1_pool_feat      = b1_outputs['features']
        b2_pool_feat      = b2_outputs['features']
        b3_pool_feat      = b3_outputs['features']
        # fmt: on

        # Log prediction accuracy
        log_accuracy(pred_class_logits, gt_labels)

        loss_dict = {}
        loss_names = self.loss_kwargs['loss_names']

        if "CrossEntropyLoss" in loss_names:
            ce_kwargs = self.loss_kwargs.get('ce')
            loss_dict['loss_cls_b1'] = cross_entropy_loss(
                b1_logits,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale') * 0.125

            loss_dict['loss_cls_b2'] = cross_entropy_loss(
                b2_logits,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale') * 0.125

            loss_dict['loss_cls_b3'] = cross_entropy_loss(
                b3_logits,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale') * 0.125


        if "TripletLoss" in loss_names:
            tri_kwargs = self.loss_kwargs.get('tri')
            loss_dict['loss_triplet_b1'] = triplet_loss(
                b1_pool_feat,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale') * 0.2

            loss_dict['loss_triplet_b2'] = triplet_loss(
                b2_pool_feat,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale') * 0.2

            loss_dict['loss_triplet_b3'] = triplet_loss(
                b3_pool_feat,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale') * 0.2
        return loss_dict, 0, 0
