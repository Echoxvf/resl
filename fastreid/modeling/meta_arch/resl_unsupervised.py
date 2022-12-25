# encoding: utf-8
import copy
from sklearn import cluster

import torch
from torch import nn
import torch.nn.functional as F
from fastreid.config import configurable
from fastreid.layers import get_norm
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.backbones.resnet import Bottleneck
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY
from dropblock import DropBlock2D


@META_ARCH_REGISTRY.register()
class RESLUnsupervised(nn.Module):
    """

    """
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
            oim_loss,
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
            b21_head:
            b22_head:
            b3_head:
            b31_head:
            b32_head:
            b33_head:
            pixel_mean:
            pixel_std:
            loss_kwargs:
        """

        super().__init__()

        self.backbone = backbone
        self.drop_block = DropBlock2D(block_size=3, drop_prob=0.1)
        # branch1
        self.b1 = neck1
        # branch2
        self.b2 = neck2
        # branch3
        self.b3 = neck3
        self.b1_head = b1_head
        self.b2_head = b2_head
        self.b3_head = b3_head
        self.oim_loss = oim_loss
        self.num_ids = 3000 
        self.backbone_mean = copy.deepcopy(self.backbone)
        # branch1
        self.b1_mean = copy.deepcopy(self.b1)
        # branch2
        self.b2_mean = copy.deepcopy(self.b2)
        # branch3
        self.b3_mean = copy.deepcopy(self.b3)
        self.b1_head_mean = copy.deepcopy(self.b1_head)
        self.b2_head_mean = copy.deepcopy(self.b2_head)
        self.b3_head_mean = copy.deepcopy(self.b3_head)



        self.weights_w = torch.nn.Linear(2048,1)
        for module, module_m in zip([self.backbone, self.b1,self.b2,self.b3, self.b1_head, self.b2_head, self.b3_head],\
            [self.backbone_mean, self.b1_mean,self.b2_mean,self.b3_mean, self.b1_head_mean, self.b2_head_mean, self.b3_head_mean]):
            for param , param_m in zip(module.parameters(), module_m.parameters()):
                param_m.requires_grad = False
                param_m.data.copy_(param.data)
        self.loss_kwargs = loss_kwargs
        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(1, -1, 1, 1), False)
    def syn_mean_net(self,):
        print("synchronize mean net")
        for module, module_m in zip([self.backbone, self.b1,self.b2,self.b3, self.b1_head, self.b2_head, self.b3_head],\
                    [self.backbone_mean, self.b1_mean,self.b2_mean,self.b3_mean, self.b1_head_mean, self.b2_head_mean, self.b3_head_mean]):
            for param , param_m in zip(module.state_dict(), module_m.state_dict()):
                module_m.state_dict()[param_m].copy_(module.state_dict()[param])
    @classmethod
    def from_config(cls, cfg):
        oim_loss = OIMLoss_ori(2048, 5555)
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
            all_blocks.layer3,
            all_blocks.layer4[:-1],
        )
        # res_conv4 = nn.Sequential(*all_blocks.layer3[1:])
        res_g_conv5 = all_blocks.layer4[-1:]
        # res_conv4 = nn.Sequential(*all_blocks.layer3[1:])
        # res_g_conv5 = all_blocks.layer4

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
            copy.deepcopy (res_g_conv5),
            # Bottleneck(2048, 512, bn_norm, False, with_se)
        )
        # branch3
        neck3 = nn.Sequential(
            # copy.deepcopy(res_conv4),
            copy.deepcopy(res_g_conv5),
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
            'oim_loss':oim_loss ,
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

    def forward(self, batched_inputs, ema=False, clustering=False):

        if self.training:
            images = self.preprocess_image(batched_inputs)
            # print(images.shape)
            features = self.backbone(images)  # (bs, 2048, 16, 8)
            bs = features.size(0)

            features1,features2,features3= torch.split(features, int(bs/3), dim=0)
            # branch1
            b1_feat = self.b1(self.drop_block(features1))
            # branch2
            b2_feat = self.b2(self.drop_block(features2))
            # branch3
            b3_feat = self.b3(self.drop_block(features3))
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"]
            reliabilities = batched_inputs["reliability"]
            if targets.sum() < 0: targets.zero_()
            b1_outputs = self.b1_head(b1_feat, targets)
            b2_outputs = self.b2_head(b2_feat, targets)
            b3_outputs = self.b3_head(b3_feat, targets)
            with torch.no_grad():
                features_mean= self.backbone_mean(images)
                features_mean1,features_mean2,features_mean3= torch.split(features_mean, int(bs/3), dim=0)
                b1_features = self.b1_mean(features_mean1)
                b2_features = self.b2_mean(features_mean2)
                b3_features = self.b3_mean(features_mean3)
                b1_outputs_mean = self.b1_head_mean(b1_features, targets)
                b2_outputs_mean = self.b1_head_mean(b2_features, targets)
                b3_outputs_mean = self.b1_head_mean(b3_features, targets)
                b1_logits_mean= b1_outputs_mean['cls_outputs']
                b2_logits_mean= b2_outputs_mean['cls_outputs']
                b3_logits_mean= b3_outputs_mean['cls_outputs']
                # w = torch.softmax(self.weights_w, dim=0)
                w1 = self.weights_w(b1_outputs_mean['features'])
                w2 = self.weights_w(b2_outputs_mean['features'])
                w3 = self.weights_w(b3_outputs_mean['features'])
                weight = torch.softmax(torch.cat([w1,w2,w3], dim=-1), dim=-1)
                # logits_ens = b1_logits*w[0] + b2_logits * w[1] +b3_logits*w[2]
                s_ema = b1_logits_mean*weight[:,[0]] + b2_logits_mean * weight[:,[1]]+b3_logits_mean*weight[:,[2]]
                # s_ema = torch.mean(torch.stack([b1_logits_mean, b2_logits_mean, b3_logits_mean],dim=0), dim=0)
                self._update_mean_net()
            losses = self.losses(b1_outputs,
                                 b2_outputs,
                                 b3_outputs,
                                 targets,
                                 reliabilities, s_ema)
            
            
            return losses
        else:
            if clustering:
                images = self.preprocess_image(batched_inputs)
                features = self.backbone(images)  # (bs, 2048, 16, 8)
                features_mean= self.backbone_mean(images)
                # branch1
                b1_feat = self.b1(self.drop_block(features))
                b1_feat_mean = self.b1_mean(features_mean)
                # branch2
                b2_feat = self.b2(self.drop_block(features))
                b2_feat_mean = self.b2_mean(features_mean)
                # branch3
                b3_feat = self.b3(self.drop_block(features))
                b3_feat_mean = self.b3_mean(features_mean)

                b1_pool_feat = self.b1_head(b1_feat)
                b1_pool_feat_mean = self.b1_head_mean(b1_feat_mean)
                b2_pool_feat = self.b2_head(b2_feat)
                b2_pool_feat_mean = self.b2_head_mean(b2_feat_mean)
                b3_pool_feat = self.b3_head(b3_feat)
                b3_pool_feat_mean = self.b3_head_mean(b3_feat_mean)
                # b1_features = self.b1_mean(features_mean1)
                # b2_features = self.b2_mean(features_mean2)
                # b3_features = self.b3_mean(features_mean3)
                # outputs_b1 = self.b1_head_mean(b1_features, targets)
                # outputs_b2 = self.b1_head_mean(b2_features, targets)
                # outputs_b3 = self.b1_head_mean(b3_features, targets)
                # pred_feat = torch.cat([b1_pool_feat, b2_pool_feat, b3_pool_feat], dim=1)
                return [F.normalize(b1_pool_feat), F.normalize(b2_pool_feat), F.normalize(b3_pool_feat), F.normalize(b1_pool_feat_mean+b2_pool_feat_mean+b3_pool_feat_mean)]
            else:
                images = self.preprocess_image(batched_inputs)
                features = self.backbone_mean(images)  # (bs, 2048, 16, 8)
                # branch1
                b1_feat = self.b1_mean(self.drop_block(features))
                # branch2
                b2_feat = self.b2_mean(self.drop_block(features))
                # branch3
                b3_feat = self.b3_mean(self.drop_block(features))
                b1_pool_feat = self.b1_head_mean(b1_feat)
                b2_pool_feat = self.b2_head_mean(b2_feat)
                b3_pool_feat = self.b3_head_mean(b3_feat)                

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
        if len(images.shape)==5:
            images=images.split(1,dim=1)
            images = torch.cat(images, dim=0).squeeze()
        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self,
               b1_outputs,
               b2_outputs,
               b3_outputs, gt_labels, reliabilities, s_ema):
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
            loss_dict['loss_soft_cls']= (softentropy(b1_logits[:self.num_ids],s_ema[:self.num_ids])+softentropy(b2_logits[:self.num_ids],s_ema[:self.num_ids])+softentropy(b3_logits[:self.num_ids],s_ema[:self.num_ids]))/3
            # w = torch.softmax(self.weights_w, dim=0)
            w1 = self.weights_w(b1_pool_feat)
            w2 = self.weights_w(b2_pool_feat)
            w3 = self.weights_w(b3_pool_feat)
            # w = torch.tensor([w1,w2,w3])
            w = torch.softmax(torch.cat([w1,w2,w3], dim=-1), dim=-1)
            logits_ens = b1_logits*w[:,[0]] + b2_logits * w[:,[1]] +b3_logits*w[:,[2]]
            # logits_ens = b1_logits*w[0] + b2_logits * w[1]+b3_logits*w[2]
            loss_dict['loss_cls'] = (weighted_cross_entropy_loss(
                b1_logits[:self.num_ids],
                gt_labels,
                reliabilities,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale') + weighted_cross_entropy_loss(
                b2_logits[:self.num_ids],
                gt_labels,
                reliabilities,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale') +  weighted_cross_entropy_loss(
                b3_logits[:self.num_ids],
                gt_labels,
                reliabilities,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale')+  weighted_cross_entropy_loss(
                logits_ens[:self.num_ids],
                gt_labels,
                reliabilities,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale'))/4

            
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
        if "OIMLoss" in loss_names:
            loss1 = self.oim_loss(b1_pool_feat, gt_labels)
            loss2 = self.oim_loss(b2_pool_feat, gt_labels)
            loss3 = self.oim_loss(b3_pool_feat, gt_labels)
            loss_dict['loss1'] =  loss1+loss2+loss3
        return loss_dict, (b1_pool_feat+b2_pool_feat+b3_pool_feat)/3, gt_labels

    def _update_mean_net(self, m=0.999):
        for module, module_m in zip([self.backbone, self.b1,self.b2,self.b3, self.b1_head, self.b2_head, self.b3_head],\
            [self.backbone_mean, self.b1_mean,self.b2_mean,self.b3_mean, self.b1_head_mean, self.b2_head_mean, self.b3_head_mean]):
            for param , param_m in zip(module.parameters(), module_m.parameters()):
                param_m.data = param_m.data * m + param.data * (1. - m)