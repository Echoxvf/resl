# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from turtle import forward
import torch
from torch import nn
import copy
from fastreid.utils.registry import Registry

META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.
The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    # if cfg.MODEL.EMA:
    #     model = MeanTeacher(cfg)
    #     return model
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    return model

class MeanTeacher(nn.Module):
    def __init__(self, cfg):
        super(MeanTeacher, self).__init__()
        meta_arch = cfg.MODEL.META_ARCHITECTURE
        self.model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
        self.model.to(torch.device(cfg.MODEL.DEVICE))
        self.model_ema = copy.deepcopy(self.model)
        for (model_name, param),  (ema_name, param_m)in zip(self.model.named_parameters(), self.model_ema.named_parameters()):
        # for param, param_m in zip(self.model.name_parameters(), self.model_ema.parameters()):
            # print(model_name, "======",ema_name)
            param_m.requires_grad = False
            param_m.data.copy_(param.data) 
        self.alpha = 0.999


    def forward(self, batched_inputs):
        if self.training:
            loss_dict = self.model(batched_inputs)
            with torch.no_grad():
                loss_dict_m = self.model_ema(copy.deepcopy(batched_inputs))
                self._update_mean_net()  # update mean net
            return loss_dict
        else:
            return self.model_ema(batched_inputs)

    @torch.no_grad()
    def _update_mean_net(self):
        for param, param_m in zip(self.model.parameters(), self.model_ema.parameters()):
            param_m.data.mul_(self.alpha).add_(param.data, alpha=1 - self.alpha)