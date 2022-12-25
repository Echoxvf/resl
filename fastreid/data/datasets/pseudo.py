# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import os.path as osp
import re
import warnings

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PseudoDataset(ImageDataset):
    def __init__(self, train=[], **kwargs):
        super(PseudoDataset, self).__init__(train,[],[] ,**kwargs)