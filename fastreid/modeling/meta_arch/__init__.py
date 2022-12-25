# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import imp
from .build import META_ARCH_REGISTRY, build_model


# import all the meta_arch, so they will be registered
from .baseline import Baseline
from .moco import MoCo
from .distiller import Distiller
from .resl import RESL
from .reslv2 import RESLv2
from .baseline_unsupervised import BaselineUn
from .mean_teacher import MeanTeacher
from .resl_unsupervised import RESLUnsupervised
from .resl2_unsupervised import RESL2Unsupervised

