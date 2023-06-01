from .build import build
from .barlowtwins_head import BarlowTwinsHead
from .byol_head import BYOLHead
from .cls_head import LinearClsHead
from .dino_head import DinoHead
from .icc_head import MomentumICCHead, ICCHead
from .moco_head import MoCoHead
from .simsiam_head import SimSiamHead
from .swav_head import SwAVHead
from .cohead import CoTrainHead
from .multilevel_cls_head import MLLinearClsHead

from .mm_head import MultiModalHead, MomentumMultiModalHead

from .test import TestHead
from .rcc_head import RCCHead
__all__ = ['build']