# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py


import torch
import torch.nn as nn
from model import MattingRefine, MattingRefine_secondstage, MattingRefine_firststage
import pandas as pd
from torch.nn import functional as F
class BaseMoE_kmeans(nn.Module):
    def __init__(self, num_experts, model_backbone,model_backbone_scale,model_refine_mode,
                 model_refine_sample_pixels,model_refine_thresholding,model_refine_kernel_size):
        super(BaseMoE_kmeans, self).__init__()
        self.num_experts = num_experts
        self.model_backbone_scale=model_backbone_scale
        # instantiate experts
        self.experts = nn.ModuleList([MattingRefine_firststage(
            model_backbone,
            model_backbone_scale,
        ) for i in range(self.num_experts)])
        self.df=pd.read_csv('bg_clusters_valid2.csv')
        self.refine = MattingRefine_secondstage(model_backbone,
                                                model_backbone_scale,
                                                model_refine_mode,
                                                model_refine_sample_pixels,
                                                model_refine_thresholding,
                                                model_refine_kernel_size)
    def forward(self, src, bgr,name):
        src_sm = F.interpolate(src,
                               scale_factor=self.model_backbone_scale,
                               mode='bilinear',
                               align_corners=False,
                               recompute_scale_factor=True)
        bgr_sm = F.interpolate(bgr,
                               scale_factor=self.model_backbone_scale,
                               mode='bilinear',
                               align_corners=False,
                               recompute_scale_factor=True)
        # Base
        x = torch.cat([src_sm, bgr_sm], dim=1)
        label=self.df[self.df['name']==str(name)]['label']
        pha_sm, fgr_sm, err_sm, hid_sm= self.experts[label.values[0]](x)
        pred_pha, pred_fgr, pred_pha_sm, pred_fgr_sm, pred_err_sm, pred_ref_sm = self.refine(src, bgr, src_sm, pha_sm, fgr_sm, err_sm, hid_sm)
        return pred_pha, pred_fgr, pred_pha_sm, pred_fgr_sm, pred_err_sm, pred_ref_sm