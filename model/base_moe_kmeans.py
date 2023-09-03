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


class Expert(nn.Module):
    def __init__(self, model_backbone, model_backbone_scale):  # input_dim代表输入维度，output_dim代表输出维度
        super(Expert, self).__init__()

        self.expert_layer =MattingRefine_firststage(
            model_backbone,
            model_backbone_scale,
        )

    def forward(self, x):
        out = self.expert_layer(x)
        return out


class BaseMoE_kmeans(nn.Module):
    def __init__(self, num_experts, model_backbone,model_backbone_scale,model_refine_mode,
                 model_refine_sample_pixels,model_refine_thresholding,model_refine_kernel_size):
        super(BaseMoE_kmeans, self).__init__()
        self.num_experts = num_experts
        self.model_backbone_scale=model_backbone_scale
        # instantiate experts

        '''专家网络'''
        for i in range(self.num_experts):
            setattr(self, "expert_layer" + str(i + 1), Expert(model_backbone, model_backbone_scale))
        self.expert_layers = [getattr(self, "expert_layer" + str(i + 1)) for i in range(self.num_experts)]
        self.df=pd.read_csv('bg_clusters_train2.csv')
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
        pha_sm, fgr_sm, err_sm, hid_sm= self.expert_layers[label.values[0]](x)
        pha_sm=pha_sm*0.8
        fgr_sm=fgr_sm*0.8
        err_sm=err_sm*0.8
        hid_sm=hid_sm*0.8
        for i in range(0,len(self.expert_layers)):
            if i !=label.values[0]:
                pha_sm_temp, fgr_sm_temp, err_sm_temp, hid_sm_temp= self.expert_layers[i](x)
                pha_sm+=pha_sm_temp*0.1
                fgr_sm+=fgr_sm_temp*0.1
                err_sm+=err_sm_temp*0.1
                hid_sm+=hid_sm_temp*0.1
        pred_pha, pred_fgr, pred_pha_sm, pred_fgr_sm, pred_err_sm, pred_ref_sm = self.refine(src, bgr, src_sm, pha_sm, fgr_sm, err_sm, hid_sm)
        return pred_pha, pred_fgr, pred_pha_sm, pred_fgr_sm, pred_err_sm, pred_ref_sm