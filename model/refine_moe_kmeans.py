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
from torch.distributions.normal import Normal
import numpy as np
from model import MattingRefine
from model.autoencoder import Autoencoder
from sklearn.decomposition import KernelPCA
from sklearn.cluster import MiniBatchKMeans
import pandas as pd
class MoE_kmeans(nn.Module):
    def __init__(self, num_experts, model_backbone,model_backbone_scale,model_refine_mode,
                 model_refine_sample_pixels,model_refine_thresholding,model_refine_kernel_size):
        super(MoE_kmeans, self).__init__()
        self.num_experts = num_experts
        # instantiate experts
        self.experts = nn.ModuleList([MattingRefine(model_backbone,
                           model_backbone_scale,
                           model_refine_mode,
                           model_refine_sample_pixels,
                           model_refine_thresholding,
                           model_refine_kernel_size)for i in range(num_experts)])
        self.df=pd.read_csv('bg_clusters_valid.csv')

    def forward(self, src, bgr,name):
        label=self.df[self.df['name']==str(name)]['label']
        pred_pha, pred_fgr, pred_pha_sm, pred_fgr_sm, pred_err_sm, pred_ref_sm = self.experts[label.values[0]](src, bgr)
        return pred_pha, pred_fgr, pred_pha_sm, pred_fgr_sm, pred_err_sm, pred_ref_sm