import torch
import torch.nn as nn
from einops import rearrange
from torch.distributions.normal import Normal
import numpy as np
from model import MattingRefine_firststage
from torch.nn import functional as F

from model import MattingRefine_secondstage


class SparseDispatcher(object):

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates  # 设有4个bn，3个专家  [4,3]
        self._num_experts = num_experts  # 3
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)  # [12,2],[12,2]  找出不是0的gates，然后对其进行sort
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)  # [12,1]
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]  # 12  描述expert对应的batchsize是哪个
        # calculate num samples that each expert gets
        self._part_sizes = (gates != 0).sum(0).tolist()  # [4，4，4]每个expert得到了多少个samples  list的长度为expert的数量
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]  # [12,3]  只是将gates（也就是expert）expend成[12,3]  第一维expend
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)  # [12,1] 把gates重新按照_expert_index拿到非零的门网络

    def dispatch(self, inp):

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)  # [12,2048*2048*3] 注意这边是按照batch_index摞起来的
        print(self._gates)
        print(self._batch_index)
        print(inp_exp.shape)
        # if sum(self._part_sizes)!=8:
        #     inp_exp = inp[self._batch_index].squeeze(1)
        #     return [inp,inp,inp]

        # else:
        print(self._part_sizes)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        # apply exp to expert outputs, so we are not longer in log space
        ans = []
        # print(expert_out[0][0].shape)#1
        # print(expert_out[0][1].shape)3
        # print(expert_out[0][2].shape)1
        # print(expert_out[0][3].shape)32
        for i in range(0, len(expert_out[0])):
            stitched = expert_out[0][i]

            for j in range(1, len(expert_out)):
                stitched = torch.cat([stitched, expert_out[j][i]], 0)  # [batchsiz,3,2048,2048]
            stitched = stitched.exp()
            if multiply_by_gates:
                stitched = stitched.mul(self._nonzero_gates.unsqueeze(2).unsqueeze(2))  # [batchsiz,3,2048,2048]
            zeros = torch.zeros(self._gates.size(0), expert_out[0][i].size(1), expert_out[0][i].size(2),
                                expert_out[0][i].size(3), requires_grad=True,
                                device=stitched.device)  # [4,3,2048,2048]->[batchsize的个数,]
            # print(zeros.shape)#4,1,h,w
            # print(stitched.shape)#8,1,h w
            # combine samples that have been processed by the same k experts
            combined = zeros.index_add(0, self._batch_index, stitched.float())  # [4,3,2048,2048]
            # add eps to all zero values in order to avoid nans when going back to log space
            combined[combined == 0] = np.finfo(float).eps
            combined.log()
            ans.append(combined)
        # back to log space
        return ans

    def expert_to_gates(self):
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MoE(nn.Module):

    def __init__(self, input_size, num_experts, model_backbone, model_backbone_scale,
                 model_refine_mode, model_refine_sample_pixels, model_refine_thresholding, model_refine_kernel_size,
                 noisy_gating=True, k=2):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.input_size = input_size
        self.k = k
        self.backbone_scale = model_backbone_scale
        # instantiate experts
        self.experts = nn.ModuleList([MattingRefine_firststage(
            model_backbone,
            model_backbone_scale,
        ) for i in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        self.refine = MattingRefine_secondstage(model_backbone,
                                                model_backbone_scale,
                                                model_refine_mode,
                                                model_refine_sample_pixels,
                                                model_refine_thresholding,
                                                model_refine_kernel_size)
        assert (self.k <= self.num_experts)

    def cv_squared(self, x):
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):

        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        eps = 1e-10
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / (noise_stddev + eps))
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / (noise_stddev + eps))
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, src, bgr, loss_coef=1e-2):
        # Downsample src and bgr for backbone
        src_sm = F.interpolate(src,
                               scale_factor=self.backbone_scale,
                               mode='bilinear',
                               align_corners=False,
                               recompute_scale_factor=True)
        bgr_sm = F.interpolate(bgr,
                               scale_factor=self.backbone_scale,
                               mode='bilinear',
                               align_corners=False,
                               recompute_scale_factor=True)

        # Base
        x = torch.cat([src_sm, bgr_sm], dim=1)
        gathered_x = rearrange(x, 'b c h w -> b (c h w)')

        gates, load = self.noisy_top_k_gating(gathered_x, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        #
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)  # 将batch分成每个expert进入多少
        # pha_sm: (B, 1, Hc, Wc) the coarse alpha prediction from matting base. Normalized to 0 ~ 1.
        # fgr_sm: (B, 3, Hc, Hc) the coarse foreground prediction from matting base. Normalized to 0 ~ 1.
        # err_sm: (B, 1, Hc, Wc) the coarse error prediction from matting base. Normalized to 0 ~ 1.
        # ref_sm: (B, 1, H/4, H/4) the quarter resolution refinement map. 1 indicates refined 4x4 patch locations.
        temp_ans = []
        for i in range(0, len(expert_inputs)):
            if expert_inputs[i].shape[0] != 0:
                temp = self.experts[i](expert_inputs[i])
                temp_ans.append(temp)
        temp_ans = dispatcher.combine(temp_ans)
        pred_pha, pred_fgr, pred_pha_sm, pred_fgr_sm, pred_err_sm, pred_ref_sm = self.refine(src, bgr, src_sm, temp_ans[0],
                                                                                   temp_ans[1], temp_ans[2],
                                                                                   temp_ans[3])
        return pred_pha, pred_fgr, pred_pha_sm, pred_fgr_sm, pred_err_sm, pred_ref_sm,loss


