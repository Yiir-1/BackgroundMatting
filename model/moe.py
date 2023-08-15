

import torch
import torch.nn as nn
from einops import rearrange
from torch.distributions.normal import Normal
import numpy as np
from model import MattingRefine

class MoE(nn.Module):
    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """
    def __init__(self, input_size, num_experts, model_backbone,model_backbone_scale,model_refine_mode,model_refine_sample_pixels,model_refine_thresholding,model_refine_kernel_size, noisy_gating=True, k=2):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.input_size = input_size
        self.k = k
        # instantiate experts
        self.experts =  nn.ModuleList([MattingRefine(model_backbone,
                           model_backbone_scale,
                           model_refine_mode,
                           model_refine_sample_pixels,
                           model_refine_thresholding,
                           model_refine_kernel_size)for i in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
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
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            src: (B, 3, H, W) the source image. Channels are RGB values normalized to 0 ~ 1.
            bgr: (B, 3, H, W) the background image . Channels are RGB values normalized to 0 ~ 1.
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """

        x = rearrange(x, 'b c h w -> b (c h w)')#

        clean_logits = x @ self.w_gate# 假设bn为4， num—experts为3 torch.Size([4, 3])
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise#torch.Size([4, 3])
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))#torch.Size([4, 3])
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)# torch.Size([4, 3])
            logits = noisy_logits#torch.Size([4, 3])
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k+1, self.num_experts), dim=1)#两个都是torch.Size([4, 3]) 一个是概率，一个是索引
        top_k_logits = top_logits[:, :self.k]#选取k个experts  这个k需要小于或等于expert的 数量
        top_k_indices = top_indices[:, :self.k]#选取k个experts  这个k需要小于或等于expert的 数量
        top_k_gates = self.softmax(top_k_logits)#选取k个experts  这个k需要小于或等于expert的 数量

        zeros = torch.zeros_like(logits, requires_grad=True)#torch.Size([4, 3])
        gates = zeros.scatter(1, top_k_indices, top_k_gates)#torch.Size([4, 3])  选门

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, src,bgr, loss_coef=1e-2):
        """Args:
        src: (B, 3, H, W) the source image. Channels are RGB values normalized to 0 ~ 1.
        bgr: (B, 3, H, W) the background image . Channels are RGB values normalized to 0 ~ 1.
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        gates, load = self.noisy_top_k_gating(bgr, self.training)#gates.shape=[4,3]  load.shape=[3]
        # calculate importance loss
        importance = gates.sum(0)#[3]
        #
        loss = self.cv_squared(importance) + self.cv_squared(load)#计算损失
        loss = loss*loss_coef

        new_gates=gates.sum(0)
        sorted_experts, index_sorted_experts=new_gates.sort(0,True)
        index_sorted_experts=index_sorted_experts[:self.k]
        # print(bgr.shape)
        # print(expert_inputs[0].shape)
        # print(src.shape)
        expert_outputs = [self.experts[i](bgr,src) for i in index_sorted_experts]
        # sorted_expert = self.normalization(sorted_experts)#归一化
        min_a = torch.min(sorted_experts)
        max_a = torch.max(sorted_experts)
        sorted_expert = (sorted_experts - min_a) / (max_a - min_a)
        sorted_expert = sorted_expert / sorted_expert.sum()
        result=[]
        for i in range(0,6):
            for j in range(0,len(expert_outputs)):
                temp_list=[]
                tenp_ans=torch.zeros_like(expert_outputs[j][i])
                tenp_ans.add(expert_outputs[j][i]*sorted_expert[j])
            result.append(tenp_ans)
        return result[0],result[1],result[2],result[3],result[4], loss