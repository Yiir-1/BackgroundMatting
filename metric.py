#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import cv2
import math
import cv2
import numpy as np
import torch

def gen_trimap(alpha, ksize=3, iterations=10):
    import cv2
    import numpy as np
    alpha = alpha * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(alpha, kernel, iterations=iterations)
    eroded = cv2.erode(alpha, kernel, iterations=iterations)
    trimap = np.zeros(alpha.shape) + 128
    trimap[eroded >= 255] = 255
    trimap[dilated <= 0] = 0
    return trimap


def BatchSAD(pred, target, mask, scale=2):
    # function loss = compute_sad_loss(pred, target, trimap)
    # error_map = (pred - target).abs() / 255.
    # batch_loss = (error_map * mask).view(B, -1).sum(dim=-1)
    # batch_loss = batch_loss / 1000.
    # return batch_loss.data.cpu().numpy()
    B = target.shape[0]
    error_map = (pred - target).abs()
    batch_loss = (error_map.cpu() * (mask == 128)).reshape((B, -1)).sum(axis=-1)
    batch_loss = batch_loss / 1000.
    return batch_loss.sum().item()/B*scale


def BatchMSE(pred, target, mask, scale=2):
    # function loss = compute_mse_loss(pred, target, trimap)
    # error_map = (single(pred) - single(target)) / 255;
    # loss = sum(sum(error_map. ^ 2. * single(trimap == 128))) / sum(sum(single(trimap == 128)));
    B = target.shape[0]
    error_map = (pred - target)
    batch_loss = (error_map.pow(2).cpu() * (mask == 128)).reshape((B, -1)).sum(axis=-1)
    batch_loss = batch_loss / ((mask == 128).numpy().astype(float).reshape((B, -1)).sum(axis=-1) + 1.)
    batch_loss = batch_loss * 1000.
    return batch_loss.sum().item()/B*scale

from scipy.ndimage import gaussian_filter

def gradient(pd, gt, mask,scale=2):
    # function loss = compute_gradient_loss(pred, target, trimap)
    # pred = mat2gray(pred);
    # target = mat2gray(target);
    # [pred_x, pred_y] = gaussgradient(pred, 1.4);
    # [target_x, target_y] = gaussgradient(target, 1.4);
    # pred_amp = sqrt(pred_x. ^ 2 + pred_y. ^ 2);
    # target_amp = sqrt(target_x. ^ 2 + target_y. ^ 2);
    # error_map = (single(pred_amp) - single(target_amp)). ^ 2;
    # loss = sum(sum(error_map. * single(trimap == 128)));
    B=1
    pd_x = gaussian_filter(pd, sigma=1.4, order=[1, 0], output=np.float32)
    pd_y = gaussian_filter(pd, sigma=1.4, order=[0, 1], output=np.float32)
    gt_x = gaussian_filter(gt, sigma=1.4, order=[1, 0], output=np.float32)
    gt_y = gaussian_filter(gt, sigma=1.4, order=[0, 1], output=np.float32)
    pd_mag = np.sqrt(pd_x**2 + pd_y**2)
    gt_mag = np.sqrt(gt_x**2 + gt_y**2)
    error_map = np.square(pd_mag - gt_mag)
    error_map=torch.Tensor(error_map)
    loss =(error_map* (mask == 128)).reshape(B, -1).sum(axis=-1)*100
    return loss.sum().item()/B*scale



def connectivity_loss(pd, gt,mask, step=0.1,scale=2):
    B=1
    from scipy.ndimage import morphology
    from skimage.measure import label, regionprops
    h, w = pd.shape
    thresh_steps = np.arange(0, 1.1, step)
    l_map = -1 * np.ones((h, w), dtype=np.float32)
    lambda_map = np.ones((h, w), dtype=np.float32)
    for i in range(1, thresh_steps.size):
        pd_th = pd >= thresh_steps[i]
        gt_th = gt >= thresh_steps[i]
        intersection = (pd_th & gt_th).numpy().astype(np.uint8)
        # connected components
        _, output, stats, _ = cv2.connectedComponentsWithStats(
            intersection, connectivity=4)
        size = stats[1:, -1]
        omega = np.zeros((h, w), dtype=np.float32)
        # largest connected component of the intersection
        if len(size) != 0:
            max_id = np.argmax(size)
            # plus one to include background
            omega[output == max_id + 1] = 1

        flag = (l_map == -1) & (omega == 0)
        l_map[flag == 1] = thresh_steps[i-1]

    l_map[l_map == -1] = 1
    d_pd = pd - l_map
    d_gt = gt - l_map
    phi_pd = 1 - d_pd * (d_pd >= 0.15).float()
    phi_gt = 1 - d_gt * (d_gt >= 0.15).float()
    loss = (np.abs(phi_pd - phi_gt)*(mask==128)).reshape((B, -1)).sum(axis=-1)
    return loss.sum().item()/B*scale

