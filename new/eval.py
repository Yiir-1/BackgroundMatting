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


import os
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms as T
from dataset import ImagesDataset, ZipDataset, VideoDataset, SampleDataset
from dataset import augmentation as A
from model import MattingBase, MattingRefine
from metric import *
from ptflops import get_model_complexity_info
from torchsummary import summary
import time
# --------------- Arguments ---------------
parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='./evaldata')
parser.add_argument('--model-path', type=str, default='./pretrain/pytorch_resnet50.pth')
parser.add_argument('--model-backbone', type=str,default='resnet50', choices=['resnet101', 'resnet50', 'mobilenetv2'])
parser.add_argument('--model-backbone-scale', type=float, default=0.25)
parser.add_argument('--model-refine-mode', type=str, default='sampling', choices=['full', 'sampling', 'thresholding'])
parser.add_argument('--model-refine-sample-pixels', type=int, default=80000)
parser.add_argument('--model-refine-threshold', type=float, default=0.7)
parser.add_argument('--model-refine-kernel-size', type=int, default=3)
parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')

args = parser.parse_args()
args.batch_size = 1


# --------------- Loading ---------------
def eval():
    dataset_valid = ZipDataset([
        ZipDataset([
            ImagesDataset(os.path.join(args.data_path, 'pha'), mode='L'),
            ImagesDataset(os.path.join(args.data_path, 'fgr'), mode='RGB')
        ], transforms=A.PairCompose([
            A.PairRandomAffineAndResize((512, 512), degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.55, 0.9), shear=(-5, 5)),
            A.PairApply(T.ToTensor())
        ]), assert_equal_length=True),
        ImagesDataset(os.path.join(args.data_path, 'Backgrounds'), mode='RGB', transforms=T.Compose([
            A.RandomAffineAndResize((512, 512), degrees=(-5, 5), translate=(0.1, 0.1), scale=(1, 1.2), shear=(-5, 5)),
            T.ToTensor()
        ])),
    ])
    dataset_valid = SampleDataset(dataset_valid, 85)
    dataloader_valid = DataLoader(dataset_valid, batch_size=1, num_workers=16)
    gen_data = []
    for (true_pha, true_fgr), true_bgr in dataloader_valid:
        gen_data.append([true_pha.cpu().detach().numpy(),
                         true_fgr.cpu().detach().numpy(),
                         true_bgr.cpu().detach().numpy()])

    pd_sad, pd_mse ,fps= paddle_valid(gen_data)
    print(f'output:  SAD: {pd_sad / len(gen_data)}, MSE: {pd_mse / len(gen_data)},fps: {fps}')

def prepare_input(resolution):
    x1 = torch.FloatTensor(1, *resolution)
    x2 = torch.FloatTensor(1, *resolution)
    return dict(src=x1,bgr=x2)


# --------------- utils ---------------
def paddle_valid(dataloader):
    model = MattingRefine(
        args.model_backbone,
        args.model_backbone_scale,
        args.model_refine_mode,
        args.model_refine_sample_pixels,
        args.model_refine_threshold,
        args.model_refine_kernel_size)
    summary(model, input_size=[(3, 512, 512),(3,512,512)], batch_size=2, device="cpu")
    device = torch.device(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)
    
    
    macs, params = get_model_complexity_info(model, (3, 512, 512),input_constructor=prepare_input, as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    
    
    
    model.eval()
    loss_count = 0
    sad = 0
    mse = 0
    fps=0.0
    with torch.no_grad():
        
        for true_pha, true_fgr, true_bgr in tqdm(dataloader):
            
            t1 = time.time()
            true_pha = torch.tensor(true_pha)
            true_fgr = torch.tensor(true_fgr)
            true_bgr = torch.tensor(true_bgr)
            true_src = true_pha * true_fgr + (1 - true_pha) * true_bgr
            
            pred_pha, *_ = model(true_src, true_bgr)

            img = true_pha[0][0].cpu().numpy()
            trimap = gen_trimap(img)
            mask_pha = torch.tensor([trimap]).unsqueeze(1)

            sad += BatchSAD(pred_pha, true_pha, mask_pha)
            mse += BatchMSE(pred_pha, true_pha, mask_pha)
            loss_count += 1
            fps = (fps + (1. / (time.time() - t1))) / 2
    return sad, mse,fps


# --------------- Start ---------------
if __name__ == '__main__':
    eval()
