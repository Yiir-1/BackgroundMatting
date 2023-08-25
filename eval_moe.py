
from torchvision.transforms.functional import to_tensor, to_pil_image

import os
import argparse
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms as T
from dataset import ImagesDataset, ZipDataset, VideoDataset, SampleDataset
from dataset import augmentation as A
from model import BaseMoE_kmeans
from metric import *


# --------------- Arguments ---------------
parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='./data')
parser.add_argument('--model-path', type=str, default='checkpoint/mattingrefine_resnet50_basemoe_kmeans/epoch-0-iter-55999-loss0.009505371563136578-model.pth')
parser.add_argument('--model-backbone', type=str,default='resnet50', choices=['resnet101', 'resnet50', 'mobilenetv2'])
parser.add_argument('--model-backbone-scale', type=float, default=0.25)
parser.add_argument('--model-refine-mode', type=str, default='sampling', choices=['full', 'sampling', 'thresholding'])
parser.add_argument('--model-refine-sample-pixels', type=int, default=80000)
parser.add_argument('--model-refine-threshold', type=float, default=0.7)
parser.add_argument('--model-refine-kernel-size', type=int, default=3)
parser.add_argument('--model-refine-thresholding', type=float, default=0.7)
parser.add_argument('--num-experts', type=int,  default=3)
parser.add_argument('--batch-size', type=int, default=4)
args = parser.parse_args()
args.device = 'cuda:0'

# --------------- Loading ---------------
def eval():
    dataset_valid = ZipDataset([
        ZipDataset([
            ImagesDataset(os.path.join(args.data_path, 'pha'), mode='L'),
            ImagesDataset(os.path.join(args.data_path, 'fgr'), mode='RGB')
        ], transforms=A.PairCompose([
            A.PairRandomAffineAndResize((1936, 1808), degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.3, 1),
                                        shear=(-5, 5)),
            A.PairApply(T.ToTensor())
        ]), assert_equal_length=True),
        ImagesDataset(os.path.join(args.data_path, 'valid'), mode='RGB', transforms=T.Compose([
            A.RandomAffineAndResize((1936, 1808), degrees=(-5, 5), translate=(0.1, 0.1), scale=(1, 1.2), shear=(-5, 5)),
            T.ToTensor()
        ])),
    ])
    dataloader_valid = DataLoader(dataset_valid,
                                  shuffle=False,
                                  pin_memory=True,
                                  drop_last=True,
                                  batch_size=args.batch_size)

    model = BaseMoE_kmeans(
                args.num_experts,
                args.model_backbone,
                args.model_backbone_scale,
                args.model_refine_mode,
                args.model_refine_sample_pixels,
                args.model_refine_thresholding,
                args.model_refine_kernel_size)
    device = torch.device(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)
    model.eval()
    loss_count = 0
    sad = 0
    mse = 0
    gra = 0
    conn = 0
    with torch.no_grad():
        for i, ((true_pha, true_fgr),true_bgr) in enumerate(dataloader_valid):
            true_src = true_pha * true_fgr + (1 - true_pha) * true_bgr
            pred_pha, pred_fgr, pred_pha_sm, pred_fgr_sm, pred_err_sm, loss_ = model(true_src,true_bgr)
            for k in range (args.batch_size):
                pred_pha_tmp=pred_pha[k][0]
                true_pha_tmp=true_pha[k][0]
                img = true_pha_tmp.cpu().numpy()
                trimap = gen_trimap(img)
                mask_pha = torch.tensor([trimap]).unsqueeze(1)
                sad += BatchSAD(pred_pha_tmp, true_pha_tmp, mask_pha)
                mse += BatchMSE(pred_pha_tmp, true_pha_tmp, mask_pha)
                pred_pha1 = pred_pha_tmp.reshape((pred_pha_tmp.shape[0], -1))
                true_pha1 = true_pha_tmp.reshape((true_pha_tmp.shape[0], -1))
                mask_pha1 = mask_pha.reshape((mask_pha.shape[2], -1))
                gra += gradient(pred_pha1, true_pha1, mask_pha1)
                conn += connectivity_loss(pred_pha1, true_pha1, mask_pha1)
                loss_count += 1
            print(f'output:  SAD: {sad / loss_count}, MSE: {mse / loss_count} , Grad: {gra/ loss_count }, Conn: {conn/ loss_count }')

# --------------- Start ---------------
if __name__ == '__main__':
    eval()