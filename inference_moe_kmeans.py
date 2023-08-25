"""
Inference images: Extract matting on images.
Example:
    python inference_moe_kmeans.py \
        --model-backbone resnet50 \
        --model-backbone-scale 0.25 \
        --model-refine-mode sampling \
        --model-refine-sample-pixels 80000 \
        --output-type com fgr pha
"""

import argparse

import numpy as np
import torch
import os
import shutil

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms.functional import to_pil_image
from threading import Thread
from tqdm import tqdm
from model import MoE_kmeans
from dataset import ImagesDataset, ZipDataset,ImagesDataset_addname,ZipDataset_withname
from dataset import augmentation as A
from model import MattingBase, MattingRefine
from inference_utils import HomographicAlignment
import os
from data_path import DATA_PATH
os.environ['CUDA_VISIBLE_DEVICES']='0'
# --------------- Arguments ---------------


parser = argparse.ArgumentParser(description='Inference images')

parser.add_argument('--model-backbone', type=str, required=True, choices=['resnet101', 'resnet50', 'mobilenetv2'])
parser.add_argument('--model-backbone-scale', type=float, default=0.25)
parser.add_argument('--model-checkpoint', type=str,default='checkpoint/mattingrefine_resnet50_basemoe_kmeans/epoch-0-iter-55999-loss0.009505371563136578-model.pth')
parser.add_argument('--model-refine-mode', type=str, default='sampling', choices=['full', 'sampling', 'thresholding'])
parser.add_argument('--model-refine-sample-pixels', type=int, default=80_000)
parser.add_argument('--model-refine-kernel-size', type=int, default=3)
parser.add_argument('--model-refine-thresholding', type=float, default=0.7)
parser.add_argument('--images-src', type=str, default='evaldata\\img')
parser.add_argument('--images-bgr', type=str, default='evaldata\\bgr')

parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
parser.add_argument('--num-workers', type=int, default=0,
                    help='number of worker threads used in DataLoader. Note that Windows need to use single thread (0).')
parser.add_argument('--preprocess-alignment', action='store_true')

parser.add_argument('--output-dir', type=str, default='./output')
parser.add_argument('--output-types', type=str, required=True, nargs='+', choices=['com', 'pha', 'fgr', 'err', 'ref'])
parser.add_argument('-y', action='store_true')
parser.add_argument('--num-experts',type=int, default=3)
args = parser.parse_args()

# --------------- Main ---------------


device = torch.device(args.device)

# Load model
model = MoE_kmeans(args.num_experts,
                args.model_backbone,
                args.model_backbone_scale,
                args.model_refine_mode,
                args.model_refine_sample_pixels,
                args.model_refine_thresholding,
                args.model_refine_kernel_size)
model = model.to(device).eval()
model.load_state_dict(torch.load(args.model_checkpoint, map_location=device), strict=False)
dataset = ZipDataset_withname([
    ImagesDataset(args.images_src),
    ImagesDataset_addname(args.images_bgr),
], assert_equal_length=True, transforms=A.PairCompose([
    HomographicAlignment() if args.preprocess_alignment else A.PairApply(nn.Identity()),
    A.PairApply(T.ToTensor())
]))
dataloader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, pin_memory=True)

# Create output directory
if os.path.exists(args.output_dir):
    if args.y or input(f'Directory {args.output_dir} already exists. Override? [Y/N]: ').lower() == 'y':
        shutil.rmtree(args.output_dir)
    else:
        exit()

for output_type in args.output_types:
    os.makedirs(os.path.join(args.output_dir, output_type))

os.makedirs(os.path.join(args.output_dir, 'src'))
os.makedirs(os.path.join(args.output_dir, 'bgr'))
# Worker function
def writer(img, path):
    img = to_pil_image(img[0].cpu())
    img.save(path)

with torch.no_grad():
    for i, ((src,bgr), names) in enumerate(tqdm(dataloader)):
        pathname = dataset.datasets[0].filenames[i]
        pathname = os.path.relpath(pathname, args.images_src)
        pathname = os.path.splitext(pathname)[0]
        src = src.to(device, non_blocking=True)
        bgr = bgr.to(device, non_blocking=True)
        name=names[0].split("\\")[2]
        pha, fgr, _, _, err, ref= model(src,bgr,name)
        Thread(target=writer, args=(src, os.path.join(args.output_dir, 'src', pathname + '.png'))).start()
        Thread(target=writer, args=(bgr, os.path.join(args.output_dir, 'bgr', pathname + '.png'))).start()


        if 'com' in args.output_types:
            com = torch.cat([fgr * pha.ne(0), pha], dim=1)
            Thread(target=writer, args=(com, os.path.join(args.output_dir, 'com', pathname + '.png'))).start()
        if 'pha' in args.output_types:
            Thread(target=writer, args=(pha, os.path.join(args.output_dir, 'pha', pathname + '.jpg'))).start()
        if 'fgr' in args.output_types:
            Thread(target=writer, args=(fgr, os.path.join(args.output_dir, 'fgr', pathname + '.jpg'))).start()
        if 'err' in args.output_types:
            err = F.interpolate(err, src.shape[2:], mode='bilinear', align_corners=False)
            Thread(target=writer, args=(err, os.path.join(args.output_dir, 'err', pathname + '.jpg'))).start()
        if 'ref' in args.output_types:
            ref = F.interpolate(ref, src.shape[2:], mode='nearest')
            Thread(target=writer, args=(ref, os.path.join(args.output_dir, 'ref', pathname + '.jpg'))).start()