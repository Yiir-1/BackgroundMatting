import argparse

from data_path import DATA_PATH
from model import MoE
parser = argparse.ArgumentParser()

parser.add_argument('--dataset-name', type=str, required=True, choices=DATA_PATH.keys())

parser.add_argument('--model-backbone', type=str, required=True, choices=['resnet101', 'resnet50', 'mobilenetv2'])
parser.add_argument('--model-backbone-scale', type=float, default=0.25)
parser.add_argument('--model-refine-mode', type=str, default='sampling', choices=['full', 'sampling', 'thresholding'])
parser.add_argument('--model-refine-sample-pixels', type=int, default=80_000)
parser.add_argument('--model-refine-thresholding', type=float, default=0.7)
parser.add_argument('--model-refine-kernel-size', type=int, default=3, choices=[1, 3])
parser.add_argument('--model-name', type=str, required=True)
parser.add_argument('--model-last-checkpoint', type=str, default=None)

parser.add_argument('--batch-size', type=int, default=2)
parser.add_argument('--num-workers', type=int, default=0)
parser.add_argument('--epoch-start', type=int, default=0)
parser.add_argument('--epoch-end', type=int, required=True)

parser.add_argument('--log-train-loss-interval', type=int, default=10)
parser.add_argument('--log-train-images-interval', type=int, default=1000)
parser.add_argument('--log-valid-interval', type=int, default=2000)

parser.add_argument('--checkpoint-interval', type=int, default=2000)
parser.add_argument('--num-experts', type=int, required=True, default=4)
args = parser.parse_args()

model = MoE(3*1936*1808,
                args.num_experts,
                args.model_backbone,
                args.model_backbone_scale,
                args.model_refine_mode,
                args.model_refine_sample_pixels,
                args.model_refine_thresholding,
                args.model_refine_kernel_size).cuda()
modelDict = {name for name, param in model.named_parameters()}
print(type(modelDict))
for i in modelDict:
    print(i)