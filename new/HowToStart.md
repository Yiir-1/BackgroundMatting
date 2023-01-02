### inference运行方式：

```shell
python inference_video.py \
--model-checkpoint "./pretrain/pytorch_resnet50.pth" \
--video-src "./images/dlh.mp4" \
--video-bgr "./images/dlh.png" \
--output-dir "./output/" \
--output-type fgr
```


### train-base 运行方式：
```shell
CUDA_VISIBLE_DEVICES=0 python train_base.py \
        --dataset-name videomatte240k \
        --model-backbone resnet50 \
        --model-name mattingbase-resnet50-videomatte240k \
        --model-pretrain-initialization "pretrain/best_deeplabv3_resnet50_voc_os16.pth" \
        --epoch-end 8
```

### Train-refine 运行方式:

```shell
CUDA_VISIBLE_DEVICES=0,1 python train_refine.py \
        --dataset-name videomatte240k \
        --model-backbone resnet50 \
        --model-name mattingrefine-resnet50-videomatte240k \
        --model-last-checkpoint "./checkpoint/mattingbase-resnet50-videomatte240k/epoch-7.pth" \
        --epoch-end 1
        
```


### speed 测试
```shell
python inference_speed_test.py \
        --model-type mattingrefine \
        --model-backbone resnet50 \
        --model-backbone-scale 0.25 \
        --model-refine-mode sampling \
        --model-refine-sample-pixels 80000 \
        --batch-size 1 \
        --resolution 1920 1080 \
        --backend pytorch \
        --precision float32\
        --model-checkpoint  "./checkpoint/mattingrefine-resnet50-videomatte240k/epoch-0.pth" 
        
```