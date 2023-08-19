import torch
import torchvision.models as models
from model.autoencoder import Autoencoder
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms as T
from data_path import DATA_PATH
from dataset import ImagesDataset, ZipDataset, VideoDataset, SampleDataset,ImagesDataset_addname
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import pandas as pd
# 加载预训练的 ResNet 模型


resnet = models.resnet50(pretrained=True).cuda()


dataset_train_bg = ImagesDataset_addname('./Background/train', mode='RGB', transforms=T.Compose([
    T.Resize((512, 512)),
    T.ToTensor()
]))
dataloader_train_bg = DataLoader(dataset_train_bg,
                                 shuffle=False,
                                 batch_size=4,
                                 num_workers=0,
                                 pin_memory=True)

def inference(dataloader, model):
    latents = []
    names=[]
    for i, (x,name) in enumerate(dataloader_train_bg):
        x = torch.FloatTensor(x)
        vec = model(x.cuda())
        # import pdb;pdb.set_trace()
        for j in range(0,len(name)):
            names.append(name[j])
        if i == 0:
            latents = vec.cpu().detach().numpy()
        else:
            latents = np.concatenate((latents, vec.cpu().detach().numpy()), axis=0)
    print('Latents Shape:', latents.shape)
    return latents,names


resnet.eval()

# 預測答案
latents,names = inference(dataset_train_bg, model=resnet)
transformer = KernelPCA(n_components=10, kernel='rbf', n_jobs=-1)
kpca = transformer.fit_transform(latents)
df = pd.DataFrame(names)

# Clustering
mbkmeans = MiniBatchKMeans(init='k-means++', n_clusters=3, random_state=28, batch_size=100).fit(kpca)
centers = mbkmeans.cluster_centers_
labels_new = mbkmeans.partial_fit(kpca).labels_
df['name']=names
df['label']=labels_new
df.to_csv('bg_clusters_train.csv')
for i in range(2400):
    temp=df['name'][i]
    temp=temp.split('/')[3]
    df.loc[i,'name']=temp

df.to_csv('bg_clusters_train.csv')