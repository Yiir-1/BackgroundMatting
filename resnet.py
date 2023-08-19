"""使用Resnet模型预训练模型+微调"""

import torch.cuda
import torchvision
from torch import nn
import os
from torchvision import transforms
# import matplotlib.pyplot as plt
import torchvision.models as models

from dataset import ImagesDataset_addname
from torch.utils.data import DataLoader
resnet = models.resnet50(pretrained=True).cuda()
from torchvision import transforms as T

dataset_train_bg = ImagesDataset_addname('./Background/train', mode='RGB', transforms=T.Compose([
    T.Resize((512, 512)),
    T.ToTensor()
]))
dataloader_train_bg = DataLoader(dataset_train_bg,
                                 shuffle=False,
                                 batch_size=4,
                                 num_workers=0,
                                 pin_memory=True)

dataset_test_bg = ImagesDataset_addname('./Background/valid', mode='RGB', transforms=T.Compose([
    T.Resize((512, 512)),
    T.ToTensor()
]))
dataloader_test_bg = DataLoader(dataset_train_bg,
                                 shuffle=False,
                                 batch_size=4,
                                 num_workers=0,
                                 pin_memory=True)

for param in resnet.parameters():
    param.requires_grad = False

#只需要将全链接层分类类别数进行更改
in_f= resnet.fc.in_features
#替换掉了全链接层 是可训练
resnet.fc = nn.Linear(in_f, 50)

#优化器只需要优化最后一层
optim =torch .optim.Adam(resnet.fc.parameters(),lr =0.001)
#损失函数
loss_fn = nn.CrossEntropyLoss()
#训练函数fit 必须要指定 model.train，model.eval Resnet中有BN层
def fit(epoch,model,trainloader,testloader):
    correct = 0
    total = 0
    running_loss =0
    model.train()  #指明这是train模式需要bn和drop
    for i, (x,name) in enumerate(dataloader_train_bg):
        if torch.cuda.is_available():
            x,y =x.to('cuda'),y.to('cuda')
        y_pred =model(x)
        loss = loss_fn(y_pred,y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred,dim=1)
            correct +=(y_pred==y).sum().item()
            total += y.size(0)
            running_loss += loss.item()
    epoch_loss = running_loss/len(trainloader.dataset)
    epoch_acc =correct/total

    test_correct = 0
    test_total = 0
    test_running_loss =0

    model.eval()
    with torch.no_grad():
        for x,y in testloader:
            if torch.cuda.is_available():
                x,y = x.to('cuda'),y.to('cuda')
            y_pred =model(x)
            loss = loss_fn(y_pred,y)
            y_pred = torch.argmax(y_pred,dim=1)
            test_correct +=(y_pred==y).sum().item()
            test_total +=y.size(0)
            test_running_loss +=loss.item()
    epoch_tst_loss =test_running_loss/len(testloader.dataset)
    epoch_tst_acc = test_correct/test_total
    return epoch_loss ,epoch_acc,epoch_tst_loss,epoch_tst_acc

# #微调
for param in resnet.parameters():
    param.requires_grad=True

extend_epoch =8
#微调的时候学习速率要更小一些
optimizer = torch.optim.Adam(resnet.parameters(),lr=0.00001)

# #训练过程
train_loss =[]
train_acc =[]
test_loss =[]
test_acc=[]

for epoch in range(extend_epoch):
    epoch_loss,epoch_acc,epoch_tst_loss,epoch_tst_acc =fit(epoch,resnet,train_dl,test_dl)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(epoch_tst_loss)
    test_acc.append(epoch_tst_acc)