import torch
import os
import torchvision
from torch import nn
from d2l import torch as d2l
from matplotlib import pyplot as plt

d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL+'hotdog.zip','fba480ffa8aa7e0febbb511d181409f899b9baa5')
data_dir = d2l.download_extract('hotdog')

train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir,'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir,'test'))

hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[i][0] for i in range(8,16)]
d2l.show_images(hotdogs+not_hotdogs,2,8,scale=1.4)
plt.show()

normalize = torchvision.transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
train_augs = torchvision.transforms.Compose([
    torchvision.trainsforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize
])
test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize
])

finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features,2)
nn.init.xavier_uniform_(finetune_net.fc.weight)

def train_fine_tuining(net,learning_rate,batch_size=128,num_epochs=100,param_group=True):
    train_iter = torchvision.datasets.ImageFolder(os.path.join(data_dir,'train'),transform=train_augs,batch_size=batch_size,shuffle=True)
    test_iter = torchvision.datasets.ImageFolder(os.path.join(data_dir,'test'),transform=test_augs,batch_size=batch_size)
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction='none')
    if param_group:
        params_1x = [param for name,param in net.named_parameters() if name not in ['fc.weight','fc.bias']]
        trainer = torch.optim.SGD([{'params':params_1x},{'params':net.fc.parameters(),'lr':learning_rate*10}],lr=learning_rate,weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(),lr=learning_rate,weight_decay=0.001)
    d2l.train_ch13(net,train_iter,test_iter,loss,trainer,num_epochs,devices)