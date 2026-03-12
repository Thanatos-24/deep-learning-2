import torch
from torch import nn
from d2l import torch as d2l


def resnet18(num_classes, in_channels=1):
    def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(d2l.Conv2D(in_channels, out_channels, kernel_size=3, padding=1))
            else:
                blk.append(d2l.Conv2D(out_channels, out_channels, kernel_size=3, padding=1))
        return nn.Sequential(*blk)
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        resnet_block(64, 64, 2, first_block=True),
        resnet_block(64, 128, 2),
        resnet_block(128, 256, 2),
        resnet_block(256, 512, 2),
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(512, num_classes)
    )
    return net

def train(net,num_gpus,batch_size,lr):
    train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.normal_(m.weight,std=0.01)

    net.apply(init_weights)
    net = nn.DataParallel(net,device_ids=devices)
    trainer = torch.optim.SGD(net.parameters(),lr=lr)
    loss = nn.CrossEntropyLoss()
    timer = d2l.Timer()
    num_epochs = 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1,num_epochs])
    for epoch in range(num_epochs):
        timer.start()
        for X,y in train_iter:
            trainer.zero_grad()
            X,y = X.to(devices[0]),y.to(devices[0])
            l = loss(net(X),y)
            l.backward()
            trainer.step()
        timer.stop()
        animator.add(epoch+1, (d2l.evaluate_accuracy_gpu(net,test_iter,devices),timer.avg()))
      