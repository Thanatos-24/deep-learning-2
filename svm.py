import torch
from torch import nn
from d2l import torch as d2l
from matplotlib import pyplot as plt
from softmax import train_ch3

batch_size = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)
num_inputs,num_outputs,num_hidden = 784,10,256

W1 = nn.Parameter(torch.randn(num_inputs,num_hidden,requires_grad=True))
b1 = nn.Parameter(torch.zeros(num_hidden,requires_grad=True))
W2 = nn.Parameter(torch.randn(num_hidden,num_outputs,requires_grad=True))
b2 = nn.Parameter(torch.zeros(num_outputs,requires_grad=True))

params = [W1,b1,W2,b2]

def relu(x):
    a = torch.zeros_like(x)
    return torch.max(a,x)

def net(X):
    X = X.reshape((-1,num_inputs))
    H = relu(X@W1+b1)
    return H@W2+b2

loss = nn.CrossEntropyLoss()
num_epochs,lr = 10,0.1
updater = torch.optim.SGD(params,lr=lr)
train_ch3(net,train_iter,test_iter,loss,num_epochs,updater)
plt.show()

