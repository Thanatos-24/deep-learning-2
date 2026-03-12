# import random 
# import torch
# from d2l import torch as d2l

# def synathetic_data(w,b,num_example):
#     x = torch.normal(0,1,(num_example,len(w)))
#     y = torch.matmul(x,w) + b
#     y += torch.normal(0,0.01,y.shape)
#     return x,y.reshape(-1,1)

# true_w = torch.tensor([2,-3.4])
# true_b = 4.2
# features,labels = synathetic_data(true_w,true_b,1000)

# def data_iter(batch_size, features, labels):
#     num_examples = len(features)
#     indices = list(range(num_examples))
#     random.shuffle(indices)
#     for i in range(0, num_examples, batch_size):
#         batch_indices = torch.tensor(indices[i:min(i+batch_size,num_examples)])
#         yield features[batch_indices], labels[batch_indices]
    
# batch_size = 10
# for X,y in data_iter(batch_size, features, labels):
#     print(X, '\n', y)
#     break

# w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
# b = torch.zeros(1, requires_grad=True)

# def linreg(X,w,b):
#     return torch.matmul(X,w)+b

# def squared_loss(y_hat,y):
#     return (y_hat-y.reshape(y_hat.shape))**2/2

# def sgd(params,lr,batch_size):
#     with torch.no_grad():
#          for param in params:
#             param -= lr*param.grad/batch_size
#             param.grad.zero_()

# lr = 0.03
# num_epochs = 3
# net = linreg
# loss = squared_loss

# for epoch in range(num_epochs):
#     for X,y in data_iter(batch_size,features,labels):
#         l = loss(net(X,w,b),y)
#         l.sum().backward()
#         sgd([w,b],lr,batch_size)
#     with torch.no_grad():
#         train_l = loss(net(features,w,b),labels)
#         print(f'epoch {epoch+1}, loss {float(train_l.mean()):f}')
import numpy as np
import torch 
from torch.utils import data
from d2l import torch as d2l
from torch import nn

true_w = torch.tensor([2,-3.4])
true_b = 4.2
features,labels = d2l.synthetic_data(true_w,true_b,1000)

def load_array(dataset,batch_size,is_train=True):
    dataset = data.TensorDataset(*dataset)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)

batch_size = 10
data_iter = load_array((features,labels),batch_size)
net = nn.Sequential(nn.Linear(2,1))
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)
loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(),lr=0.03)
num_epochs = 3
for epoch in range(num_epochs):
    for X,y in data_iter:
        l = loss(net(X),y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features),labels)
    print(f'epoch {epoch+1}, loss {l:f}')