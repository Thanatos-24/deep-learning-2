import torch
from torch import nn
from d2l import torch as d2l

n_train,n_test,num_inputs,batch_size = 20,100,200,5
num_epochs = 100
true_w,true_b = torch.ones((num_inputs,1))*0.01,0.05
train_data = d2l.synthetic_data(true_w,true_b,n_train)
train_iter = d2l.load_array(train_data,batch_size)
test_data = d2l.synthetic_data(true_w,true_b,n_test)
test_iter = d2l.load_array(test_data,batch_size,is_train=False)

def init_params():
    w = torch.normal(0,1,size=(num_inputs,1),requires_grad=True)
    b = torch.zeros(1,requires_grad=True)
    return [w,b]

def l2_penalty(w):
    return torch.sum(w**2)/2

def train(lambd):
    w,b = init_params()
    net,loss = lambda X: X@w+b, nn.MSELoss()
    params = init_params()
    trainer = torch.optim.SGD(params,lr=0.003)
    animator = d2l.Animator(xlabel='epochs',ylabel='loss',yscale='log',xlim=[5,num_epochs])
    for epoch in range(num_epochs):
        for X,y in train_iter:
            l = loss(net(X),y) + lambd*l2_penalty(w)
            l.backward()
            trainer.step()
        if (epoch+1) % 5 == 0:
            animator.add(epoch+1,float(l))