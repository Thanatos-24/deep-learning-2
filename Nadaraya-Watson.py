import torch
from torch import nn
from d2l import torch as d2l

n_train = 50
x_train, _ = torch.sort(torch.rand(n_train) * 5)

def f(x):
    return 2 * torch.sin(x) + x**0.8

y_train = f(x_train) + torch.normal(0,0.5,(n_train,))
x_test = torch.arange(0,5,0.1)
y_true = f(x_test)
n_test = len(x_test)

def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_true, y_hat], 'x', 'y', legend=['Truth', 'Pred'], xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5)
    d2l.plt.show()

# X_repeat = x_test.repeat_interleave(n_train).reshape((-1,n_train))
# attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)
# y_hat = torch.matmul(attention_weights,y_train)
# plot_kernel_reg(y_hat)

class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(-((queries - keys)*self.w)**2 / 2, dim=1)
        return torch.bmm(self.attention_weights.unsqueeze(1), values.unsqueeze(-1)).reshape(-1)

x_tile = x_train.repeat((n_train,1))
y_tile = y_train.repeat((n_train,1))
keys = x_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train,-1))
values = y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train,-1))

model = NWKernelRegression()
loss_fn = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(model.parameters(), lr=0.5)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])
for epoch in range(5):
    trainer.zero_grad()
    l = loss_fn(model(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.mean().detach()):f}')


keys = x_train.repeat((n_test,1))
values = y_train.repeat((n_test,1))
y_hat = model(x_test, keys, values).detach()
plot_kernel_reg(y_hat)
