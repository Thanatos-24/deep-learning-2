import torch
from torch import nn
from d2l import torch as d2l
from pre_process import load_data_time_machine
from RNN import train_ch8
from matplotlib import pyplot as plt
from ez_RNN import RNNModel

batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)

vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
device = d2l.try_gpu()
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
model = RNNModel(lstm_layer, len(vocab))
model = model.to(device)    
num_epochs, lr = 500, 1
train_ch8(model, train_iter, vocab, lr, num_epochs, device)
plt.show()
