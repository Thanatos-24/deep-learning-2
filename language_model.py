import torch
import random
from d2l import torch as d2l

tokens = d2l.tokenize(d2l.read_time_machine())
print(tokens[0])
corpus = [token for line in tokens for token in line]