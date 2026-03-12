import collections
import re
import random
from d2l import torch as d2l
import torch

d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL+'timemachine.txt','090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():
    with open(d2l.download('time_machine')) as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+',' ',line.strip().lower()) for line in lines]

def tokenize(lines,token='word'):
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('unknown token type: '+token)

def count_corpus(tokens):
    if len(tokens) == 0 or isinstance(tokens[0],list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

class Vocab:
    def __init__(self,tokens=None,min_freq=0,reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter = count_corpus(tokens)
        self.token_freqs = sorted(counter.items(),key=lambda x:x[1],reverse=True)
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens
        uniq_tokens += [token for token, freq in self.token_freqs if freq >= min_freq and token not in uniq_tokens]
        self.idx_to_token, self.token_to_idx = [], dict()
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self,tokens):
        if not isinstance(tokens,(list,tuple)):
            return self.token_to_idx.get(tokens,self.unk)
        return [self.__getitem__(token) for token in tokens]

def load_corpus_time_machine(max_tokens=-1):
    lines = read_time_machine()
    tokens = tokenize(lines,token = 'char')
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus,vocab

def seq_data_iter_random(corpus,batch_size,num_steps):
    corpus = corpus[random.randint(0,num_steps-1):]
    num_subseqs = (len(corpus)-1)//num_steps
    initial_indices = list(range(0,num_subseqs*num_steps,num_steps))
    random.shuffle(initial_indices)
    def data(pos):
        return corpus[pos:pos+num_steps]
    num_batches = num_subseqs//batch_size
    for i in range(0,batch_size*num_batches,batch_size):
        initial_indices_per_batch = initial_indices[i:i+batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j+1) for j in initial_indices_per_batch]
        yield torch.tensor(X),torch.tensor(Y)

def seq_data_iter_sequential(corpus,batch_size,num_steps):
    offset = random.randint(0,num_steps-1)
    num_tokens = ((len(corpus)-offset-1)//batch_size)*batch_size
    Xs = torch.tensor(corpus[offset:offset+num_tokens])
    Ys = torch.tensor(corpus[offset+1:offset+num_tokens+1])
    Xs,Ys = Xs.reshape(batch_size,-1),Ys.reshape(batch_size,-1)
    num_batches = Xs.shape[1]//num_steps
    for i in range(0,num_steps*num_batches,num_steps):
        X = Xs[:,i:i+num_steps]
        Y = Ys[:,i:i+num_steps]
        yield X,Y

class SeqDataLoader:
    def __init__(self,batch_size,num_steps,use_random_iter,max_tokens=-1):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus,self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size,self.num_steps = batch_size,num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus,self.batch_size,self.num_steps)

def load_data_time_machine(batch_size,num_steps,use_random_iter=False,max_tokens=-1):
    data_iter = SeqDataLoader(batch_size,num_steps,use_random_iter,max_tokens)
    return data_iter,data_iter.vocab

if __name__ == '__main__':
    lines = read_time_machine()
    print(lines[0])
    print(lines[1])
    
    tokens = tokenize(lines)
    print(tokens[0])
    
    vocab = Vocab(tokens)
    # print(list(vocab.token_to_idx.items())[:10])
    for i in [0,10]:
        print('words:',tokens[i])
        print('indices:',vocab[tokens[i]])
    
    corpus,vocab = load_corpus_time_machine()
    print(len(corpus),len(vocab))
    
    for X,Y in seq_data_iter_random(corpus,batch_size=2,num_steps=5):
        print('X:',X,'\nY:',Y)
        break
    
    for X,Y in seq_data_iter_sequential(corpus,batch_size=2,num_steps=5):
        print('X:',X,'\nY:',Y)
        break
    
    train_iter, vocab = load_data_time_machine(batch_size=2,num_steps=5,use_random_iter=True)