import os
import torch
from d2l import torch as d2l

d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip','94646ad1522d915e7b0f9296181140edcf86a4f5')

def read_data_nmt():
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir,'fra.txt'),'r', encoding='utf-8') as f:
        return f.read()

raw_text = read_data_nmt()
# print(raw_text[:75])

def preprocess_nmt(text):
    def no_space(char,prev_char):
        return char in set(',.!?') and prev_char != ' '

    text = text.replace('\u202f',' ').replace('\xa0',' ').lower()
    out = [
        ' ' + char if i > 0 and no_space(char,text[i-1]) else char
        for i,char in enumerate(text)
    ]
    return ''.join(out)

text = preprocess_nmt(raw_text)

def tokenize_nmt(text,num_examples=None):
    source, target = [], []
    for i,line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

source, target = tokenize_nmt(text)

src_vocab = d2l.Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])

def truncate_pad(line,num_steps,padding_token):
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))

def build_array_nmt(lines,vocab,num_steps):
    lines = [vocab[line] for line in lines]
    lines = [line + [vocab['<eos>']] for line in lines]
    array = torch.tensor([truncate_pad(line,num_steps,vocab['<pad>']) for line in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len


def load_data_nmt(batch_size,num_steps,num_examples=600):
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text,num_examples)
    src_vocab = d2l.Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source,src_vocab,num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target,tgt_vocab,num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays,batch_size)
    return data_iter, src_vocab, tgt_vocab