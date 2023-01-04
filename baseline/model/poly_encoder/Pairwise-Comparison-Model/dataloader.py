from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk
from typing import List

import numpy as np
import torch
import warnings
import random

warnings.filterwarnings(action='ignore')

PAD_TOK = '[PAD]'
CLS_TOK = '[CLS]'
SEP_TOK = '[SEP]'

class IRData(Dataset):
    def __init__(self, data_path, tokenizer, cand_size, max_len=512, model_type='poly'):
        self._data = load_from_disk(data_path)
        self.cls = CLS_TOK
        self.sep = SEP_TOK
        self.pad = PAD_TOK
        self.max_len = max_len
        self.cand_size = cand_size
        self.tokenizer = tokenizer
        self.model_type = model_type
    def __len__(self):
        return len(self._data)
    
    def _tokenize(self, sent):
        tokens = self.tokenizer(
            sent,truncation= True,
            max_length = 512,
            padding = 'longest'
        )
        input_ids = tokens['input_ids']
        return input_ids

    def __getitem__(self, idx):
        # in_batch_negative
        # title add
        turn = self._data[idx]
        query = turn['question'] #batch , seq
        title = turn['title']
        context = '@' + title + ':'+turn['context']
        # cands = [i['context'] for i in self._data if i['title'] != title][:self.cand_size] #batch, cand_size , seq_len
        cands = random.sample(['@' + i['title'] + ':' + i['context'] for i in self._data if i['title'] != title],self.cand_size) #batch, cand_size , seq_len
        label = [1] + [0 for _ in range(self.cand_size)]
        if self.model_type =='cross':
            pair_seq = list(map(lambda x: self._tokenize(query + self.sep + x), [context] + cands))
            return(pair_seq, label)
        else:
            query_ids = self._tokenize(query)
            cands = [self._tokenize(cand) for cand in [context] + cands]
            # return {'query_ids' : query_ids, 'cands' : cands, 'label' : label}
            return (query_ids, cands,label)
            # return {'query_ids' : torch.LongTensor(query_ids), 'cands' : torch.LongTensor(cands), 'label' : torch.LongTensor(label)}
