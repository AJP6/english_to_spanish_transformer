import torch 
import os
from torch.utils.data import Dataset 
from transformer import *

class SentenceData(Dataset): 
    def __init__(self, src_path, tgt_path, eng_tokenizer = None, spa_tokenizer = None): 
        # we dont want the test data to learn vocabulary --> integer mapping, only evaluate it
        self.eng_tokenizer = eng_tokenizer or Tokenizer(max_length=30)
        self.spa_tokenizer = spa_tokenizer or Tokenizer(max_length=30)
        
        with open(src_path) as eng: 
            english = [line.strip() for line in eng]
        with open(tgt_path) as spa: 
            spanish = [line.strip() for line in spa]

        english = ['<bos> ' + sentence + ' <eos>' for sentence in english]
        spanish = ['<bos> ' + sentence + ' <eos>' for sentence in spanish]

        if eng_tokenizer is None:
            self.eng_tokenizer.fit(english)
        if spa_tokenizer is None:
            self.spa_tokenizer.fit(spanish)

        self.english = [self.eng_tokenizer.tokenize(sentence) for sentence in english]
        self.spanish = [self.spa_tokenizer.tokenize(sentence) for sentence in spanish]
    
    def __len__(self): 
        return len(self.english)

    def __getitem__(self, idx): 
        return self.english[idx], self.spanish[idx]
        