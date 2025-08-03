import torch
import torch.nn as nn
import math

class TextEmbedder(nn.Module):
    def __init__(self, embed_length = 512, max_length = 100):  
        super().__init__()
        self.embed_length = embed_length
        self.max_length = max_length
        self.pad_token = '<pad>'
        self.unknown_token = '<unc>'
        self.vocab = dict()
        self.reverse_vocab = dict() 

        self.embedding = None


    def fit(self, sentences: list[str]): 
        token_set = set()
        for s in sentences: 
            tokens = s.lower().split()
            token_set.update(tokens)

        token_set.update([self.unknown_token, self.pad_token])
        vocab = {tok: i for tok, i in enumerate(token_set)}
        self.vocab_size = len(self.vocab)
        self.embedding = nn.Embedding(self.vocab_size, self.embed_length)

    def embed(self, sentence):
        tokens = sentence.lower().split()
        token_ids = [self.vocab.get(token, self.vocab[self.unkown_token]) for token in tokens] 
        return torch.tensor(token_ids).unsqueeze(0)
    
    def positional_encoding(max_len, embed_len):
        pe = torch.zeros(max_len, embed_len)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_len, 2).float() * (-math.log(10000.0) / embed_len))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def embed_and_encode(self, sentence):
        t1 = self.embed(sentence)
        t2 = self.positional_encoding(self.max_length, self.embed_length)
        return t1 + t2

class MultiHeadAttention(nn.Module):
    def __init__(self, head_dim, heads):
        pass

class Encoder(nn.Module):
    pass

class Decoder(nn.Module):
    pass 

class Transformer(nn.Module):
    pass