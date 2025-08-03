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

        pe = self.positional_encoding(max_length, embed_length)
        self.register_buffer('pos_encoding', pe)  # shape: [1, max_len, embed_len]
        # we were recalculating pos encodings with each run, which is bad
        # register-buffer wont be updated with grad desc because pos encodings are fixed not learned

    def fit(self, sentences: list[str]): 
        token_set = set()
        for s in sentences: 
            tokens = s.lower().split()
            token_set.update(tokens)

        token_set.update([self.unknown_token, self.pad_token])
        self.vocab = {tok: i for i, tok in enumerate(sorted(token_set))} # sorted this cause otherwise we'd get different token mapping each time
        self.reverse_vocab = {i: tok for tok, i in self.vocab.items()} # maps integer ID back to token
        self.vocab_size = len(self.vocab)
        self.embedding = nn.Embedding(self.vocab_size, self.embed_length)

    def embed(self, sentence):
        tokens = sentence.lower().split()
        token_ids = [self.vocab.get(token, self.vocab[self.unknown_token]) for token in tokens] 

        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]  # truncates it so it fits
        else:
            # pad with pad_token id
            pad_id = self.vocab[self.pad_token]
            token_ids.extend([pad_id] * (self.max_length - len(token_ids)))

        return torch.tensor(token_ids).unsqueeze(0) # shape here is [1, seq_len]
    
    def positional_encoding(self, max_len, embed_len):
        pe = torch.zeros(max_len, embed_len)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_len, 2).float() * (-math.log(10000.0) / embed_len))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0) # shape here is [1, max_len, embed_len]
    
    def embed_and_encode(self, sentence):
        if self.embedding is None:
            raise ValueError("model needs to be fitted with training data before you can embed, so gotta call fit() first.")
        
        tokens_in_tensor = self.embed(sentence) # shape: [1 seq_len]
        embedded = self.embedding(tokens_in_tensor) # shape: [1, seq_len, embed_dim]
        pos_encoded = self.pos_encoding[:, :embedded.size(1), :] # shape: [1, seq_len, embed_dim]
        return embedded + pos_encoded

class MultiHeadAttention(nn.Module):
    def __init__(self, head_dim, heads):
        pass

class Encoder(nn.Module):
    pass

class Decoder(nn.Module):
    pass 

class Transformer(nn.Module):
    pass