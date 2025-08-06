import torch
import torch.nn as nn
import math

#https://github.com/shubhamprasad0/transformer-from-scratch/tree/main/src
#https://www.youtube.com/watch?v=U0s0f995w14 (min 35)


class Tokenizer():
    def __init__(self, max_length=100):  
        self.max_length = max_length
        self.pad_token = '<pad>'
        self.unknown_token = '<unk>'
        self.vocab = dict()
        self.reverse_vocab = dict() 

    def fit(self, sentences): 
        token_set = set()
        for s in sentences: 
            tokens = s.lower().split()
            token_set.update(tokens)

        token_set.update([self.unknown_token, self.pad_token])
        self.vocab = {tok: i for i, tok in enumerate(sorted(token_set))} # sorted this cause otherwise we'd get different token mapping each time
        self.reverse_vocab = {i: tok for tok, i in self.vocab.items()} # maps integer ID back to token
        self.vocab_size = len(self.vocab)
        self.padding_idx = self.vocab['<pad>']
        self.embedding = nn.Embedding(self.vocab_size, self.embed_length, padding_idx=self.padding_idx)

    def tokenize(self, sentence):
        tokens = sentence.lower().split()
        token_ids = [self.vocab.get(token, self.vocab[self.unknown_token]) for token in tokens] 

        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]  # truncates it so it fits
        else:
            # pad with pad_token id
            pad_id = self.vocab[self.pad_token]
            token_ids.extend([pad_id] * (self.max_length - len(token_ids)))

        return torch.tensor(token_ids).unsqueeze(0) # shape here is [1, seq_len]

class PositionalEncoding(nn.Module): 
    def __init__(self, embed_len, max_seq_len): 
        super().__init__()

        pe_mat = torch.zeros(max_seq_len, embed_len)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_len, 2).float() * (-math.log(10000.0) / embed_len))

        pe_mat[:, 0::2] = torch.sin(position * div_term)
        pe_mat[:, 1::2] = torch.cos(position * div_term)

        return self.register_buffer('pe_mat', pe_mat.unsqueeze(0))
    
    def forward(self, x): 
        return x + self.pe_mat[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0

        self.query_linear = nn.Linear(embed_dim, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)

        self.final_linear = nn.Linear(embed_dim, embed_dim)
        self.scale_factor = math.sqrt(self.head_dim)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        # q, k, v shape have: [batch_size, num_heads, seq_len, head_dim]
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale_factor
        # now shape: [batch_size, num_heads, seq_len, seq_len]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) # fills masked positions with attention = -1e9 ≈ 0

        attention_weights = nn.functional.softmax(scores, dim=-1) # softmax across the last dimension (across key positions)

        return torch.matmul(attention_weights, v) # outputted shape: [batch_size, num_heads, seq_len, head_dim]

    def forward(self, q, k, v, mask=None): # inputted shape: [batch_size, seq_len, embed_dim]
        batch_size, seq_len, embed_dim = k.shape

        q = self.query_linear(q)
        k = self.key_linear(k)
        v = self.value_linear(v)

        # splits the data into heads
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # shape after transpose is: [batch_size, num_heads, seq_len, head_dim]


        attention_output = self.scaled_dot_product_attention(q, k, v, mask)

        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim) # concatenates heads back together
        # shape is now back to: [batch_size, seq_len, embed_dim = num_heads * head_dim]
        
        return self.final_linear(attention_output) # learns how to combine info from all heads
    
# dff stands for dimension of feed forward
class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads, d_ff, dropout=0.1): 
        super().__init__()
        self.multi_head_attn = MultiHeadAttention(embed_dim, num_heads) 
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(normalized_shape=embed_dim) 
        # each token is represented by a vector of length embed_dim, so LayerNorm normalizes the values within each token vector
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, d_ff), 
            nn.Relu(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(normalized_shape=embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn = self.multi_head_attn(x, x, x)  # x shape is [batch_size, seq_len, embed_dim]. this uses same imput for q, k, and v
        x = self.norm1(x + self.dropout1(attn))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_out))

        return x 

        
class Decoder(nn.Module):
    def __init__(self, embed_dim, num_heads, d_ff, dropout=0.1): 
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.cross_attn = MultiHeadAttention(embed_dim, num_heads)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, d_ff), 
            nn.Relu(), 
            nn.Dropout(dropout),
            nn.Linear(d_ff, embed_dim),
            nn.Dropout(dropout)
        )

        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, x, enc_out, mask=None): 
        self_attn = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout1(self_attn))

        cross_attn = self.cross_attn(x, enc_out, enc_out, mask=mask)
        x = self.norm2(x + self.dropout2(cross_attn)) # x is what decoder has generated thus far, cross_attn is relevant info from encoder
        
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_out))

        return x

class Transformer(nn.Module):
    def __init__(self, seq_len, embed_dim, src_vocab_size, tgt_vocab_size, padding_idx, dropout=0.01): 
        super().__init__()
        #parameters
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.padding_idx = padding_idx

        #layers 
        self.encoder_embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=padding_idx)
        self.decoder_embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=padding_idx)
        self.positional_encoder = PositionalEncoding(embed_dim, seq_len)

        self.encoder_layers = nn.ModuleList([
            Encoder(embed_dim, 8, embed_dim * 2 ) for _ in range(7)
        ])
        self.decoder_layers = nn.ModuleList([
            Decoder(embed_dim, 8, embed_dim * 2) for _ in range(7)
        ])
        self.dropout = nn.Dropout(dropout)

        # converts final hidden states back to vocabulary probabilities
        # [batch_size, seq_len, embed_dim] → [batch_size, seq_len, tgt_vocab_size]
        self.final_linear_layer = nn.Linear(embed_dim, tgt_vocab_size)

    def generate_masks(self, src, tgt): 
        # src mask hides padding tokens from attention in encoder
        src_mask = (src != self.padding_idx).unsqueeze(1).unsqueeze(2) # [batch_size, 1, 1, src_len] needs to match scores = [batch_size, num_heads, seq_len, seq_len]
        tgt_mask_pad = (tgt != self.padding_idx).unsqueeze(1).unsqueeze(2) # [batch_size, 1, 1, tgt_len]
        tgt_mask_cas = torch.tril(torch.ones(tgt.size(1), tgt.size(1))).bool().unsqueeze(0).unsqueeze(0) # tgt.size(1) gets length of tgt seq
        # we want to create a lower triangular matrix here so that it cant see future tokens
        tgt_mask = tgt_mask_pad & tgt_mask_cas
        # shape: [batch_size, 1, tgt_len, tgt_len]

        return src_mask, tgt_mask 

    def forward(self, src, tgt): 
        src_mask, tgt_mask = self.generate_masks(src, tgt)

        src_embed = self.dropout(self.positional_encoder(self.encoder_embed(src)))
        tgt_embed = self.dropout(self.positional_encoder(self.decoder_embed(tgt)))
        # [batch_size, src_len] -> [batch_size, src_len, embed_dim]

        enc_out = src_embed
        for encoder_layer in self.encoder_layers:
            enc_out = encoder_layer(enc_out) 

        dec_out = tgt_embed
        for decoder_layer in self.decoder_layers:
            dec_out = decoder_layer(dec_out, enc_out, src_mask)

        return self.final_linear_layer(dec_out) # outputs shape: [batch_size, tgt_len, tgt_vocab_size]

def main():
    pass

if __name__ == '__main__':
    main()