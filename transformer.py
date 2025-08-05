import torch
import torch.nn as nn
import math

#https://github.com/shubhamprasad0/transformer-from-scratch/tree/main/src
#https://www.youtube.com/watch?v=U0s0f995w14 (min 35)

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
        # q, k, v shave hape: [batch_size, num_heads, seq_len, head_dim]
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
    def __init__(self): 
        super.__init__()

    def generate_auto_regressive_mask(): 
        pass

    def forward(self): 
        pass

def main():
    embedded_text = TextEmbedder.embed_and_encode(sentence="gigityhghghg this is a test")
    print("embedded shape:", embedded_text.shape)
    attention_text = MultiHeadAttention.forward(x=embedded_text)
    print("attention output shape:", attention_text.shape)

if __name__ == '__main__':
    main()