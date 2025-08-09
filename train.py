import torch
from transformer import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

src_tokenize = Tokenizer(max_length=100)
tgt_tokenize = Tokenizer(max_length=100)

src_sentences = ['i am fun', 'blue salads are fantastic']
tgt_sentences = ['yo soy divertido', 'las ensaladas azules son fantasticos']

src_tokenize.fit(src_sentences)
tgt_tokenize.fit(tgt_sentences)

model = Transformer(seq_len=100, 
                    embed_dim=512, 
                    src_vocab_size=src_tokenize.vocab_size, 
                    tgt_vocab_size=tgt_tokenize.vocab_size,
                    padding_idx=src_tokenize.padding_idx).to(device=device)

src = src_tokenize.tokenize('i am fun').to(device=device)
tgt = tgt_tokenize.tokenize('yo soy divertido').to(device=device)

output = model(src, tgt)
print(output.shape)


