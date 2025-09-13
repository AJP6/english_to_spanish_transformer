import torch
import torch.nn as nn
from transformer import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATH = "best_transformer.pt"
TRAIN_PATH_ENG = "english_train.txt"
TRAIN_PATH_SPA = "spanish_train.txt"

def load_tokenizers():
    from data import SentenceData
    train_data = SentenceData(TRAIN_PATH_ENG, TRAIN_PATH_SPA)
    return train_data.eng_tokenizer, train_data.spa_tokenizer

def make_model(src_vocab_size, tgt_vocab_size, padding_idx, seq_len=30, embed_dim=512):
    model = Transformer(seq_len=seq_len, 
                        embed_dim=embed_dim, 
                        src_vocab_size=src_vocab_size,
                        tgt_vocab_size=tgt_vocab_size,
                        padding_idx=padding_idx).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)) # loads the trained weights
    model.eval()
    return model


def translate(src_sentence, transformer, src_tokenizer, tgt_tokenizer, max_len=30):
    # tokenize the input sentence
    src_tokens = src_tokenizer.tokenize(f"<bos> {src_sentence} <eos>").to(DEVICE)
    tgt_tokens = [tgt_tokenizer.vocab['<bos>']] # greedy decoding will generate one token at a time starting from this

    for _ in range(max_len):
        tgt_tensor = torch.tensor([tgt_tokens], device=DEVICE) # shape: [1, current_seq_len]
        output = transformer(src_tokens, tgt_tensor) # forward pass through the model, returns [batch_size, tgt_seq_len, tgt_vocab_size]
        next_id = output[0, -1].argmax(-1).item() # chooses the token with the highest probability for the next position (greedy decoding)
        if next_id == tgt_tokenizer.vocab['<eos>']:
            break
        tgt_tokens.append(next_id)

    # convert token IDs to words (skip <bos>)
    return " ".join([tgt_tokenizer.reverse_vocab[i] for i in tgt_tokens[1:]])

if __name__ == "__main__":
    # example
    eng_tokenizer, spa_tokenizer = load_tokenizers()
    transformer = make_model(
        src_vocab_size=eng_tokenizer.vocab_size,
        tgt_vocab_size=spa_tokenizer.vocab_size,
        padding_idx=spa_tokenizer.vocab['<pad>']
    )
    
    while True:
        src_sentence = input("Enter English sentence to translate (or 'quit'): ")
        if src_sentence.lower() == 'quit':
            break
        translation = translate(src_sentence, transformer, eng_tokenizer, spa_tokenizer)
        print(f"Spanish translation: {translation}")