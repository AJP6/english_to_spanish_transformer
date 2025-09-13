import torch
import torch.nn as nn
from transformer import *
from data import *
from torch.utils.data import DataLoader, Dataset

EPOCHS = 60
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAIN_PATH_SPA = 'spanish_train.txt'
TRAIN_PATH_ENG = 'english_train.txt'
TEST_PATH_SPA = 'spanish_test.txt'
TEST_PATH_ENG = 'english_test.txt'

def main():  
    train_data = SentenceData(TRAIN_PATH_ENG, TRAIN_PATH_SPA)
    train_loader = DataLoader(train_data, 
                              batch_size=32, 
                              shuffle=True)
    
    test_data = SentenceData(TEST_PATH_ENG, TEST_PATH_SPA, 
                             eng_tokenizer=train_data.eng_tokenizer, 
                             spa_tokenizer=train_data.spa_tokenizer)
    test_loader = DataLoader(test_data,
                             batch_size=32,
                             shuffle=False)
    print("Dataloaders Made")

    ai_model = Transformer(seq_len=30, 
                           embed_dim=512,
                           src_vocab_size=train_data.eng_tokenizer.vocab_size,
                           tgt_vocab_size=train_data.spa_tokenizer.vocab_size,
                           padding_idx=train_data.spa_tokenizer.vocab['<pad>']).to(DEVICE)
    
    optimization = torch.optim.Adam(ai_model.parameters(), lr=1e-4)
    criteria = nn.CrossEntropyLoss(ignore_index=train_data.spa_tokenizer.vocab['<pad>'])

    best_test_loss = float('inf')

    for epoch in range(EPOCHS):
        ai_model.train()
        total_loss = 0
        for src_batch, tgt_batch in train_loader:
            src_batch = src_batch.to(DEVICE) # source batch shape; [batch size, src_seq_len]
            tgt_batch = tgt_batch.to(DEVICE) # target batch shape; [batch size, tgt_seq_len]

            tgt_input = tgt_batch[:, :-1]
            tgt_target = tgt_batch[:, 1:]

            optimization.zero_grad() # zeros-out gradients stored in .grad for all the parameters
            output = ai_model(src_batch, tgt_input)  # shape: [batch, seq_len, vocab_size]

            # flatten output and target for loss
            loss = criteria(output.reshape(-1, output.size(-1)), tgt_target.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ai_model.parameters(), 1.0)
            optimization.step() # updates weights based on .grad values
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Train loss {avg_train_loss:.4f}")

        ai_model.eval() # disables dropout
        test_loss = 0
        with torch.no_grad(): # saves memory by not computing gradients
            for src_batch, tgt_batch in test_loader:
                src_batch = src_batch.to(DEVICE)
                tgt_batch = tgt_batch.to(DEVICE)

                tgt_input = tgt_batch[:, :-1]
                tgt_target = tgt_batch[:, 1:]

                out = ai_model(src_batch, tgt_input)
                loss = criteria(out.reshape(-1, out.size(-1)),
                                 tgt_target.reshape(-1))
                test_loss += loss.item()

        test_loss /= len(test_loader)
        print(f"Val loss {test_loss:.4f}")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(ai_model.state_dict(), "best_transformer.pt")
            print("saved best model")

if __name__ == "__main__": 
    main()


# finish training loop
# figure out which loss function to use
# translate.py for inference use