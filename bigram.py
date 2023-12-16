import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 10000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ------------

## load data
with open('data/train.txt', 'r') as f:
        text = f.read()
    
chars = sorted(list(set(text)))
vocab_size = len(chars)

print("Vocab size:", vocab_size)
print("".join(chars))


## Tokenizer (encode and decode)
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

print("Encode:", encode("hello"))
print("Decode:", decode([75, 72, 79, 79, 82]))


## Dataset (train/test split)
data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:100])
n = int(len(data)*0.9)
train_data = data[:n]
val_data = data[n:]


 # data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
    
class Bigram(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, vocab_size)
        print("Embedding: ", self.embedding_table.weight.shape)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        logits = self.embedding_table(idx) # (B,T,C)
        # print("Logits: ", logits.shape)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
            # print("Logits: ", logits.shape) # torch.Size([256, 98])
            # print("Targets: ", targets.shape) # torch.Size([256])

        return logits, loss
    
    def generate(self, idx: torch.Tensor, n: int):
        print("Generate")
        # generate n tokens based on the context x
        for _ in range(n):
            logits, _ = self.forward(idx, None)
            # print("Logits: ", logits.shape)
            logits = logits[:, -1, :] # (B,1,C)
            # print("Logits: ", logits.shape)

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # print(idx.shape)

            idx = torch.cat((idx, idx_next), dim=1) # (B,T+1)
        
        return idx
    


def main():

    ## Model
    model = Bigram(vocab_size)
    x, y = get_batch('train')
    # x, y = torch.from_numpy(np.array([[75],[72]])), torch.from_numpy(np.array([[72],[79]]))
    print(x.shape, y.shape)

    logits, loss = model(x, y)
    print("Loss:", loss) # expect -ln(1/98) = 4.5849


    idx = torch.zeros((1, 1), dtype=torch.long)
    idx[0, 0] = stoi['`']
    print(decode(model.generate(idx, 100)[0].tolist()))



    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for i in range(max_iters):
        x, y = get_batch('train')
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % eval_interval == 0:
            # every once in a while evaluate the loss on train and val sets
            losses = estimate_loss(model)
            print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            # x, y = get_batch('val')
            # logits, loss = model(x, y)
            # print(f"Step {i}: val loss = {loss:.4f}")
    

    idx = torch.zeros((1, 1), dtype=torch.long)
    idx[0, 0] = stoi['`']
    print(decode(model.generate(idx, 1000)[0].tolist()))


if __name__ == "__main__":
    main()