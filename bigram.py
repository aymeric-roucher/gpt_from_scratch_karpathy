import torch
import torch.nn as nn
from torch.nn import functional as F
import unidecode

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 1000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('recherche.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    text = unidecode.unidecode(text)

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
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


class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, input, targets=None):
        logits = self.token_embedding_table(input)
        B, T, C = logits.shape

        if targets is None:
            loss = None
        else:
            logits = logits.view(B*T, C) # there is one more dimension because the logits still have to be chosen from from C options
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, n_tokens):
        for _ in range(n_tokens):
            logits, _ = self(idx[:, -block_size:]) # (B, T)
            print(idx.shape, logits.shape)
            # take the probabilities at the last timestep
            logits = logits[:, -1, :]
            # logits are just the probabilities, let us softmax and sample from them
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
model = BigramModel(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter%100 == 0: # periodically show evaluation results
        out = estimate_loss(model)
        print(iter, out)
    
    # sample batch
    xb, yb = get_batch('train')
    logits, loss = model(xb, targets=yb)
    optimizer.zero_grad()
    # retropropagate loss
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long)
print(decode(model.generate(context, n_tokens=500)[0].tolist()))