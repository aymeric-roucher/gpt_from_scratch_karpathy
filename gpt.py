import torch
import torch.nn as nn
from torch.nn import functional as F
import unidecode
import numpy as np

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 100 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
embed_dim=30
n_heads=3
dropout=0.2
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
        out[split] = losses.mean().item()
    model.train()
    return out

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.key_matrix = nn.Linear(embed_dim, head_dim, bias=False)
        self.query_matrix = nn.Linear(embed_dim, head_dim, bias=False)
        self.value_matrix = nn.Linear(embed_dim, head_dim, bias=False)
        self.head_dim = head_dim
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        key = self.key_matrix(x)
        query = self.query_matrix(x)
        value = self.value_matrix(x)

        # self-attention
        product = key @ query.transpose(-2, -1) / np.sqrt(C) # (B, T, T)
        # mask
        product = product.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        weights = F.softmax(product, dim=-1)
        logits = weights @ value
        return logits
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, head_dim, n_heads):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(embed_dim, head_dim) for _ in range(n_heads)])
        self.reproject = nn.Linear(n_heads*head_dim, embed_dim)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.reproject(out)
        return out


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ decoder block """
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        head_dim = embed_dim // n_heads
        self.attention = MultiHeadAttention(embed_dim, head_dim, n_heads)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, X):
        X = X + self.ln1(self.attention(X))
        X = X + self.ln2(self.feed_forward(X))
        return X


class GPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size, n_heads):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding_table = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.Sequential(*[Block(embed_dim, n_heads) for _ in range(3)])
        self.language_model_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, input, targets=None):
        B, T = input.shape
        token_embeddings = self.token_embedding_table(input)
        positional_embeddings = self.position_embedding_table(torch.arange(T)) # Dim: B, T

        X = token_embeddings + positional_embeddings

        X = self.blocks(X)
        logits = self.language_model_head(X)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # there is one more dimension because the logits still have to be chosen from from C options
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, n_tokens):
        for _ in range(n_tokens):
            logits, _ = self(idx[:, -block_size:]) # (B, T)
            # take the probabilities at the last timestep
            logits = logits[:, -1, :]
            # logits are just the probabilities, let us softmax on the last dimension (the different logits) and sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
model = GPT(vocab_size, embed_dim=embed_dim, block_size=block_size, n_heads=n_heads)
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