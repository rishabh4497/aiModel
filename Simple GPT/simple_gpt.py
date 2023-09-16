!pip install sentencepiece
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

"""**Load the data**"""

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(text[:500])

"""**Tokenize the data**"""

# chars = sorted(list(set(text)))
# vocab_size = len(chars)
# stoi = { ch:i for i,ch in enumerate(chars) }
# itos = { i:ch for i,ch in enumerate(chars) }
# encode = lambda s: [stoi[c] for c in s]
# decode = lambda l: ''.join([itos[i] for i in l])

# print(decode(encode("Hello world")))
spm.SentencePieceTrainer.train(input='input.txt', model_prefix='m', vocab_size=652)
sp = spm.SentencePieceProcessor()
sp.load('m.model')

"""**Setting up the parameters**"""

params = {
    'batch_size': 64,
    'block_size': 512,
    'max_iters': 2000,
    'eval_interval': 500,
    'learning_rate': 3e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'vocab_size': sp.get_piece_size(),
    'n_embd': 384,
    'n_head': 6,
    'n_layer': 6
}

"""**Data Preparation**"""

data = torch.tensor(sp.encode_as_ids(text), dtype=torch.long)

def fetch_data_batch(data, params):
    ix = torch.randint(len(data) - params['block_size'], (params['batch_size'],))
    x = torch.stack([data[i:i + params['block_size']] for i in ix])
    y = torch.stack([data[i + 1:i + params['block_size'] + 1] for i in ix])
    return x.to(params['device']), y.to(params['device'])

"""**Model Architecture**"""

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(params['n_embd'], params['n_head'])
        self.feed_forward = nn.Sequential(
            nn.Linear(params['n_embd'], 4 * params['n_embd']),
            nn.ReLU(),
            nn.Linear(4 * params['n_embd'], params['n_embd'])
        )
        self.norm1 = nn.LayerNorm(params['n_embd'])
        self.norm2 = nn.LayerNorm(params['n_embd'])

    def forward(self, x):
        x = x + self.attn(x, x, x)[0]
        x = self.norm1(x)
        return x + self.feed_forward(self.norm2(x))

class Simple_GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(params['vocab_size'], params['n_embd'])
        self.pos_emb = nn.Embedding(params['block_size'], params['n_embd'])
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(params['n_layer'])])
        self.norm = nn.LayerNorm(params['n_embd'])
        self.head = nn.Linear(params['n_embd'], params['vocab_size'])

    def forward(self, x, y=None):
      pos_emb = self.pos_emb(torch.arange(x.size(1), device=params['device']))
      x = self.tok_emb(x) + pos_emb.unsqueeze(0)
      for block in self.blocks:
          x = block(x)
      logits = self.head(self.norm(x))
      loss = F.cross_entropy(logits.view(-1, params['vocab_size']), y.view(-1)) if y is not None else None
      return logits, loss

"""**Training**"""

device = params['device']
model = Simple_GPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'])

for i in range(params['max_iters']):
    xb, yb = fetch_data_batch(data, params)
    _, loss = model(xb, yb)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if i % params['eval_interval'] == 0:
        print(f"Iter {i}: Loss {loss.item():.4f}")

"""**Model Evaluation**"""

def generate_text(seed, model, sp, params, length=100):
    model.eval()
    with torch.no_grad():
        ids = torch.tensor(sp.encode_as_ids(seed)).unsqueeze(0).to(params['device'])
        for _ in range(length):
            logits, _ = model(ids)
            next_token = torch.argmax(logits[0, -1]).unsqueeze(0).unsqueeze(0)
            ids = torch.cat([ids, next_token], dim=-1)
    return sp.decode_ids(ids[0].tolist())

print(f"Generated text:\n{generate_text('Harry is', model, sp, params, length=100)}")
