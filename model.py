"""GPT-style decoder-only transformer, built from scratch (no pretrained
models, no Hugging Face model classes — just torch.nn primitives).

This is the exact architecture from this project's pretraining
(03_pretrain.ipynb) and fine-tuning (04_finetune.ipynb) notebooks, pulled
out into a standalone module so both notebooks import one definition
instead of each carrying its own copy (they originally did — this file
removes that duplication without changing the architecture itself).

Two small adaptations from the original notebook code, both behavior-
preserving (same math, same parameter count, same trained weights are
still loadable):
  - GPTLanguageModel takes `vocab_size` as a constructor argument instead
    of reading it from a notebook-global variable.
  - forward() derives its device from the input tensor (`idx.device`)
    instead of reading a notebook-global `device` variable.

Hyperparameters below are the exact values used for the trained checkpoint
referenced throughout this repo (33.5M parameters total).
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

n_embd = 512      # size of each token's embedding vector
n_head = 8        # attention heads per block
n_layer = 8       # transformer blocks stacked
block_size = 256  # context length (tokens attended over at once)
dropout = 0.2      # dropout probability throughout


class Head(nn.Module):
    """One self-attention head: for each token, decides what to look for
    (query), what it offers (key), and what it passes on (value), then
    blends other tokens' values weighted by query-key similarity — with a
    causal mask so a token can only attend to itself and earlier tokens."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """Several attention heads run in parallel, each free to learn a
    different kind of relationship between tokens; their outputs are
    concatenated and projected back to n_embd."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Per-token MLP: expand to 4x width, ReLU, project back down. This is
    where the model does most of its per-token "thinking", after attention
    has let tokens exchange information."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """One transformer block: attention ("talk" — exchange information
    across tokens), then feed-forward ("think" — process per token),
    each wrapped in a residual connection + pre-layer-norm."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """Token + learned positional embeddings, n_layer stacked Blocks, a
    final layer norm, then a linear head projecting back to vocab_size
    logits. Trained with next-token cross-entropy — nothing else."""

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
