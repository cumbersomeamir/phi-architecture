import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW

class PhiRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim)).to("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, x):
        seq_len = x.size(0)
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        sinusoid_inp = torch.einsum("i , j -> i j", t, self.inv_freq)
        return torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)

def apply_rotary_pos_emb(x, sincos):
    sin, cos = sincos[..., :x.size(-1) // 2], sincos[..., x.size(-1) // 2:]
    return torch.cat((x[..., :x.size(-1) // 2] * cos - x[..., x.size(-1) // 2:] * sin, 
                      x[..., :x.size(-1) // 2] * sin + x[..., x.size(-1) // 2:] * cos), dim=-1)

class PhiAttention(nn.Module):
    def __init__(self, embed_size, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.head_dim = embed_size // num_heads

        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size must be divisible by number of heads"

        self.qkv = nn.Linear(embed_size, embed_size * 3)
        self.fc_out = nn.Linear(embed_size, embed_size)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, x, mask=None):
        N, seq_length, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(N, seq_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul(q, k.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-1e20"))
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(N, seq_length, self.embed_size)
        out = self.fc_out(out)
        return out

class PhiMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.act_fn = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act_fn(self.fc1(x)))

class PhiDecoderLayer(nn.Module):
    def __init__(self, embed_size, ffn_hidden, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.attention = PhiAttention(embed_size, num_heads=num_heads)
        self.mlp = PhiMLP(embed_size, ffn_hidden)

    def forward(self, x):
        attn_output = self.attention(self.norm1(x))
        x = attn_output + x
        x = self.mlp(self.norm2(x)) + x
        return x

class PhiModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, ffn_hidden, num_heads):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = PhiRotaryEmbedding(embed_size)
        self.layers = nn.ModuleList([PhiDecoderLayer(embed_size, ffn_hidden, num_heads
