import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence

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

        assert (self.head_dim * num_heads == embed_size), "Embedding size must be divisible by number of heads"

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
        self.layers = nn.ModuleList([PhiDecoderLayer(embed_size, ffn_hidden, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        N, seq_length = x.shape
        positions = self.pos_embedding(torch.arange(seq_length, device=x.device)).unsqueeze(0).repeat(N, 1, 1)
        x = self.dropout(self.token_embedding(x) + positions)

        for layer in self.layers:
            x = layer(x)

        return self.norm(x)

class PhiForCausalLM(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, ffn_hidden, num_heads):
        super().__init__()
        self.phi_model = PhiModel(vocab_size, embed_size, num_layers, ffn_hidden, num_heads)
        self.lm_head = nn.Linear(embed_size, vocab_size)

    def forward(self, x, labels=None):
        x = self.phi_model(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return logits, loss



def data_process(raw_text_iter, vocab, tokenizer):
    """Convert raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

def setup_data_loaders(batch_size=20, seq_len=35):
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, WikiText2(split='train')), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])  # Set <unk> token's index for out-of-vocabulary tokens
    
    def collate_batch(batch):
        batch = pad_sequence(batch, batch_first=True, padding_value=vocab["<pad>"])
        batch = batch[:, :seq_len]  # Truncate sequences to seq_len
        label = batch.clone()
        return batch, label

    train_iter, val_iter, test_iter = WikiText2()
    train_data = data_process(train_iter, vocab, tokenizer)
    val_data = data_process(val_iter, vocab, tokenizer)
    test_data = data_process(test_iter, vocab, tokenizer)

    train_dataset = TensorDataset(train_data, train_data)  # Using input data as labels for simplicity
    val_dataset = TensorDataset(val_data, val_data)
    test_dataset = TensorDataset(test_data, test_data)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_batch)

    return train_dataloader, val_dataloader, test_dataloader, vocab

# Initialize the model and other components for training
batch_size = 20
seq_len = 35
vocab_size = 20000  # This is an approximation, adjust according to the actual vocab size
embed_size = 256
num_layers = 4
ffn_hidden = 512
num_heads = 8

train_dataloader, val_dataloader, test_dataloader, vocab = setup_data_loaders(batch_size=batch_size, seq_len=seq_len)
vocab_size = len(vocab)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PhiForCausalLM(vocab_size, embed_size, num_layers, ffn_hidden, num_heads).to(device)

optimizer = AdamW(model.parameters(), lr=5e-4)

def train_epoch(model, data_loader, optimizer):
    model.train()
    total_loss = 0
    for batch, _ in data_loader:
        inputs = batch.to(device)
        optimizer.zero_grad()
        _, loss = model(inputs, labels=inputs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

# Training loop
epochs = 5
for epoch in range(epochs):
    loss = train_epoch(model, train_dataloader, optimizer)
    print(f"Epoch {epoch+1}, Loss: {loss}")

# Save the model
torch.save(model.state_dict(), "phi_for_causal_lm_wikitext2.pt")

# Hugging Face upload code remains the same (ensure you have your token and have installed the required libraries)

