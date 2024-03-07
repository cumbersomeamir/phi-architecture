import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW

class PhiRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        self.dim = dim
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim)).to("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, x):
        seq_len = x.size(1)  # Changed to index 1 to reflect sequence length in the batch
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
        qkv = self.qkv(x).reshape(N, seq_length, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.matmul(q, k.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-1e20"))
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(N, seq_length, self.embed_size)
        return self.fc_out(out)

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
        self.attention = PhiAttention(embed_size, num_heads)
        self.norm2 = nn.LayerNorm(embed_size)
        self.mlp = PhiMLP(embed_size, ffn_hidden)
    def forward(self, x):
        x = self.attention(self.norm1(x)) + x
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

def generate_synthetic_data(num_samples=10000, sequence_length=50, vocab_size=51200):
    input_ids = torch.randint(0, vocab_size, (num_samples, sequence_length))
    labels = input_ids.clone()
    return input_ids, labels

# Training setup
vocab_size = 51200
embed_size = 2560
num_layers = 32
ffn_hidden = 10240
num_heads = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PhiForCausalLM(vocab_size, embed_size, num_layers, ffn_hidden, num_heads).to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

# Generate synthetic data
input_ids, labels = generate_synthetic_data()
input_ids, labels = input_ids.to(device), labels.to(device)
dataset = TensorDataset(input_ids, labels)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Training loop with limited batches
model.train()
max_batches = 10  # Limit the number of batches per epoch
for epoch in range(1):
    for i, batch in enumerate(loader):
        if i >= max_batches: break  # Stop after processing max_batches
        inputs, labels = batch
        optimizer.zero_grad()
        _, loss = model(inputs, labels=labels)
        loss.backward()
        optimizer.step()
        print(f"Batch {i}, Loss: {loss.item()}")

# Save the model
torch.save(model.state_dict(), "phi_for_causal_lm_state_dict.pt")

# Hugging Face Hub upload (example)
from huggingface_hub import HfApi, HfFolder

HfFolder.save_token("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")  # Save your token

api.create_repo(token="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", name="phi-arc", organization="Amirkid", repo_type="model")

# Upload the model
api.upload_file(
    token="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",
    repo_id="Amirkid/phi-arc",  # Your Hugging Face username and model name
    path_or_fileobj="phi_for_causal_lm_state_dict.pt",  # Path to the model file
    path_in_repo="phi_for_causal_lm_state_dict.pt",  # Path in the repository (can be the same as the filename)
)
