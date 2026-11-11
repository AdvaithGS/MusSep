import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return torch.cos(emb), torch.sin(emb)

def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RoFormerAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.rope = RotaryPositionEmbedding(self.head_dim)

    def forward(self, x):
        B, S, D = x.shape
        
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        cos, sin = self.rope(S)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # PyTorch native SDPA (automatically selects FlashAttention if conditions are met)
        out = F.scaled_dot_product_attention(q, k, v)
        
        out = out.transpose(1, 2).reshape(B, S, D)
        return self.proj(out)

class InterIntraRoFormerBlock(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=8):
        super().__init__()
        # Time-wise (Intra-band)
        self.intra_norm1 = nn.LayerNorm(hidden_dim)
        self.intra_attn = RoFormerAttention(hidden_dim, num_heads)
        self.intra_norm2 = nn.LayerNorm(hidden_dim)
        self.intra_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Frequency-wise (Inter-band)
        self.inter_norm1 = nn.LayerNorm(hidden_dim)
        self.inter_attn = RoFormerAttention(hidden_dim, num_heads)
        self.inter_norm2 = nn.LayerNorm(hidden_dim)
        self.inter_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x):
        B, T, K, D = x.shape
        
        # 1. Intra-band processing (Time-wise first, as per paper)
        x_intra = x.transpose(1, 2).reshape(B * K, T, D)
        x_intra = x_intra + self.intra_attn(self.intra_norm1(x_intra))
        x_intra = x_intra + self.intra_mlp(self.intra_norm2(x_intra))
        x = x_intra.reshape(B, K, T, D).transpose(1, 2)
        
        # 2. Inter-band processing (Frequency-wise)
        x_inter = x.reshape(B * T, K, D)
        x_inter = x_inter + self.inter_attn(self.inter_norm1(x_inter))
        x_inter = x_inter + self.inter_mlp(self.inter_norm2(x_inter))
        x = x_inter.reshape(B, T, K, D)
        
        return x