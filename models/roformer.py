import torch
import torch.nn as nn
import math

class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) for sequence modeling.
    """
    def __init__(self, dim, max_seq_len=4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return torch.cos(emb), torch.sin(emb)

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k shape: (B, num_heads, seq_len, head_dim)
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
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply RoPE
        cos, sin = self.rope(S)
        cos = cos.unsqueeze(0).unsqueeze(0) # (1, 1, S, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, S, D)
        
        return self.proj(out)

class InterIntraRoFormerBlock(nn.Module):
    """
    Dual-path RoFormer block handling inter-band and intra-band sequences sequentially.
    """
    def __init__(self, hidden_dim=128, num_heads=8):
        super().__init__()
        # Intra-band (Time-wise sequence modeling)
        self.intra_attn = RoFormerAttention(hidden_dim, num_heads)
        self.intra_norm = nn.LayerNorm(hidden_dim)
        self.intra_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Inter-band (Frequency-wise sequence modeling)
        self.inter_attn = RoFormerAttention(hidden_dim, num_heads)
        self.inter_norm = nn.LayerNorm(hidden_dim)
        self.inter_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x):
        # x shape: (B, Time, Bands, Hidden)
        B, T, K, D = x.shape
        
        # 1. Intra-band processing (over Time, per Band)
        x_intra = x.transpose(1, 2).reshape(B * K, T, D)
        attn_out = self.intra_attn(self.intra_norm(x_intra))
        x_intra = x_intra + attn_out
        x_intra = x_intra + self.intra_mlp(self.intra_norm(x_intra))
        x = x_intra.reshape(B, K, T, D).transpose(1, 2)
        
        # 2. Inter-band processing (over Bands, per Time-step)
        x_inter = x.reshape(B * T, K, D)
        attn_out = self.inter_attn(self.inter_norm(x_inter))
        x_inter = x_inter + attn_out
        x_inter = x_inter + self.inter_mlp(self.inter_norm(x_inter))
        x = x_inter.reshape(B, T, K, D)
        
        return x
