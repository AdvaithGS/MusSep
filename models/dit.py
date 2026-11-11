import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# (Assume RotaryPositionEmbedding and apply_rotary_pos_emb are imported from models.roformer)
from .roformer import RotaryPositionEmbedding, apply_rotary_pos_emb

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    def forward(self, t):
        half = self.mlp[0].in_features // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        return self.mlp(emb)

class JointAxialDiTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Time-wise (Intra-stem)
        self.time_qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)
        self.time_rope = RotaryPositionEmbedding(self.head_dim)
        
        # Stem-wise (Inter-stem)
        self.stem_qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.stem_proj = nn.Linear(hidden_dim, hidden_dim)
        # Note: No RoPE for stem_qkv. Stem identity is symmetric aside from class labels.
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # AdaLN Modulation: scales/shifts for Time Attn, Stem Attn, and MLP
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 9 * hidden_dim, bias=True)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False)

    def forward(self, x, t_emb):
        """
        x: (Batch, Num_Stems, Time, Hidden)
        t_emb: (Batch, Hidden)
        """
        B, S, T, D = x.shape
        
        # Chunk AdaLN parameters
        params = self.adaLN_modulation(t_emb).view(B, 1, 1, 9, D)
        shift_time, scale_time, gate_time, shift_stem, scale_stem, gate_stem, shift_mlp, scale_mlp, gate_mlp = params.unbind(3)
        
        # --- 1. Time-wise Attention (with RoPE) ---
        x_time = self.norm1(x) * (1 + scale_time) + shift_time
        x_time = x_time.reshape(B * S, T, D)
        
        qkv_time = self.time_qkv(x_time).reshape(B * S, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q_t, k_t, v_t = qkv_time[0], qkv_time[1], qkv_time[2]
        
        cos, sin = self.time_rope(T)
        cos, sin = cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)
        q_t, k_t = apply_rotary_pos_emb(q_t, k_t, cos, sin)
        
        attn_time = F.scaled_dot_product_attention(q_t, k_t, v_t)
        attn_time = self.time_proj(attn_time.transpose(1, 2).reshape(B * S, T, D))
        attn_time = attn_time.view(B, S, T, D)
        x = x + gate_time * attn_time
        
        # --- 2. Stem-wise Attention (NO RoPE) ---
        x_stem = self.norm2(x) * (1 + scale_stem) + shift_stem
        x_stem = x_stem.transpose(1, 2).reshape(B * T, S, D)
        
        qkv_stem = self.stem_qkv(x_stem).reshape(B * T, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q_s, k_s, v_s = qkv_stem[0], qkv_stem[1], qkv_stem[2]
        
        attn_stem = F.scaled_dot_product_attention(q_s, k_s, v_s)
        attn_stem = self.stem_proj(attn_stem.transpose(1, 2).reshape(B * T, S, D))
        attn_stem = attn_stem.view(B, T, S, D).transpose(1, 2)
        x = x + gate_stem * attn_stem
        
        # --- 3. MLP ---
        x_mlp = self.norm3(x) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.mlp(x_mlp)
        
        return x

class JointSubStemDiT(nn.Module):
    def __init__(self, in_channels=128, hidden_dim=512, num_heads=8, depth=8, num_stem_classes=10):
        super().__init__()
        self.in_proj = nn.Linear(in_channels * 2, hidden_dim) # Updated
        
        # Class embeddings injected per slot to break permutation symmetry
        self.stem_embedding = nn.Embedding(num_stem_classes, hidden_dim)
        self.time_embedder = TimestepEmbedder(hidden_dim)
        
        self.blocks = nn.ModuleList([
            JointAxialDiTBlock(hidden_dim, num_heads) for _ in range(depth)
        ])
        
        # Final projection back to DAC-VAE latent dimension
        self.out_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.out_proj = nn.Linear(hidden_dim, in_channels)
def forward(self, x, t, stem_labels, parent_latent):
        """
        x: Noisy target latents (Batch, Num_Stems, In_Channels, Time)
        t: Flow matching timestep (Batch,)
        stem_labels: Integer IDs for each slot (Batch, Num_Stems)
        parent_latent: The mixed stem from Stage 1 (Batch, In_Channels, Time)
        """
        B, S, C, T_seq = x.shape
        
        # 1. Condition on parent mixture
        # Broadcast parent_latent to all stem slots: (B, S, C, T_seq)
        parent_expanded = parent_latent.unsqueeze(1).expand(B, S, C, T_seq)
        
        # Concatenate noisy latents and conditioning parent along channels -> (B, S, 2C, T_seq)
        x_cond = torch.cat([x, parent_expanded], dim=2)
        
        # Permute for transformer: (B, S, T_seq, 2C)
        x_cond = x_cond.permute(0, 1, 3, 2) 
        
        # 2. Embed inputs
        # Project 2C down to hidden_dim. Output shape: (B, S, T_seq, Hidden)
        x_hidden = self.in_proj(x_cond)
        
        # 3. Inject slot identities to break permutation symmetry
        # Shape: (B, S, 1, Hidden)
        stem_emb = self.stem_embedding(stem_labels).unsqueeze(2)
        x_hidden = x_hidden + stem_emb
        
        # 4. Embed timestep
        # Shape: (B, Hidden)
        t_emb = self.time_embedder(t)
        
        # 5. Axial Transformer Blocks
        for block in self.blocks:
            x_hidden = block(x_hidden, t_emb)
            
        # 6. Project out
        # Revert hidden_dim back to original In_Channels (C)
        x_hidden = self.out_norm(x_hidden)
        x_out = self.out_proj(x_hidden)
        
        # Permute back to latent shape: (B, S, In_Channels, Time)
        return x_out.permute(0, 1, 3, 2)