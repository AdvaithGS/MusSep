import torch
import torch.nn as nn
from .band_split import BandSplit
from .roformer import InterIntraRoFormerBlock

class BSRoFormer(nn.Module):
    def __init__(self, channels=2, hidden_dim=128, num_heads=8, depth=12, bands=None):
        super().__init__()
        self.band_split = BandSplit(channels=channels, hidden_dim=hidden_dim, bands=bands)
        
        # Stack of alternating Time -> Band RoFormer blocks
        self.blocks = nn.ModuleList([
            InterIntraRoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads)
            for _ in range(depth)
        ])
        
        # Mask generation: Project hidden dim back to real/imag for each band
        self.mask_projections = nn.ModuleList([
            nn.Linear(hidden_dim, (end - start) * 2 * channels)
            for start, end in self.band_split.bands
        ])

    def forward(self, x):
        """
        x: Complex STFT spectrogram (B, C, F, T)
        """
        B, C, F, T = x.shape
        
        # 1. Split into bands and project to hidden_dim
        # h shape: (B, T, Num_Bands, Hidden_Dim)
        h = self.band_split(x)
        
        # 2. Sequence Modeling
        for block in self.blocks:
            h = block(h)
            
        # 3. Reconstruct Complex Mask
        mask_bands = []
        for i, proj in enumerate(self.mask_projections):
            # Project back to flat feature space
            m = proj(h[:, :, i, :])
            # Reshape back to (B, C, 2, band_bins, T)
            band_bins = self.band_split.bands[i][1] - self.band_split.bands[i][0]
            m = m.view(B, T, C, 2, band_bins).permute(0, 2, 3, 4, 1)
            mask_bands.append(m)
            
        # Concatenate along frequency axis
        mask_ri = torch.cat(mask_bands, dim=3)
        
        # Recombine Real and Imaginary
        mask = torch.complex(mask_ri[:, :, 0, :, :], mask_ri[:, :, 1, :, :])
        
        # Apply mask and return
        return x * mask