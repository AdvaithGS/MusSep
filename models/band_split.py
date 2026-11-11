import torch
import torch.nn as nn

def get_bs_roformer_bands():
    """
    Generates the 62-band scheme from the original BS-RoFormer paper.
    Calculated for sample_rate=44100, n_fft=2048 (1025 bins).
    """
    bands = []
    # 2 bins/band < 1 kHz (approx up to bin 46)
    bands.extend([(i, i+2) for i in range(1, 47, 2)]) 
    # 4 bins/band up to 2 kHz (approx up to bin 93)
    bands.extend([(i, i+4) for i in range(47, 95, 4)])
    # 12 bins/band up to 4 kHz (approx up to bin 186)
    bands.extend([(i, i+12) for i in range(95, 191, 12)])
    # 24 bins/band up to 8 kHz (approx up to bin 372)
    bands.extend([(i, i+24) for i in range(191, 383, 24)])
    # 48 bins/band up to 16 kHz (approx up to bin 743)
    bands.extend([(i, i+48) for i in range(383, 767, 48)])
    # Remaining bins split into 2 bands
    bands.append((767, 896))
    bands.append((896, 1025))
    return bands

class BandSplit(nn.Module):
    def __init__(self, channels=2, hidden_dim=128, bands=None):
        super().__init__()
        self.bands = bands if bands is not None else get_bs_roformer_bands()
        self.num_bands = len(self.bands)
        
        # Distinct LayerNorm and Linear projection per band
        self.norms = nn.ModuleList([
            nn.LayerNorm((end - start) * 2 * channels) 
            for start, end in self.bands
        ])
        self.projections = nn.ModuleList([
            nn.Linear((end - start) * 2 * channels, hidden_dim)
            for start, end in self.bands
        ])

    def forward(self, x):
        """
        x: Complex STFT tensor (B, C, F, T)
        """
        B, C, F, T = x.shape
        
        # Separate real and imaginary, then concatenate along a new dimension
        # Shape becomes (B, C, 2, F, T)
        x_real = x.real
        x_imag = x.imag
        x_ri = torch.stack([x_real, x_imag], dim=2)
        
        band_outputs = []
        for i, (start, end) in enumerate(self.bands):
            # Extract band: (B, C, 2, band_bins, T)
            band = x_ri[:, :, :, start:end, :]
            
            # Flatten channels, real/imag, and bins into one feature dimension
            # Shape: (B, T, C * 2 * band_bins)
            band = band.permute(0, 4, 1, 2, 3).contiguous().view(B, T, -1)
            
            # Apply per-band norm and projection
            projected = self.projections[i](self.norms[i](band))
            band_outputs.append(projected)
            
        # Output shape: (B, T, Num_Bands, Hidden_Dim)
        return torch.stack(band_outputs, dim=2)