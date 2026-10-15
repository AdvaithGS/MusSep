import torch
import torch.nn as nn

class BandSplit(nn.Module):
    """
    Splits the complex spectrogram into sub-bands and projects them into a common hidden dimension.
    """
    def __init__(self, sample_rate=44100, n_fft=2048, hidden_dim=128):
        super().__init__()
        self.n_fft = n_fft
        self.hidden_dim = hidden_dim
        
        # Standard contiguous sub-bands definition (start_bin, end_bin)
        # In practice, these are tuned to specific musical frequencies.
        self.bands = [
            (0, 16), (16, 32), (32, 64), (64, 128), 
            (128, 256), (256, 512), (512, n_fft // 2 + 1)
        ]
        self.num_bands = len(self.bands)
        
        # Linear projections for each band
        self.projections = nn.ModuleList([
            nn.Linear((end - start) * 2, hidden_dim) # *2 for real and imaginary parts
            for start, end in self.bands
        ])
        
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        Args:
            x: Complex spectrogram tensor of shape (Batch, Channels, Freq, Time)
               or real/imag concatenated along channels.
               Assuming concatenated for this implementation: (B, C*2, F, T)
        Returns:
            Projected bands: (Batch, Time, Num_Bands, Hidden_Dim)
        """
        B, C, F, T = x.shape
        
        band_outputs = []
        for i, (start, end) in enumerate(self.bands):
            # Extract band: (B, C, band_freqs, T)
            band = x[:, :, start:end, :]
            
            # Reshape for linear layer: (B, T, C * band_freqs)
            band = band.permute(0, 3, 1, 2).contiguous().view(B, T, -1)
            
            # Project to hidden_dim
            projected = self.projections[i](band)
            band_outputs.append(projected)
            
        # Stack bands: (B, T, num_bands, hidden_dim)
        out = torch.stack(band_outputs, dim=2)
        out = self.layer_norm(out)
        return out
