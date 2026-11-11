# utils/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiResolutionComplexSTFTLoss(nn.Module):
    def __init__(self, window_sizes=[4096, 2048, 1024, 512, 256], hop_length=147):
        super().__init__()
        self.window_sizes = window_sizes
        self.hop_length = hop_length

    def forward(self, pred_wave, target_wave):
        """
        pred_wave, target_wave: (Batch, Channels, Time)
        Must be computed in float32 to avoid NaN gradients.
        """
        pred = pred_wave.float()
        target = target_wave.float()
        
        # 1. Waveform L1 Loss
        loss_wave = F.l1_loss(pred, target)
        
        # 2. Multi-Resolution Complex STFT L1 Loss
        loss_stft = 0.0
        for n_fft in self.window_sizes:
            window = torch.hann_window(n_fft, device=pred.device, dtype=torch.float32)
            
            # Predict STFT
            pred_stft = torch.stft(
                pred.view(-1, pred.shape[-1]), 
                n_fft=n_fft, hop_length=self.hop_length, 
                window=window, return_complex=True
            )
            # Target STFT
            target_stft = torch.stft(
                target.view(-1, target.shape[-1]), 
                n_fft=n_fft, hop_length=self.hop_length, 
                window=window, return_complex=True
            )
            
            # L1 on Real and Imaginary parts
            loss_real = F.l1_loss(pred_stft.real, target_stft.real)
            loss_imag = F.l1_loss(pred_stft.imag, target_stft.imag)
            
            loss_stft += (loss_real + loss_imag)
            
        return loss_wave + loss_stft