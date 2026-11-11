import torch
import torch.nn as nn
import torchaudio.transforms as T
from transformers import AutoModel

class DACVAEWrapper(nn.Module):
    """
    Wraps Meta's facebook/dacvae-watermarked. 
    Handles 44.1kHz Stereo <-> 48kHz Mono conversion internally.
    """
    def __init__(self, target_sr=48000, orig_sr=44100):
        super().__init__()
        # Load the continuous DAC-VAE
        self.vae = AutoModel.from_pretrained("facebook/dacvae-watermarked")
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
            
        self.target_sr = target_sr
        self.orig_sr = orig_sr
        
        # Resamplers
        self.up_sample = T.Resample(orig_freq=orig_sr, new_freq=target_sr)
        self.down_sample = T.Resample(orig_freq=target_sr, new_freq=orig_sr)

    def encode(self, audio):
        """
        audio: (Batch, Channels=2, Samples) at 44.1 kHz
        Returns: (Batch, Channels=2, Latent_Dim=128, Frames) at 25 Hz
        """
        B, C, S = audio.shape
        
        # 1. Resample to 48kHz
        audio_48k = self.up_sample(audio)
        
        # 2. Fold stereo into batch dimension for mono processing
        # Shape: (B * 2, 1, Samples_48k)
        audio_mono = audio_48k.view(B * C, 1, -1)
        
        with torch.no_grad():
            # 3. Encode to continuous latent space
            # latents shape: (B * 2, 128, Frames)
            latents = self.vae.encoder(audio_mono)
            
        # 4. Unfold batch back to stereo channels
        _, D, F = latents.shape
        latents = latents.view(B, C, D, F)
        
        return latents

    def decode(self, latents):
        """
        latents: (Batch, Channels=2, Latent_Dim=128, Frames) at 25 Hz
        Returns: (Batch, Channels=2, Samples) at 44.1 kHz
        """
        B, C, D, F = latents.shape
        
        # 1. Fold stereo into batch
        latents_mono = latents.view(B * C, D, F)
        
        with torch.no_grad():
            # 2. Decode
            audio_mono = self.vae.decoder(latents_mono)
            
        # 3. Unfold and downsample back to 44.1 kHz
        audio_48k = audio_mono.view(B, C, -1)
        audio_44k = self.down_sample(audio_48k)
        
        return audio_44k