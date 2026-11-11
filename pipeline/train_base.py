import yaml
import torch
from models.bs_roformer import BSRoFormer

# 1. Load config
with open("configs/roformer_base.yaml", "r") as f:
    config = yaml.safe_load(f)

# 2. Instantiate Base Model
model = BSRoFormer(
    channels=config['model']['channels'],
    hidden_dim=config['model']['hidden_dim'],
    num_heads=config['model']['num_heads'],
    depth=config['model']['depth']
)

# 3. Example Forward Pass Pipeline
# audio_chunk shape: (Batch, Channels, Samples)
window = torch.hann_window(config['audio']['n_fft']).to(device)
stft = torch.stft(
    audio_chunk, 
    n_fft=config['audio']['n_fft'], 
    hop_length=config['audio']['hop_length'], 
    window=window, 
    return_complex=True
)

# stft shape: (Batch, Channels, Freq, Time)
separated_stft = model(stft)

# separated_audio shape: (Batch, Channels, Samples)
separated_audio = torch.istft(
    separated_stft, 
    n_fft=config['audio']['n_fft'], 
    hop_length=config['audio']['hop_length'], 
    window=window
)