import torch
from torch.utils.data import Dataset
import torchaudio
import json
import random

class UnifiedManifestDataset(Dataset):
    """
    Dataset loader reading from an offline-generated JSON manifest 
    that unifies MoisesDB and Slakh2100 taxonomies.
    """
    def __init__(self, manifest_path, segment_length=10, sample_rate=44100):
        with open(manifest_path, 'r') as f:
            self.tracks = json.load(f)
            
        self.segment_samples = segment_length * sample_rate
        self.sample_rate = sample_rate
        self.max_stems = 6
        
        # Calculate dataset weights to down-weight synthesized Slakh data
        moises_count = sum(1 for t in self.tracks if t["source"] == "moises")
        slakh_count = sum(1 for t in self.tracks if t["source"] == "slakh")
        
        if slakh_count > 0 and moises_count > 0:
            self.weights = [
                1.0 if t["source"] == "moises" else (moises_count / slakh_count) * 0.5 
                for t in self.tracks
            ]
        else:
            self.weights = [1.0 for _ in self.tracks]

    def __len__(self):
        return len(self.tracks)

    def _load_and_pad(self, file_path):
        audio, sr = torchaudio.load(file_path)
        
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1) # Force stereo
            
        # Random crop or pad
        if audio.shape[1] > self.segment_samples:
            start = random.randint(0, audio.shape[1] - self.segment_samples)
            audio = audio[:, start:start + self.segment_samples]
        else:
            pad_amount = self.segment_samples - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, pad_amount))
            
        return audio

    def __getitem__(self, idx):
        track = self.tracks[idx]
        
        # 1. Load the parent mixture (conditioning stem from Stage 1)
        parent_audio = self._load_and_pad(track["parent_mixture"])
        
        # 2. Initialize containers for the joint sub-stem set
        audio_tensor = torch.zeros(self.max_stems, 2, self.segment_samples)
        stem_labels = torch.zeros(self.max_stems, dtype=torch.long)
        valid_mask = torch.zeros(self.max_stems, dtype=torch.bool)
        
        # 3. Load available sub-stems
        num_loaded = min(len(track["sub_stems"]), self.max_stems)
        
        for i in range(num_loaded):
            stem_info = track["sub_stems"][i]
            audio_tensor[i] = self._load_and_pad(stem_info["path"])
            stem_labels[i] = stem_info["class_id"]
            valid_mask[i] = True
                
        return parent_audio, audio_tensor, stem_labels, valid_mask