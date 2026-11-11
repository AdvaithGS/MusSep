import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os
import random

class MoisesHierarchicalDataset(Dataset):
    """
    Dataset loader for MoisesDB handling hierarchical sub-stems.
    Expects directory structure:
    MoisesDB/
      track_01/
        vocals/
          lead.wav
          harmony.wav
        other/
          guitar.wav
          piano.wav
          synth.wav
    """
    def __init__(self, root_dir, target_parent="other", segment_length=10, sample_rate=44100):
        super().__init__()
        self.root_dir = root_dir
        self.target_parent = target_parent # The parent stem to split (e.g., 'other')
        self.segment_samples = segment_length * sample_rate
        self.sample_rate = sample_rate
        
        # Define the global vocabulary for sub-stems (determines stem_labels)
        # 0 is reserved for padding/empty
        self.sub_stem_vocab = {
            "pad": 0,
            "guitar": 1,
            "piano": 2,
            "synth": 3,
            "strings": 4,
            "brass": 5,
            "percussion": 6
        }
        self.num_classes = len(self.sub_stem_vocab)
        self.max_stems = 6 # Maximum number of sub-stems to model jointly
        
        # Scan dataset
        self.tracks = []
        for track_name in os.listdir(root_dir):
            track_path = os.path.join(root_dir, track_name, target_parent)
            if os.path.exists(track_path):
                sub_stems = [f for f in os.listdir(track_path) if f.endswith('.wav')]
                if len(sub_stems) > 0:
                    self.tracks.append({
                        "path": track_path,
                        "sub_stems": sub_stems
                    })

    def __len__(self):
        return len(self.tracks)

    def _load_and_pad(self, file_path):
        audio, sr = torchaudio.load(file_path)
        # Handle stereo/mono and resampling if necessary (simplified for draft)
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1) # force stereo
            
        # Random crop
        if audio.shape[1] > self.segment_samples:
            start = random.randint(0, audio.shape[1] - self.segment_samples)
            audio = audio[:, start:start + self.segment_samples]
        else:
            pad_amount = self.segment_samples - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, pad_amount))
            
        return audio

    def _map_filename_to_class(self, filename):
        # Simplified mapping logic
        name = filename.lower().replace('.wav', '')
        for key in self.sub_stem_vocab:
            if key in name and key != "pad":
                return self.sub_stem_vocab[key]
        return self.sub_stem_vocab["pad"] # Unknown maps to pad/ignore

    def __getitem__(self, idx):
        track = self.tracks[idx]
        
        # Initialize containers for the joint set
        # Shape: (Max_Stems, Channels=2, Samples)
        audio_tensor = torch.zeros(self.max_stems, 2, self.segment_samples)
        stem_labels = torch.zeros(self.max_stems, dtype=torch.long)
        valid_mask = torch.zeros(self.max_stems, dtype=torch.bool)
        
        # Load available sub-stems up to max_stems
        # If multiple of the same class exist (e.g., two guitars), they should ideally be merged.
        # For v1, we take them up to max_stems.
        num_loaded = min(len(track["sub_stems"]), self.max_stems)

        parent_path = os.path.join(track["path"], f"../{self.target_parent}.wav") 
        if os.path.exists(parent_path):
            parent_audio = self._load_and_pad(parent_path)
        else:
            parent_audio = torch.zeros(2, self.segment_samples)

        for i in range(num_loaded):
            file_name = track["sub_stems"][i]
            file_path = os.path.join(track["path"], file_name)
            
            audio = self._load_and_pad(file_path)
            class_id = self._map_filename_to_class(file_name)
            
            if class_id != self.sub_stem_vocab["pad"]:
                audio_tensor[i] = audio
                stem_labels[i] = class_id
                valid_mask[i] = True
                
        return parent_audio, audio_tensor, stem_labels, valid_mask

