# pipeline/train_generative.py
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import random
import os

# Import our custom modules
from data.unified_dataset import UnifiedManifestDataset
from models.dit import JointSubStemDiT
from pipeline.flow_matching import MaskedOTFlowMatcher

def train():
    parser = argparse.ArgumentParser(description="Train the Stage 2 Generative DiT")
    parser.add_argument("--config", type=str, default="configs/dit_latent.yaml", help="Path to config file")
    parser.add_argument("--manifest", type=str, required=True, help="Path to the preprocessed latent manifest JSON")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/", help="Directory to save models")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 1. Load Configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    tier = config['model']['tier']
    model_cfg = config['tiers'][tier]
    
    batch_size = config.get('training', {}).get('batch_size', 16)
    epochs = config.get('training', {}).get('epochs', 100)
    lr = float(config.get('training', {}).get('learning_rate', 1e-4))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Hardware Precision Setup
    # Prefer bf16 on Ampere+ architectures; fallback to fp16 with GradScaler otherwise
    use_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = GradScaler(enabled=not use_bf16)
    print(f"Using device: {device} | Autocast dtype: {dtype} | GradScaler enabled: {not use_bf16}")

    # 3. Dataset and DataLoader
    # Expects pre-encoded .pt latents to avoid the VAE bottleneck during training
    dataset = UnifiedManifestDataset(manifest_path=args.manifest)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )

    # 4. Model and Objective
    dit_model = JointSubStemDiT(
        in_channels=config['model']['in_channels'],
        hidden_dim=model_cfg['hidden_dim'],
        num_heads=model_cfg['num_heads'],
        depth=model_cfg['depth'],
        num_stem_classes=10 # Assuming vocab size of 10
    ).to(device)

    flow_matcher = MaskedOTFlowMatcher(sigma_min=1e-5).to(device)
    optimizer = optim.AdamW(dit_model.parameters(), lr=lr, weight_decay=1e-4)

    # 5. Training Loop
    dit_model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        valid_batches = 0

        for batch_idx, (parent_latent, sub_stems, stem_labels, valid_mask) in enumerate(dataloader):
            # Move to device
            parent_latent = parent_latent.to(device, non_blocking=True)
            sub_stems = sub_stems.to(device, non_blocking=True)
            stem_labels = stem_labels.to(device, non_blocking=True)
            valid_mask = valid_mask.to(device, non_blocking=True)

            # --- Classifier-Free Guidance (CFG) Dropout ---
            # Drop labels 10% of the time to train the unconditional path
            if random.random() < 0.10:
                # Assuming 0 is the 'pad' / unconditional token in the vocab
                stem_labels = torch.zeros_like(stem_labels)

            optimizer.zero_grad(set_to_none=True)

            # Mixed Precision Forward Pass
            with torch.autocast(device_type='cuda', dtype=dtype):
                loss = flow_matcher(dit_model, sub_stems, parent_latent, stem_labels, valid_mask)

            # Backward Pass
            scaler.scale(loss).backward()

            # Unscale before clipping to ensure accurate norm calculation
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(dit_model.parameters(), max_norm=1.0)

            # Optimizer Step
            scaler.step(optimizer)
            scaler.update()

            if loss.item() > 0: # Ignores batches where no valid stems existed
                epoch_loss += loss.item()
                valid_batches += 1

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / valid_batches if valid_batches > 0 else 0
        print(f"=== Epoch {epoch+1} Completed | Avg Loss: {avg_loss:.4f} ===")

        # Save Checkpoint
        checkpoint_path = os.path.join(args.checkpoint_dir, f"dit_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': dit_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'config': config
        }, checkpoint_path)

if __name__ == "__main__":
    train()