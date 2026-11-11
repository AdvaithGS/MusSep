#!/bin/bash

# 1. Initialize Git Repository
git init MusSep
cd MusSep

# 2. Create Directory Structure
mkdir -p configs data models pipeline utils scripts tests checkpoints

# 3. Create Python Module Init Files
touch data/__init__.py
touch models/__init__.py
touch pipeline/__init__.py
touch utils/__init__.py

# 4. Create Core Files based on finalized architecture
touch configs/roformer_base.yaml
touch configs/dit_latent.yaml
touch configs/distillation.yaml

touch data/unified_dataset.py

touch models/band_split.py
touch models/roformer.py
touch models/bs_roformer.py
touch models/dac_vae.py
touch models/dit.py

touch pipeline/train_base.py
touch pipeline/train_generative.py
touch pipeline/train_distill.py
touch pipeline/flow_matching.py
touch pipeline/distillation.py
touch pipeline/infer.py

touch utils/loss.py
touch utils/metrics.py
touch utils/audio.py

touch scripts/preprocess_latents.py
touch scripts/separate.py
touch scripts/evaluate.py

# 5. Output established requirements
cat <<EOT > requirements.txt
torch>=2.1.0
torchaudio>=2.1.0
transformers>=4.35.0
huggingface-hub>=0.19.0
pyyaml>=6.0
numpy>=1.24.0
soundfile>=0.12.1
EOT

# 6. Create standard ignore file
cat <<EOT > .gitignore
__pycache__/
*.pyc
.DS_Store
checkpoints/
*.pt
*.pth
data/**/*.wav
data/**/*.pt
!data/*.py
.env
venv/
EOT

echo "MusSep repository structure initialized successfully."