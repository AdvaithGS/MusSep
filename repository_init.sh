#!/usr/bin/env bash
# Sets up a Python environment for MusSep and checks it.
#
#   ./repository_init.sh            # creates .venv, installs requirements, runs the tests
#   ./repository_init.sh --cpu      # CPU-only PyTorch wheels (no CUDA download)
#   ./repository_init.sh --no-vae   # skip the DAC-VAE package (Stage 1 only, or training on existing latents)
#
# Conda users: `conda env create -f environment.yml` instead.
set -euo pipefail
cd "$(dirname "$0")"

CPU=0; VAE=1
for arg in "$@"; do
  case "$arg" in
    --cpu) CPU=1 ;;
    --no-vae) VAE=0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip

if [ "$CPU" = 1 ]; then
  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
fi
if [ "$VAE" = 1 ]; then
  pip install -r requirements.txt
else
  grep -v '^dacvae' requirements.txt > .venv/requirements.novae.txt
  pip install -r .venv/requirements.novae.txt
fi

python -m pytest -q tests
echo
echo "Ready. Activate with: . .venv/bin/activate"
