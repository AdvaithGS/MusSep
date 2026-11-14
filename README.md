# MusSep
I am currently working on an ensemble approach to MSS(Musical Source Separation), combining several state-of-the-art existing and established architectures to achieve superior separation performance across traditional Vocal, Drum, and Bass (VDB) stems, as well as expanding into second-level hierarchical separation.

The [MDX23 Challenge](https://d3qvx1ggyg4lu1.cloudfront.net/challenges/sound-demixing-challenge-2023/problems/music-demixing-track-mdx-23#timeline) was what started my interest in the problem.

Second-Level Hierarchical implies separation for sub-stems like kick, snare, lead vocals, background vocals, piano, electric guitar and acoustic guitar - these are sub-tracks usually included together as the umbrella terms of "Drums", "Vocals" and "Others".

## Overview
(MSS)Music source separation refers to the process of isolating individual sound sources, or stems, from a mixed audio signal. The primary goal of music source separation is to isolate distinct elements such as vocals, drums, bass, and other instruments from a composite track, enhancing the ability to analyze and work with specific stems.

Existing architectures have failed to deliver uniformly high performance across the fundamental stems defined by the Vocal-Drums-Bass (VDB) convention. While many models aim to separate these VDB stems effectively, few extend their scope to more complex hierarchies of separation beyond the conventional VDB setup. When such a complex separation is attempted, it typically results in significantly reduced performance across the conventional VDB stems.
Also, most of these architectures suffer from a lack of cross-dataset training, which exacerbates the challenges posed by the long tail effect. The long tail effect refers to the phenomenon where certain instruments or stems appear infrequently in training datasets, leading to poorer performance in separating these underrepresented elements.

Considering all of the above, this project focuses on adeptly combining existing architectures to achieve better overall performance across stems. By leveraging multiple architectures, we can extend the scope of separation beyond the traditional VDB setup to include additional stems such as guitar and piano, as well as enter the second hierarchical level of separation for certain stems.

## Background
4-Stem music source separation has been implemented in the past using a variety of architectures, each focusing on different techniques such as mel-spectrograms, raw audio waveforms, unsupervised learning from unlabelled data (particularly in scenarios of low data availability), [recurrent neural networks](https://arxiv.org/abs/1910.06379), specific frequency bands, generative adversarial networks [GANs](https://kkp15.github.io/), transformers, convolutional neural networks [CNNs](https://www.ijert.org/audio-stems-separation-using-deep-learning), multitask learning, and DTT-Net. Including hybrid transformers, rotary positional encoding transformers, sparse compression networks, and adversarial networks, revealed higher performance on specific stems.

## Primary Models Considered

### SCNet (Sparse Compression Network)
[SCNet](https://arxiv.org/abs/2401.13276), proposed by Tong et al, is a frequency-domain network that splits spectrograms into subbands. Its key innovation is a sparse compression technique that applies higher compression ratios to less informative subbands, maintaining fidelity in critical subbands while reducing computational load. This approach enhances separation performance and reduces model complexity, making SCNet suitable for real-time applications.

### Mel-Band RoFormer
The [Mel-Band RoFormer](https://arxiv.org/pdf/2310.01809), uses a mel-scale-based projection scheme to capture perceptual differences in frequencies, particularly in vocals and drums. Its key innovation is the use of overlapping subbands based on the mel scale, mimicking the human auditory system’s preference for higher resolution at lower frequencies. This is an [improved](https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model) implementation.

### Hybrid Transformer Demucs (HT Demucs)
The Hybrid Transformer Demucs [HT Demucs](https://arxiv.org/abs/1909.01174), developed by Rouard et al., combines temporal and spectral U-Nets with Transformer layers to enhance its receptive field. By integrating both time and frequency domains, HT Demucs applies self-attention within and cross-attention across them, capturing short- and long-term dependencies for complex musical mixtures. 

### Band-Split RNN for MSS
Band-Split RNN [BSRNN](https://arxiv.org/abs/2209.15174), is a frequency-domain MSS model that explicitly divides the input spectrogram into subbands based on predefined bandwidths. The model operates in the high-sample-rate frequency domain, allowing it to handle super wide-band signals efficiently. It leverages both sequence-level and band-level processing via RNNs to capture intra-band dependencies.

### Wave-U-Net: End-to-End Separation in the Time Domain
[Wave-U-Net](https://arxiv.org/abs/1806.03185), is a time-domain adaptation of the U-Net architecture designed for audio source separation. Unlike spectrogram-based methods, Wave-U-Net operates directly on waveforms, enabling the model to process both magnitude and phase information without relying on fixed transformations. It uses multi-scale feature maps to capture long-term temporal dependencies, essential for accurate separation. Additionally, it introduces a context-aware prediction framework and a refined upsampling technique to minimize artifacts. The model demonstrated competitive performance, particularly in singing voice separation, against state-of-the-art spectrogram-based systems.

## Secondary Models Considered

### Mel-Band RoFormer (Vocal Variant)
This improved version of the Mel-Band RoFormer was trained on a larger dataset than the original Mel-Band Roformer and can be utilized for separating the vocal stem into further sub-stems.

### DrumSep
Drum separation based on a Hybrid Demucs([HT Demucs](https://arxiv.org/abs/1909.01174)) model.
Code to facilitate access to the model presented in the thesis "Separación de fuentes en grabaciones de batería mediante aprendizaje automático" (2022).
The model takes as input an audio file (mp3, wav, flac, or ogg) corresponding to a drum recording and exports 4 audio files classified as - Kick, Snare, Cymbals, Toms

## How it works

MusSep separates in two levels.

1. **Primary stems (Stage 1).** A Band-Split RoFormer turns the mixture into vocals, bass, drums and other. It is deterministic: complex masks on the STFT, as in the paper.
2. **Sub-stems (Stages 2-4).** A generative model splits vocals, drums and other into labelled sub-stems. It works in the latent space of Meta's DAC-VAE and generates every sub-stem of a parent at once, conditioned on the parent stem. It is trained with flow matching, then distilled for 1-4 step sampling.

```
mixture ──► BS-RoFormer ──► vocals ──┐
                        ├─► drums  ──┼─► DAC-VAE encode ─► joint sub-stem DiT ─► DAC-VAE decode ─► ratio masks on the parent ─► sub-stems
                        ├─► other  ──┘   (L and R separately,   (flow matching,        (outputs sum to the parent,
                        └─► bass         128 + 128 channels)     or distilled)           keep its phase)
```

### Stage 1: BS-RoFormer (`models/bs_roformer.py`)

- The module layout and state-dict keys are those of the community implementation (lucidrains / ZFTurbo), so public BS-RoFormer checkpoints load with `load_community_state_dict` (or `--stage1 model.ckpt --stage1-config model.yaml`). The tests check every key name and shape, and check our rotary embedding against `rotary-embedding-torch`.
- 62 frequency bands, a time transformer and a band transformer per layer, RoPE on both axes, one mask head per stem.
- Training (`configs/roformer_base.yaml`) follows the paper: random-mix augmentation (each stem from an independent song, ±3 dB gain, 10% silent stems, −50 dB loudness gate), waveform L1 + multi-resolution complex STFT L1 (windows 4096-256, hop 147), AdamW 5e-4 ×0.9 every 40k steps, EMA 0.999, 8-second excerpts, inference in 8-second windows with a 4-second hop. Two additions not in the paper: gradient-norm clipping at 1.0 (a guard against loss spikes; with Adam it mostly just down-weights outlier steps), and an EMA warmup (see below).
- Tier `base` (dim 384, 6 layer pairs, 8 heads of 48) has 72.2M parameters for one stereo stem, the paper's model size.

### Stage 2: taxonomy and latents

Three parents are split; bass is not. The candidate classes live in `configs/taxonomy.yaml`:

| Parent | Candidate classes |
|---|---|
| vocals | lead_vocals, backing_vocals (the residual: every vocal that is not the lead) |
| drums | kick, snare, toms, hihat (falls back to cymbals), cymbals, drums_residual |
| other | guitar, piano, keys, strings, wind, percussion, other_residual |

MoisesDB labels map to classes by stem name and track type. For example, overheads, full-kit recordings and drum machines go to `drums_residual`, and the click track is excluded. Slakh2100 maps by General MIDI program and only supervises "other", since its drums are one stem.

**Minimum-data rule.** Classes are not frozen by hand. `scripts/taxonomy_stats.py` prints songs and minutes per class, per dataset. Any class with fewer than `min_songs` (default 20) **MoisesDB training** songs is merged: into its stated fallback (hi-hat → cymbals), otherwise into its parent's residual. Slakh does not count towards the threshold, so a class cannot survive on synthetic data alone. In vocals, backing vocals is the residual: choir, vocoder, beatbox and any unmapped vocal label land there. A residual is never merged away; if backing vocals still has fewer than 20 songs, vocals is not split at all, because a lead/backing split trained on so few songs would mostly learn to output silence in the second slot. (A rare residual does not stop drums or other from being split.) The frozen result, with the printed report as its header, is what preprocessing and the model use.

```bash
python -m scripts.taxonomy_stats --moisesdb /data/moisesdb --slakh /data/slakh2100 --freeze configs/taxonomy.frozen.yaml
```

**Latents** (`scripts/preprocess_latents.py`). Each sub-stem and each parent is encoded once with the frozen DAC-VAE, so the VAE never runs in training.
- Encoding uses the posterior mean, so a dataset encodes the same way twice.
- The VAE is mono: left and right are encoded separately and stored side by side (`[L, R]`, 2 × latent width channels).
- Presence is decided per latent frame (the RMS of each hop against −60 dBFS), not per song. A class is stored if it is active for at least a second, and the loader decides per crop whether it is present (active in ≥5% of the crop's frames). A guitar that plays only in the chorus is absent in verse crops: it is left out of the "present classes" layout there, and in the full-set layout its slot is trained on that silent stretch.
- Optional extra copies with a random gain per class, and an optional conditioning latent with the other stems mixed in at −30 to −15 dB, to imitate imperfect Stage 1 output.
- The manifest records latent statistics, the latent of silence, the frozen taxonomy and the split of every song.
- Size: float16 latents take `2 bytes × channels × frame rate` per second per stem. At 256 channels and 25 frames/s (the published DAC-VAE figures; the code reads the real values from the model) that is about 3.8 MB per stem per 5-minute song.

### Stage 3: joint sub-stem DiT (`models/dit.py`, `pipeline/flow_matching.py`)

- Tokens sit on a (stem slot × time) grid. Each block attends along time within a slot (RoPE) and across slots at each time step. Slots have no position and are told apart only by their class embedding, so the model is equivariant to slot order (tested). Padding slots are masked out of attention and the loss.
- The parent latent is concatenated to every slot's input. The class embedding and flow time drive adaLN-Zero modulation.
- OT conditional flow matching (t = 0 noise, t = 1 data) with a per-slot validity mask, and 10% label dropout for classifier-free guidance.
- **Absent classes ("hybrid").** For each example a coin decides between two layouts. Either every class of the parent is listed, and absent ones must generate the latent of silence, or only the present classes are listed. So at inference you can ask for the whole set, or just `--stems guitar,piano`.
- Inference: Euler integration (50 steps by default), windows of the training length cross-faded in latent space, decoding, then a ratio-mask projection onto the parent's STFT. The outputs then sum to the parent stem and keep its phase; generated audio is only used as a magnitude guide. The projection is on by default. It needs every class, so a request for one stem (`--stems guitar`) still generates the whole set in one pass and returns only the guitar; `--no-project` generates only the requested slots, faster but without those guarantees.
- Label dropout stays on in training even though guidance is off by default (scale 1.0), so classifier-free guidance remains available. The distilled student is trained without it; guidance can instead be baked in through `teacher_cfg_scale`.
- EMA weights are what is evaluated and shipped. The decay ramps up as `min(decay, (1 + n) / (10 + n))`, reaching 0.999 after about 9k steps and 0.9999 after about 90k, so short runs are not judged on an average dominated by the initialisation.

### Stage 4: consistency distillation (`pipeline/distillation.py`)

The student starts from the teacher's weights. It learns to map two points on the teacher's ODE trajectory, one Euler step apart, to the same clean latent. The distance is Pseudo-Huber in latent space, and the target comes from an EMA copy of the student. This takes sampling from 50 steps to 1-4. Validation reports the latent distance between the student's few-step samples and the teacher's 50-step samples from the same noise.

## Installation

```bash
git clone https://github.com/AdvaithGS/MusSep.git && cd MusSep
./repository_init.sh            # .venv, requirements, tests; add --cpu for CPU-only PyTorch
# or: conda env create -f environment.yml && conda activate mussep
```

The DAC-VAE comes from `pip install git+https://github.com/facebookresearch/dacvae` (in `requirements.txt`); its weights download from the Hugging Face Hub on first use. It is not needed for Stage 1.

## Data

| Dataset | Used for | Layout expected |
|---|---|---|
| [MUSDB18-HQ](https://sigsep.github.io/datasets/musdb.html) | Stage 1 | `<root>/{train,test}/<song>/{mixture,vocals,bass,drums,other}.wav`; the official 14-song validation split is built in |
| [MoisesDB](https://music.ai/research/) | Stages 2-4 | `<root>/<provider>/<track id>/data.json` and `<stem>/<file id>.<ext>`, as distributed; 10% of songs held out, stratified: every class that can pass the 20-song rule gets up to 2 held-out songs, never so many that its training count drops below 20 |
| [Slakh2100](http://www.slakh.com/) | Stages 2-4, "other" only; weighted to a third of the "other" examples (vocals and drums examples are all MoisesDB) | `<root>/{train,validation,test}/TrackXXXXX/metadata.yaml` and `stems/Sxx.flac` |

## Running it

```bash
# Stage 1: one model per stem, as in the paper
python -m pipeline.train_base --musdb-root /data/musdb18hq --stems vocals --tier base --out runs/vocals

# Stage 2: freeze the taxonomy, encode latents, train the DiT
python -m scripts.preprocess_latents --moisesdb /data/moisesdb --slakh /data/slakh2100 \
    --taxonomy configs/taxonomy.yaml --out /data/latents --num-aug 1 --bleed-db -30 -15
python -m pipeline.train_generative --manifest /data/latents/manifest.json --tier base --out runs/stage2

# Stage 4: distil to 2-step sampling
python -m pipeline.train_distill --teacher runs/stage2/best.pt --manifest /data/latents/manifest.json --out runs/stage4

# Separate
python -m scripts.separate song.wav --out separated/ \
    --stage1 runs/vocals/best.pt runs/drums/best.pt runs/bass/best.pt --stage2 runs/stage4/last.pt
python -m scripts.separate drums.wav --out separated/ --stage2 runs/stage4/last.pt --input-is drums --stems kick,snare

# Benchmarks
python -m scripts.evaluate stage1 --musdb-root /data/musdb18hq --stage1 runs/*/best.pt --metric museval
python -m scripts.evaluate stage2 --moisesdb /data/moisesdb --stage2 runs/stage4/last.pt --manifest /data/latents/manifest.json
```

- Every training script resumes from `<out>/last.pt`, keeps `best.pt` by validation score, and logs to `<out>/log.jsonl`.
- Model size comes from `--tier` (`debug`, `small`, `base`).
- Precision is bf16 where the GPU supports it natively, else fp16 with a gradient scaler. STFTs, losses and the VAE always run in fp32.

## Benchmarks

**No model has been trained yet, so there are no results to report.**

- **Stage 1 target.** The BS-RoFormer paper reports 9.80 dB average SDR on the MUSDB18-HQ test set for its model trained on MUSDB18-HQ alone (vocals 10.66, bass 11.31, drums 9.49, other 7.73). That number is comparable only with:
  - `scripts/evaluate.py stage1 --metric museval` (BSSEval v4: median over 1-second frames, then median over the 50 test songs),
  - on the full test set,
  - for models trained on MUSDB18-HQ alone.

  The script says when a run does not meet these conditions.
- **Community checkpoints.** Nearly all published BS-RoFormer checkpoints used extra training data. Their scores belong in a separate row and must not be presented as MUSDB-only.
- **Stage 2** has no established benchmark. `evaluate.py stage2` gives the true parent stem to the generator for held-out MoisesDB songs and reports per class:
  - SDR and SI-SDR where the class is present,
  - the output level relative to the parent where it is absent (it should be silence).

  It measures Stage 2 alone, not the cascade.

## Tests

```bash
pytest -q tests
```

48 tests run on CPU in under a minute, on synthetic data laid out like the real datasets, with tiny models and a toy linear VAE. They cover:
- shapes and the community state-dict layout,
- RoPE against the reference package,
- the paper-size parameter count,
- slot equivariance and padding isolation in the DiT,
- the flow-matching target and samplers,
- the minimum-data rule,
- one end-to-end run through every script: freeze, encode, all three trainings with resume, separation and both evaluations.

They check the plumbing, not separation quality.

## Directory structure

```
configs/    roformer_base.yaml · dit_latent.yaml · distillation.yaml · taxonomy.yaml
data/       musdb18.py (Stage 1 random mixing) · moisesdb18.py · slakh.py · taxonomy.py (classes, mapping, minimum-data rule)
            unified_dataset.py (Stage 2 latent crops) · dataset.py · mix_pipeline.py · transforms.py
models/     bs_roformer.py · band_split.py · roformer.py (community-compatible Stage 1)
            dac_vae.py (DAC-VAE wrapper, stereo, chunking) · dit.py (joint sub-stem DiT)
pipeline/   train_base.py · train_generative.py · train_distill.py · flow_matching.py · distillation.py · infer.py
scripts/    taxonomy_stats.py · preprocess_latents.py · separate.py · evaluate.py
utils/      audio.py · loss.py · metrics.py · training.py · config.py
tests/
```

## Datasets and Models Sourcing

### MoisesDB
The [largest publicly available set](https://music.ai/blog/press/introducing-moisesdb-the-ultimate-multitrack-dataset-for-source-separation-beyond-4-stems/) of multitrack audio recordings for source separation beyond 4-stems. This comprehensive dataset comprises 240 previously unreleased songs created by 47 artists that span twelve high-level genres. The total duration of the dataset is 14 hours, 24 minutes and 46 seconds, where the average recording is 3:36 seconds, with a standard deviation of 66 seconds.

### Python Audio Separator
The [Python Audio Separator](https://github.com/nomadkaraoke/python-audio-separator/tree/main) is an easy to use stem (e.g. instrumental/vocals) separation from CLI or as a python package, using a variety of amazing pre-trained models (primarily from UVR)

## Future Expansion
Possibly include the [AudioX](https://github.com/ZeyueT/AudioX/) architecture, which uses a Diffusion Transformer(DiT) model to convert multimodal (masked) input in audio (music or any other sound). It has been called an Anything-to-Audio and Music Generation model. AudioX can generate both general audio and music with high quality, while offering flexible natural language control and seamless processing of various modalities including text, video, image, music, and audio. Its key innovation is a multi-modal masked training strategy that masks inputs across modalities and forces the model to learn from masked inputs,yielding robust and unified cross-modal representations.
