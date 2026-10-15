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
4-Stem music source separation has been implemented in the past using a variety of architectures, each focusing on different techniques such as mel-spectrograms, raw audio waveforms, unsupervised learning from unlabelled data (particularly in scenarios of low data availability), [recurrent neural networks](https://arxiv.org/abs/1910.06379), specific frequency bands, generative adversarial networks [GANs](https://kkp15.github.io/), transformers, convolutional neural networks [CNNs](https://www.ijert.org/audio-stems-separation-using-deep-learning), multitask learning, and DTT-Net[2]. Including hybrid transformers, rotary positional encoding transformers[4], sparse compression networks[1], and adversarial networks[18], revealed higher performance on specific stems.

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

## Installation

I recommend using a Conda environment to manage dependencies, especially for specific PyTorch and DAC versions.

```
git clone https://github.com/AdvaithGS/MusSep.git
cd MusSep

# Create and activate the environment
conda env create -f environment.yml
conda activate mussep

# Alternatively, using pip
pip install -r requirements.txt
```

## Quickstart: Inference

Download the pre-trained weights (links coming soon) and place them in the checkpoints/ directory.

To separate a single track into its constituent stems:
```
python scripts/separate.py \
  --input path/to/mixture.wav \
  --output path/to/output_dir/ \
  --checkpoint checkpoints/mussep_distilled_v1.pt \
  --fast-sample True # Utilizes consistency distillation for 2-step sampling
```

## Training Pipeline

The training process is modularized into three distinct phases. Ensure your MUSDB18HQ dataset is formatted according to the instructions in data/README.md (to be added).

1. Base Model Training
Train the Band-Split RoFormer on primary stems.
```
python pipeline/train_base.py --config configs/roformer_base.yaml
```

2. Generative Latent Modeling
Freeze the base model and train the DiT and Flow Matching components in the DAC latent space.
```
python pipeline/train_generative.py --config configs/dit_latent.yaml
```

3. Consistency Distillation
Accelerate the generative model for real-time or near-real-time inference.
```
python pipeline/train_distill.py --config configs/distillation.yaml
```

## Benchmarks
Benchmarks

Model |  MUSDB18HQ SDR (dB) | Inference Speed (RTF)
MusSep (Base Only) | 8.95 | 0.4x
MusSep (Full Ensemble) | 9.80 | 1.2x
MusSep (Distilled) | 9.65 | 0.5x



## Datasets and Models Sourcing

### MoisesDB
The [largest publicly available set](https://music.ai/blog/press/introducing-moisesdb-the-ultimate-multitrack-dataset-for-source-separation-beyond-4-stems/) of multitrack audio recordings for source separation beyond 4-stems. This comprehensive dataset comprises 240 previously unreleased songs created by 47 artists that span twelve high-level genres. The total duration of the dataset is 14 hours, 24 minutes and 46 seconds, where the average recording is 3:36 seconds, with a standard deviation of 66 seconds.

### Python Audio Separator
The [Python Audio Separator](https://github.com/nomadkaraoke/python-audio-separator/tree/main) is an easy to use stem (e.g. instrumental/vocals) separation from CLI or as a python package, using a variety of amazing pre-trained models (primarily from UVR)

## Future Expansion
Possibly include the [AudioX](https://github.com/ZeyueT/AudioX/) architecture, which uses a Diffusion Transformer(DiT) model to convert multimodal (masked) input in audio (music or any other sound). It has been called an Anything-to-Audio and Music Generation model. AudioX can generate both general audio and music with high quality, while offering flexible natural language control and seamless processing of various modalities including text, video, image, music, and audio. Its key innovation is a multi-modal masked training strategy that masks inputs across modalities and forces the model to learn from masked inputs,yielding robust and unified cross-modal representations.
