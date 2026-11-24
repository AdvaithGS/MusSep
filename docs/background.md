# Background and models considered

- [Background](#background)
- [Primary Models Considered](#primary-models-considered)
- [Secondary Models Considered](#secondary-models-considered)
- [Tools](#tools)

## Background
Four-stem music source separation has been approached with a variety of architectures and techniques: mel-spectrogram and raw-waveform models, unsupervised learning from unlabelled data (particularly when little labelled data is available), [recurrent neural networks](https://arxiv.org/abs/1910.06379), band-split models, [generative adversarial networks](https://kkp15.github.io/), transformers, [convolutional neural networks](https://www.ijert.org/audio-stems-separation-using-deep-learning), multitask learning and DTT-Net. Hybrid transformers, rotary-position-embedding transformers, sparse-compression networks and adversarial networks have shown higher performance on specific stems.

## Primary Models Considered

### SCNet (Sparse Compression Network)
[SCNet](https://arxiv.org/abs/2401.13276), proposed by Tong et al., is a frequency-domain network that splits spectrograms into subbands. Its key innovation is a sparse compression technique that applies higher compression ratios to less informative subbands, maintaining fidelity in critical subbands while reducing computational load. This improves separation performance and reduces model complexity, making SCNet suitable for real-time applications.

### Mel-Band RoFormer
The [Mel-Band RoFormer](https://arxiv.org/abs/2310.01809) uses a mel-scale-based projection scheme to capture perceptual differences in frequencies, particularly in vocals and drums. Its key innovation is the use of overlapping subbands based on the mel scale, mimicking the human auditory system's preference for higher resolution at lower frequencies. An [improved implementation](https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model) is also available.

### Hybrid Transformer Demucs (HT Demucs)
[HT Demucs](https://arxiv.org/abs/2211.08553), developed by Rouard et al., combines temporal and spectral U-Nets with Transformer layers to enlarge its receptive field. Working in both the time and frequency domains, it applies self-attention within each domain and cross-attention across them, capturing short- and long-term dependencies in complex musical mixtures.

### Band-Split RNN for MSS
[Band-Split RNN (BSRNN)](https://arxiv.org/abs/2209.15174) is a frequency-domain MSS model that explicitly divides the input spectrogram into subbands of predefined bandwidths. It operates on high-sample-rate audio, which lets it handle super-wide-band signals efficiently, and combines sequence-level and band-level modelling with RNNs to capture intra-band and inter-band dependencies.

### Wave-U-Net: End-to-End Separation in the Time Domain
[Wave-U-Net](https://arxiv.org/abs/1806.03185) is a time-domain adaptation of the U-Net architecture for audio source separation. Unlike spectrogram-based methods, it operates directly on waveforms, so it models magnitude and phase together without relying on a fixed transformation. Multi-scale feature maps capture long-term temporal dependencies, which are essential for accurate separation. It also introduces context-aware prediction and a refined upsampling technique that reduce artifacts. The model performed competitively against state-of-the-art spectrogram-based systems, particularly in singing voice separation.

## Secondary Models Considered

### Mel-Band RoFormer (Vocal Variant)
This variant of the Mel-Band RoFormer was trained on a larger dataset than the original model and can be used to separate the vocal stem into further sub-stems.

### DrumSep
DrumSep separates drum recordings with a [Hybrid Demucs](https://arxiv.org/abs/2111.03600) model. It accompanies the thesis "Separación de fuentes en grabaciones de batería mediante aprendizaje automático" (2022). The model takes a drum recording (mp3, wav, flac or ogg) and outputs four files: kick, snare, cymbals and toms.

## Tools

### Python Audio Separator
The [Python Audio Separator](https://github.com/nomadkaraoke/python-audio-separator/tree/main) is a command-line tool and Python package for stem separation (for example, vocals and instrumental) using a range of pre-trained models, primarily from Ultimate Vocal Remover (UVR).
