# Training

The commands are in the [README](../README.md#running-it). They depend on each other in this order:

```mermaid
flowchart TD
    MUS["MUSDB18-HQ"] --> T1["train_base.py<br/>one Stage 1 model per stem"]
    MOI["MoisesDB + Slakh2100"] --> CV["check_vae.py<br/>is the VAE good enough?"]
    CV --> TX["taxonomy_stats.py --freeze<br/>which classes survive"]
    TX --> PRE["preprocess_latents.py<br/>encode once"]
    PRE --> T2["train_generative.py<br/>Stage 2 / 3 DiT"]
    T2 --> T4["train_distill.py<br/>Stage 4, 1-4 step sampling"]
    T1 --> SEP["separate.py"]
    T4 --> SEP
    T1 --> EV["evaluate.py<br/>stage1, stage2, cascade"]
    T4 --> EV
```

## Running on rented or free GPUs

Every stage runs on one GPU. The code is set up for low-cost, interruptible machines:

- **Measure before you pay.** `python -m scripts.benchmark stage1 --tiers small base --price-per-hour 0.45` finds the largest batch that fits for each tier. It reports seconds per step, peak memory, and the hours and cost of the config's full run. With `--musdb-root` (Stage 1) or `--manifest` (Stage 2) it also times the data loader at several worker counts and says whether the CPU keeps up with the GPU. A free Kaggle or Colab session is enough to run it.
- **Interruptions cost minutes.** Checkpoints are written every `save_every` steps and at least every `save_every_minutes` (20). SIGTERM or Ctrl-C finishes the current step, saves `last.pt` and exits. Spot instances, Colab and Kaggle send SIGTERM before they reclaim the machine. Rerunning the same command resumes.

```mermaid
flowchart TB
    STEP["Training step"] --> DUE{"save_every steps,<br/>20 minutes, or a<br/>stop signal?"}
    DUE -->|"no"| STEP
    DUE -->|"yes"| SAVE["Save last.pt"]
    SAVE --> STOP{"Stop signal<br/>SIGTERM or Ctrl-C?"}
    STOP -->|"no"| STEP
    STOP -->|"yes"| EXIT["Exit cleanly;<br/>rerun the same command<br/>to resume"]
```

- **CPU balance.**
  - Data-loader workers default to every available CPU but two (container limits respected).
  - Each worker runs a single intra-op thread, and the main process keeps only a few, so they don't oversubscribe the cores.
  - Batches are prefetched (`prefetch_factor`) into pinned memory.
  - Pre-encoding loads and resamples the next songs in worker processes while the GPU encodes the current one, and skips songs whose latents already exist.
  - Resampling in the VAE wrapper and the final projection run on the GPU.

```mermaid
flowchart LR
    subgraph CPU["CPU: data-loader workers, one thread each"]
        W1["Read and mix audio,<br/>or crop cached latents"]
    end
    subgraph GPU["GPU"]
        G1["Forward, backward,<br/>optimizer step"]
    end
    W1 -->|"prefetched batches,<br/>pinned memory"| G1
```

- **Stage 2 data in RAM.** `cache_in_memory: auto` loads every latent file into memory when they fit in half of it. Rented machines often keep data on network disks, where reading thousands of small files a minute starves the GPU.
- **Speed switches.** TF32 is on (`tf32: true`). `torch.compile` is off by default (`compile: true` or `--compile`); measure it with the benchmark first, since it costs compile time at start-up and the gain varies.
- **Disk.** Copy datasets to the machine's local disk before training, not a mounted network volume. Stage 1 reads random excerpts constantly.
