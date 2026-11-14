"""End-to-end inference: mixture -> primary stems (Stage 1, BS-RoFormer) ->
labelled sub-stems (Stage 2, latent flow matching, optionally distilled).

Library module; the command line lives in ``scripts/separate.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from data.taxonomy import Taxonomy
from models.bs_roformer import BSRoformer, build_bs_roformer, load_community_state_dict
from models.dac_vae import DACVAEWrapper, LatentNormalizer
from models.dit import JointSubStemDiT
from pipeline.flow_matching import ODESolver
from utils.audio import chunked_apply, ratio_mask_projection, resample
from utils.config import load_yaml
from utils.training import AmpContext, load_checkpoint

STAGE1_KIND = "mussep_stage1"
STAGE2_KIND = "mussep_stage2"


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

@dataclass
class Stage1Model:
    model: BSRoformer
    stems: List[str]
    sample_rate: int = 44100
    chunk_size: int = 352800


def load_stage1(
    checkpoint: str | Path,
    config: Optional[str | Path] = None,
    stems: Optional[Sequence[str]] = None,
    use_ema: bool = True,
) -> Stage1Model:
    """Loads a Stage 1 model.

    * A checkpoint written by ``pipeline/train_base.py`` is self-describing.
    * A community BS-RoFormer checkpoint (plain state dict, ``.ckpt`` or
      ``.safetensors``) needs ``config``: the YAML it was trained with, in the
      Music-Source-Separation-Training layout (``model:``, ``audio:``,
      ``training.instruments`` / ``training.target_instrument``). ``stems``
      overrides the stem names. Such checkpoints were, as a rule, trained
      with data beyond MUSDB18HQ: see ``load_community_state_dict``.
    """
    path = Path(checkpoint)
    payload: Any = None
    if path.suffix != ".safetensors":
        payload = torch.load(str(path), map_location="cpu", weights_only=True)

    if isinstance(payload, Mapping) and payload.get("kind") == STAGE1_KIND:
        model = build_bs_roformer(payload["model_kwargs"])
        state = payload["ema"] if use_ema and payload.get("ema") is not None else payload["model"]
        model.load_state_dict(state)
        return Stage1Model(
            model.eval(),
            list(stems or payload["target_stems"]),
            int(payload["audio"]["sample_rate"]),
            int(payload["audio"]["segment_samples"]),
        )

    if config is None:
        raise ValueError(
            f"{path} is not a MusSep Stage 1 checkpoint; pass the model YAML it was trained with (config=...)"
        )
    cfg = load_yaml(config)
    model = build_bs_roformer(cfg["model"])
    load_community_state_dict(model, payload if payload is not None else path, strict=True)
    if stems is None:
        training = cfg.get("training") or {}
        target = training.get("target_instrument")
        stems = [target] if target else list(training.get("instruments") or [])
    if len(stems) != model.num_stems:
        raise ValueError(
            f"the model outputs {model.num_stems} stem(s) but {len(stems)} name(s) are known ({list(stems)}); "
            "pass the stem names explicitly"
        )
    audio = cfg.get("audio") or {}
    return Stage1Model(
        model.eval(), list(stems), int(audio.get("sample_rate", 44100)), int(audio.get("chunk_size", 352800))
    )


@torch.no_grad()
def run_stage1(
    entry: Stage1Model,
    mixture: torch.Tensor,
    device: torch.device,
    amp: Optional[AmpContext] = None,
    overlap: float = 0.5,
    batch_size: int = 1,
    window: str = "flat",
) -> Dict[str, torch.Tensor]:
    """Chunked separation of a whole ``(C, T)`` mixture at the model's sample
    rate. ``overlap = 0.5`` with 8-second chunks is the paper's protocol
    (4-second hop, overlap-average)."""
    model = entry.model.to(device).eval()
    amp = amp or AmpContext(device, enabled=False)
    hop = max(1, int(round(entry.chunk_size * (1.0 - overlap))))

    def apply(batch: torch.Tensor) -> torch.Tensor:
        with amp.autocast():
            return model(batch)

    estimates = chunked_apply(apply, mixture, entry.chunk_size, hop, batch_size, window, device)
    return {name: estimates[index] for index, name in enumerate(entry.stems)}


class Stage1Separator:
    """One or several Stage 1 models (the paper trains one model per stem).

    If ``other`` is not produced by any model but vocals, bass and drums are,
    it is derived as ``mixture - (vocals + bass + drums)``.
    """

    def __init__(
        self,
        models: Sequence[Stage1Model],
        device: torch.device,
        amp: Optional[AmpContext] = None,
        overlap: float = 0.5,
        batch_size: int = 1,
        window: str = "flat",
    ):
        if not models:
            raise ValueError("Stage1Separator needs at least one model")
        self.models, self.device, self.amp = list(models), device, amp
        self.overlap, self.batch_size, self.window = overlap, batch_size, window

    @property
    def stems(self) -> List[str]:
        return [name for entry in self.models for name in entry.stems]

    def separate(self, mixture: torch.Tensor, sample_rate: int) -> Dict[str, torch.Tensor]:
        """``mixture`` (C, T) at ``sample_rate`` -> ``{stem: (C, T)}`` at ``sample_rate``."""
        length = mixture.shape[-1]
        result: Dict[str, torch.Tensor] = {}
        for entry in self.models:
            signal = mixture if sample_rate == entry.sample_rate else resample(mixture, sample_rate, entry.sample_rate)
            for name, audio in run_stage1(
                entry, signal, self.device, self.amp, self.overlap, self.batch_size, self.window
            ).items():
                if sample_rate != entry.sample_rate:
                    audio = resample(audio, entry.sample_rate, sample_rate)
                    audio = torch.nn.functional.pad(audio, (0, max(0, length - audio.shape[-1])))[..., :length]
                result[name] = audio
        if "other" not in result and all(name in result for name in ("vocals", "bass", "drums")):
            result["other"] = mixture.float().cpu() - result["vocals"] - result["bass"] - result["drums"]
        return result


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------

@dataclass
class Stage2Model:
    model: JointSubStemDiT
    taxonomy: Taxonomy
    normalizer: LatentNormalizer
    silence: torch.Tensor            # (C,) raw (un-normalised) latent of silence
    info: Dict[str, Any]
    segment_frames: int
    distilled: bool = False
    num_steps: int = 50
    cfg_scale: float = 1.0


def load_stage2(checkpoint: str | Path, use_ema: bool = True) -> Stage2Model:
    """Loads a checkpoint written by ``train_generative.py`` or ``train_distill.py``."""
    payload = load_checkpoint(checkpoint)
    if payload.get("kind") != STAGE2_KIND:
        raise ValueError(f"{checkpoint} is not a MusSep Stage 2 checkpoint")
    model = JointSubStemDiT(**payload["model_kwargs"])
    state = payload["ema"] if use_ema and payload.get("ema") is not None else payload["model"]
    model.load_state_dict(state)
    latent = payload["latent"]
    distilled = bool(payload.get("distilled", False))
    return Stage2Model(
        model=model.eval(),
        taxonomy=Taxonomy.from_dict(payload["taxonomy"]),
        normalizer=LatentNormalizer(latent["mean"], latent["std"]),
        silence=latent["silence"].float(),
        info={key: value for key, value in latent.items() if not torch.is_tensor(value)},
        segment_frames=int(payload["segment_frames"]),
        distilled=distilled,
        num_steps=int(payload.get("num_steps", 2 if distilled else 50)),
        cfg_scale=float(payload.get("cfg_scale", 1.0)),
    )


class SubStemGenerator:
    """Splits one parent stem into labelled sub-stems."""

    def __init__(
        self,
        stage2: Stage2Model,
        vae: DACVAEWrapper,
        device: torch.device,
        amp: Optional[AmpContext] = None,
        batch_size: int = 4,
    ):
        info = stage2.info
        for key in ("sample_rate", "hop_length", "latent_dim"):
            if int(info[key]) != int(getattr(vae, key)):
                raise ValueError(
                    f"VAE mismatch: checkpoint was trained with {key}={info[key]}, the loaded VAE has {getattr(vae, key)}"
                )
        self.stage2, self.vae, self.device = stage2, vae.to(device), device
        self.model = stage2.model.to(device).eval()
        self.normalizer = stage2.normalizer.to(device)
        self.amp = amp or AmpContext(device, enabled=False)
        self.batch_size = batch_size
        self.channels = int(info["channels"])

    # -- latent-domain generation ----------------------------------------

    def _sample(self, parent, labels, sampler, num_steps, cfg_scale, generator) -> torch.Tensor:
        """``parent`` (B, C, L) normalised, ``labels`` (B, S) -> (B, S, C, L)."""
        def model(*args):
            with self.amp.autocast():
                return self.model(*args)

        if sampler == "consistency":
            return ODESolver.sample_consistency(model, labels, parent, num_steps=num_steps, generator=generator)
        return ODESolver.sample_euler(
            model, labels, parent, num_steps=num_steps, cfg_scale=cfg_scale, generator=generator
        )

    @torch.no_grad()
    def generate_latents(
        self,
        parent_latent: torch.Tensor,
        class_ids: Sequence[int],
        sampler: str,
        num_steps: int,
        cfg_scale: float = 1.0,
        overlap_frames: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """``parent_latent`` (C, F) raw -> sub-stem latents (S, C, F) raw.

        Long inputs are generated in windows of the training length and
        cross-faded in latent space.
        """
        length = self.stage2.segment_frames
        frames = parent_latent.shape[-1]
        parent = self.normalizer.normalize(parent_latent.to(self.device).float())
        if frames < length:
            silence = self.normalizer.normalize(self.stage2.silence.to(self.device)[:, None])
            parent = torch.cat([parent, silence.expand(-1, length - frames)], dim=-1)
        total = parent.shape[-1]
        overlap = length // 8 if overlap_frames is None else overlap_frames
        hop = max(1, length - overlap)
        starts = list(range(0, max(1, total - length + 1), hop))
        if starts[-1] + length < total:
            starts.append(total - length)

        labels = torch.tensor(list(class_ids), dtype=torch.long, device=self.device)
        out = torch.zeros(len(class_ids), parent.shape[0], total, device=self.device)
        norm = torch.zeros(total, device=self.device)
        for offset in range(0, len(starts), self.batch_size):
            batch_starts = starts[offset:offset + self.batch_size]
            windows = torch.stack([parent[:, s:s + length] for s in batch_starts])
            batch_labels = labels[None].expand(len(batch_starts), -1)
            samples = self._sample(windows, batch_labels, sampler, num_steps, cfg_scale, generator)
            for sample, start in zip(samples, batch_starts):
                weight = torch.ones(length, device=self.device)
                if overlap > 0 and start > 0:
                    weight[:overlap] = torch.linspace(0.0, 1.0, overlap + 2, device=self.device)[1:-1]
                if overlap > 0 and start + length < total:
                    weight[-overlap:] = torch.linspace(1.0, 0.0, overlap + 2, device=self.device)[1:-1]
                out[..., start:start + length] += sample * weight
                norm[start:start + length] += weight
        out = out / norm.clamp_min(1e-6)
        return self.normalizer.denormalize(out[..., :frames])

    # -- audio-domain API -------------------------------------------------

    @torch.no_grad()
    def separate(
        self,
        parent_audio: torch.Tensor,
        sample_rate: int,
        parent: str,
        stems: Optional[Sequence[str]] = None,
        sampler: Optional[str] = None,
        num_steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        project: Optional[bool] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Splits ``parent_audio`` (C, T) - e.g. the "other" stem from Stage 1.

        ``stems``: classes to generate (default: every class of ``parent``).
        ``sampler``: ``"euler"`` (flow-matching teacher) or ``"consistency"``
        (distilled student); default follows the checkpoint.
        ``project`` (default on): re-synthesise the outputs as ratio masks on
        the parent's STFT, so that they keep its phase and the full set sums to
        the parent. The masks need every class of the parent, so with projection
        the whole set is always generated (the joint model does this in one pass;
        the cost is the extra slots) and only the requested classes are returned.
        ``project=False`` generates only the requested slots and returns the
        decoded audio as is: cheaper, but it does not sum to anything and has
        the VAE's phase, not the recording's.
        """
        taxonomy = self.stage2.taxonomy
        all_names = taxonomy.parent_classes(parent)
        names = list(stems) if stems else all_names
        unknown = [name for name in names if name not in all_names]
        if unknown:
            raise ValueError(f"{unknown} are not classes of '{parent}' (choose from {all_names})")
        if project is None:
            project = True
        generate = all_names if project else names
        sampler = sampler or ("consistency" if self.stage2.distilled else "euler")
        num_steps = num_steps or self.stage2.num_steps
        cfg_scale = self.stage2.cfg_scale if cfg_scale is None else cfg_scale

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        length = parent_audio.shape[-1]
        latent = self.vae.encode(parent_audio.float(), sample_rate)                       # (C*D, F)
        latents = self.generate_latents(
            latent, [taxonomy.class_id(name) for name in generate], sampler, num_steps, cfg_scale, generator=generator
        )
        audio = self.vae.decode(latents, sample_rate, length, channels=self.channels)         # (S, C, T), on device
        if project:   # STFT masks on the GPU too; the CPU only receives the finished stems
            audio = ratio_mask_projection(parent_audio.float().to(audio.device), audio)
        audio = audio.cpu()
        return {name: audio[generate.index(name)] for name in names}


# ---------------------------------------------------------------------------
# Both stages
# ---------------------------------------------------------------------------

def group_stems_by_parent(taxonomy: Taxonomy, stems: Sequence[str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for name in stems:
        if name not in taxonomy.class_names[1:]:
            raise ValueError(f"unknown sub-stem '{name}'; available: {taxonomy.class_names[1:]}")
        grouped.setdefault(taxonomy.parent_of(name), []).append(name)
    return grouped


def separate_hierarchical(
    mixture: torch.Tensor,
    sample_rate: int,
    stage1: Optional[Stage1Separator],
    stage2: Optional[SubStemGenerator] = None,
    parents: Optional[Sequence[str]] = None,
    stems: Optional[Sequence[str]] = None,
    input_is: Optional[str] = None,
    **sampling: Any,
) -> Dict[str, Any]:
    """Returns ``{"primary": {stem: audio}, "sub": {parent: {class: audio}}}``.

    ``input_is`` names the parent stem that ``mixture`` already is (skips
    Stage 1). ``parents`` restricts which parents are split; ``stems``
    requests specific sub-stem classes instead of whole sets.
    """
    if input_is is not None:
        primary = {input_is: mixture.float().cpu()}
    elif stage1 is not None:
        primary = stage1.separate(mixture, sample_rate)
    else:
        raise ValueError("either a Stage 1 separator or input_is is required")

    sub: Dict[str, Dict[str, torch.Tensor]] = {}
    if stage2 is not None:
        taxonomy = stage2.stage2.taxonomy
        requested = group_stems_by_parent(taxonomy, stems) if stems else None
        targets = list(requested) if requested else list(parents or taxonomy.parents)
        for parent in targets:
            if parent not in primary:
                if requested or parents:
                    raise ValueError(f"parent stem '{parent}' was not produced by Stage 1 (have {list(primary)})")
                continue
            sub[parent] = stage2.separate(
                primary[parent], sample_rate, parent, requested[parent] if requested else None, **sampling
            )
    return {"primary": primary, "sub": sub}
