"""Every entry point, end to end, on synthetic data with tiny models and the toy VAE:

taxonomy freeze -> latent pre-encoding -> Stage 1 training -> Stage 2 training ->
Stage 4 distillation -> separation CLI -> both evaluations.

This checks plumbing (shapes, checkpoint formats, resume, CLI wiring), not quality.
"""
import json

import pytest
import torch

from data.moisesdb18 import index_moisesdb
from data.slakh import index_slakh
from data.taxonomy import Taxonomy
from data.unified_dataset import UnifiedManifestDataset
from models.dac_vae import build_vae
from pipeline import train_base, train_distill, train_generative
from pipeline.infer import load_stage1, load_stage2
from scripts import evaluate, separate
from scripts.preprocess_latents import preprocess
from scripts.taxonomy_stats import freeze_taxonomy
from utils.training import load_checkpoint

from .conftest import TOY_VAE, tmp_config

TRAIN = {"max_steps": 2, "log_every": 1, "valid_every": 2, "save_every": 2, "batch_size": 2, "valid_batches": 2, "seed": 0}


@pytest.fixture(scope="module")
def latents(moisesdb_root, slakh_root, tmp_path_factory):
    taxonomy = Taxonomy.load("configs/taxonomy.yaml")
    tracks = index_moisesdb(moisesdb_root, taxonomy, val_fraction=0.5) + index_slakh(slakh_root, taxonomy)
    frozen, _, _ = freeze_taxonomy(taxonomy, tracks, min_songs=2, verbose=False)
    out = tmp_path_factory.mktemp("latents")
    manifest = preprocess(tracks, frozen, build_vae(TOY_VAE), out, vae_name=TOY_VAE, num_aug=1,
                          bleed_range_db=(-30.0, -20.0), verbose=False)
    return out / "manifest.json", manifest, frozen


def test_manifest(latents):
    path, manifest, frozen = latents
    entries = manifest["entries"]
    assert {e["split"] for e in entries} >= {"train", "val"}
    assert {e["aug"] for e in entries if e["split"] == "train"} == {0, 1}
    assert all(e["aug"] == 0 for e in entries if e["split"] == "val")
    assert {e["parent"] for e in entries if e["source"] == "slakh"} == {"other"}   # Slakh supervises "other" only
    assert manifest["latent_dim"] == 8 and manifest["channels"] == 2 and len(manifest["latent_stats"]["mean"]) == 16
    payload = torch.load(path.parent / entries[0]["latent_path"], weights_only=True)
    assert payload["stems"].shape[0] == len(entries[0]["classes"]) and "cond" in payload
    assert payload["stems"].dtype == torch.float16
    assert payload["class_active"].shape == (payload["stems"].shape[0], payload["stems"].shape[-1])
    # the parent latent is the encoding of the sum of the classes
    assert payload["parent"].shape == payload["stems"].shape[1:]


def test_unified_dataset_layouts(latents):
    path, _, frozen = latents
    full = UnifiedManifestDataset(path, "train", segment_frames=40, full_set_prob=1.0)
    parent, stems, labels, valid = full[0]
    assert parent.shape == (16, 40) and stems.shape == (frozen.max_slots, 16, 40)
    names = frozen.parent_classes(full.entries[0]["parent"])
    assert valid.sum() == len(names)                                   # every class, absent ones -> silence
    assert labels[: len(names)].tolist() == [frozen.class_id(n) for n in names]
    sparse = UnifiedManifestDataset(path, "train", segment_frames=40, full_set_prob=0.0)
    _, _, labels, valid = sparse[0]
    assert valid.sum() == len(sparse.entries[0]["classes"])            # present classes only
    assert (labels[~valid] == 0).all()
    weights = sparse.make_sampler().weights
    other = [(float(w), e) for w, e in zip(weights, sparse.entries) if e["parent"] == "other"]
    slakh = sum(w for w, e in other if e["source"] == "slakh")
    assert abs(slakh / sum(w for w, _ in other) - sparse.slakh_fraction) < 1e-6   # a third of "other" only
    assert all(float(w) == 1.0 for w, e in zip(weights, sparse.entries) if e["source"] == "moisesdb")


@pytest.fixture(scope="module")
def stage1(musdb_root, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("stage1")
    config = tmp_config("configs/roformer_base.yaml", tmp / "c.yaml", model={"tier": "debug"},
                        audio={"segment_seconds": 1.0}, data={"num_workers": 0, "epoch_size": 8},
                        training={**TRAIN, "valid_max_seconds": 2.0, "target_stems": ["vocals", "drums"]})
    args = train_base.build_parser().parse_args(["--config", str(config), "--musdb-root", str(musdb_root),
                                                 "--out", str(tmp / "run"), "--device", "cpu"])
    path = train_base.train(args)
    return path, config, args


def test_stage1_checkpoint_and_resume(stage1):
    path, config, args = stage1
    state = load_checkpoint(path)
    assert state["step"] == 2 and state["target_stems"] == ["vocals", "drums"] and state["best"] is not None
    log = [json.loads(line) for line in (path.parent / "log.jsonl").read_text().splitlines()]
    assert {r["kind"] for r in log} == {"train", "valid"}
    args.max_steps = 3
    train_base.train(args)                                             # resumes from last.pt
    assert load_checkpoint(path)["step"] == 3
    entry = load_stage1(path)
    assert entry.stems == ["vocals", "drums"] and entry.chunk_size == 44100


@pytest.fixture(scope="module")
def stage2(latents, tmp_path_factory):
    manifest, _, _ = latents
    tmp = tmp_path_factory.mktemp("stage2")
    config = tmp_config("configs/dit_latent.yaml", tmp / "c.yaml", model={"tier": "debug"},
                        data={"segment_seconds": 0.4, "num_workers": 0}, training=TRAIN)
    path = train_generative.train(train_generative.build_parser().parse_args(
        ["--config", str(config), "--manifest", str(manifest), "--out", str(tmp / "run"), "--device", "cpu"]))
    return path


def test_stage2_checkpoint(stage2, latents):
    loaded = load_stage2(stage2)
    assert not loaded.distilled and loaded.num_steps == 50 and loaded.segment_frames == 40
    assert loaded.taxonomy.class_names == latents[2].class_names
    assert (stage2.parent / "best.pt").is_file()


@pytest.fixture(scope="module")
def stage4(stage2, latents, tmp_path_factory):
    manifest, _, _ = latents
    tmp = tmp_path_factory.mktemp("stage4")
    config = tmp_config("configs/distillation.yaml", tmp / "c.yaml", distill={"num_scales": 4},
                        data={"segment_seconds": 0.4, "num_workers": 0}, training=TRAIN)
    return train_distill.train(train_distill.build_parser().parse_args(
        ["--config", str(config), "--teacher", str(stage2), "--manifest", str(manifest), "--out", str(tmp / "run"),
         "--device", "cpu"]))


def test_distilled_checkpoint(stage4):
    loaded = load_stage2(stage4)
    assert loaded.distilled and loaded.num_steps == 2


def test_subset_request_is_projected_from_the_full_set(stage4, musdb_root):
    from pipeline.infer import SubStemGenerator
    from utils.audio import load_audio

    generator = SubStemGenerator(load_stage2(stage4), build_vae(TOY_VAE), torch.device("cpu"))
    parent, sr = load_audio(musdb_root / "test" / "Delta - Four" / "drums.wav", channels=2)
    full = generator.separate(parent, sr, "drums", seed=0)
    one = generator.separate(parent, sr, "drums", stems=["kick"], seed=0)
    assert list(one) == ["kick"]
    torch.testing.assert_close(one["kick"], full["kick"])
    torch.testing.assert_close(sum(full.values()), parent, atol=1e-3, rtol=1e-3)     # the full set sums to the parent
    raw = generator.separate(parent, sr, "drums", stems=["kick"], seed=0, project=False)
    assert not torch.allclose(raw["kick"], one["kick"])


def test_separate_cli(stage1, stage4, musdb_root, tmp_path):
    song = musdb_root / "test" / "Delta - Four" / "mixture.wav"
    written = separate.run(separate.build_parser().parse_args(
        [str(song), "--out", str(tmp_path), "--stage1", str(stage1[0]), "--stage2", str(stage4), "--device", "cpu"]))
    names = {p.relative_to(tmp_path / "mixture").as_posix() for p in written}
    taxonomy = load_stage2(stage4).taxonomy
    assert {"vocals.wav", "drums.wav"} <= names
    for parent in ("vocals", "drums"):
        if parent in taxonomy.parents:
            assert {f"{parent}/{c}.wav" for c in taxonomy.parent_classes(parent)} <= names
    # a subset of classes, from an isolated parent stem, with the Euler sampler
    stem = musdb_root / "test" / "Delta - Four" / "drums.wav"
    wanted = taxonomy.parent_classes("drums")[:2]
    written = separate.run(separate.build_parser().parse_args(
        [str(stem), "--out", str(tmp_path / "b"), "--stage2", str(stage4), "--input-is", "drums",
         "--stems", ",".join(wanted), "--sampler", "euler", "--steps", "3", "--device", "cpu"]))
    assert sorted(p.stem for p in written) == sorted(wanted)


def test_evaluate(stage1, stage4, musdb_root, moisesdb_root, latents, tmp_path):
    result = evaluate.main(["stage1", "--musdb-root", str(musdb_root), "--stage1", str(stage1[0]), "--split", "valid",
                            "--metric", "usdr", "--max-seconds", "2", "--device", "cpu", "--out", str(tmp_path / "s1.json")])
    assert set(result["summary"]) == {"vocals", "drums", "avg"} and not result["full_protocol"]
    result = evaluate.main(["stage2", "--moisesdb", str(moisesdb_root), "--stage2", str(stage4), "--manifest",
                            str(latents[0]), "--max-seconds", "1", "--device", "cpu", "--out", str(tmp_path / "s2.json")])
    assert result["sdr"] and result["distilled"]
    assert json.loads((tmp_path / "s2.json").read_text())["stage"] == 2
