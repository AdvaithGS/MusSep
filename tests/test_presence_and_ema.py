"""Per-crop class presence in the Stage 2 loader, and EMA warmup."""
import json

import torch
import torch.nn as nn

from data.taxonomy import Taxonomy
from data.unified_dataset import UnifiedManifestDataset
from utils.training import ModelEMA


def _manifest(tmp_path, class_active):
    taxonomy = Taxonomy.load("configs/taxonomy.yaml")
    frames, channels = 100, 4
    ids = [taxonomy.class_id("guitar"), taxonomy.class_id("piano")]
    stems = torch.stack([torch.full((channels, frames), 3.0), torch.full((channels, frames), 5.0)])
    payload = {"parent": stems.sum(0).half(), "stems": stems.half(), "class_ids": torch.tensor(ids)}
    if class_active is not None:
        payload["class_active"] = class_active
    torch.save(payload, tmp_path / "e.pt")
    manifest = {"vae": "toy", "sample_rate": 8000, "hop_length": 80, "latent_dim": 2, "channels": 2, "latent_rate": 100,
                "taxonomy": taxonomy.to_dict(), "latent_stats": {"mean": [0.0] * channels, "std": [1.0] * channels},
                "silence_latent": [-1.0] * channels,
                "entries": [{"id": "x", "track_id": "x", "source": "moisesdb", "split": "train", "parent": "other",
                             "aug": 0, "latent_path": "e.pt", "num_frames": frames, "classes": ["guitar", "piano"]}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path / "manifest.json", taxonomy


def test_class_silent_in_the_crop_is_absent(tmp_path):
    active = torch.ones(2, 100, dtype=torch.bool)
    active[1, 2:] = False                                   # piano plays 2% of the crop
    path, taxonomy = _manifest(tmp_path, active)
    sparse = UnifiedManifestDataset(path, segment_frames=100, full_set_prob=0.0)
    _, stems, labels, valid = sparse[0]
    assert labels[valid].tolist() == [taxonomy.class_id("guitar")]
    full = UnifiedManifestDataset(path, segment_frames=100, full_set_prob=1.0)
    _, stems, labels, valid = full[0]
    names = taxonomy.parent_classes("other")
    piano, keys = names.index("piano"), names.index("keys")
    assert valid.sum() == len(names)
    assert torch.all(stems[piano] == 5.0)                  # listed, with its real (here: stored) encoding as target
    assert torch.all(stems[keys] == -1.0)                  # never stored: the silence latent


def test_old_manifests_without_activity_still_load(tmp_path):
    path, taxonomy = _manifest(tmp_path, None)
    _, _, labels, valid = UnifiedManifestDataset(path, segment_frames=100, full_set_prob=0.0)[0]
    assert sorted(labels[valid].tolist()) == sorted([taxonomy.class_id("guitar"), taxonomy.class_id("piano")])


def test_ema_warmup():
    model = nn.Linear(1, 1, bias=False)
    nn.init.zeros_(model.weight)
    ema = ModelEMA(model, decay=0.9999, warmup=True)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema.update(model)                                       # n = 0: decay 0.1
    assert abs(float(ema.module.weight) - 0.9) < 1e-6
    plain = ModelEMA(nn.Linear(1, 1, bias=False), decay=0.9999)
    nn.init.zeros_(plain.module.weight)
    plain.update(model)
    assert float(plain.module.weight) < 1e-3
    ema.updates = 10 ** 6
    with torch.no_grad():
        model.weight.fill_(2.0)
    before = float(ema.module.weight)
    ema.update(model)                                       # capped at the configured decay
    assert abs(float(ema.module.weight) - (0.9999 * before + 0.0001 * 2.0)) < 1e-6
