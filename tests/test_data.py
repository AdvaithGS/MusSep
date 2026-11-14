"""Dataset indexers, random mixing, taxonomy and the minimum-data rule."""
import random

import pytest
import torch

from data.moisesdb18 import index_moisesdb, split_of, stratified_val
from data.musdb18 import MUSDBRandomMixDataset, MUSDBTrackDataset, index_musdb
from data.slakh import index_slakh
from data.taxonomy import Taxonomy, normalize_label, plan_merges
from scripts.taxonomy_stats import freeze_taxonomy

from .conftest import MUSDB_TEST, MUSDB_TRAIN, MUSDB_VALID


@pytest.fixture(scope="module")
def taxonomy():
    return Taxonomy.load("configs/taxonomy.yaml")


def test_musdb_splits(musdb_root):
    assert [t["name"] for t in index_musdb(musdb_root, "train")] == sorted(MUSDB_TRAIN)
    assert [t["name"] for t in index_musdb(musdb_root, "valid")] == sorted(MUSDB_VALID)
    assert len(index_musdb(musdb_root, "train_full")) == len(MUSDB_TRAIN) + len(MUSDB_VALID)
    assert [t["name"] for t in index_musdb(musdb_root, "test")] == MUSDB_TEST


def test_random_mix_dataset(musdb_root):
    ds = MUSDBRandomMixDataset(musdb_root, target_stems=("vocals", "drums"), segment_samples=22050, epoch_size=8,
                               silence_prob=0.0, seed=3)
    mixture, targets = ds[0]
    assert mixture.shape == (2, 22050) and targets.shape == (2, 2, 22050)
    assert targets.abs().max() > 0
    m2, t2 = ds[0]
    torch.testing.assert_close(mixture, m2)          # seeded items are reproducible
    coherent = MUSDBRandomMixDataset(musdb_root, target_stems=("vocals", "bass", "drums", "other"), segment_samples=22050,
                                     random_mix=False, gain_range_db=None, silence_prob=0.0, seed=1)
    m, t = coherent[0]
    torch.testing.assert_close(m, t.sum(0))          # mixture is the sum of the stems


def test_track_dataset(musdb_root):
    item = MUSDBTrackDataset(musdb_root, "valid", ("vocals",), max_seconds=1.0)[0]
    assert item["mixture"].shape == (2, 44100) and set(item["targets"]) == {"vocals"}


def test_taxonomy_file(taxonomy):
    assert taxonomy.class_names[0] == "<null>"
    assert taxonomy.parents == ["vocals", "drums", "other"]
    assert not taxonomy.frozen
    lookup = taxonomy.moisesdb_lookup
    assert lookup("Drums", "Hi-Hat").raw_class == "hihat"
    assert lookup("vocals", "Lead Female Singer").raw_class == "lead_vocals"
    assert lookup("vocals", "other (vocoder, beatboxing etc)").raw_class == "backing_vocals"
    assert lookup("vocals", "kazoo").raw_class == "backing_vocals"         # backing vocals is the vocals residual
    assert taxonomy.residual["vocals"] == "backing_vocals"
    assert lookup("other_keys", "synth pad").raw_class == "keys"
    assert lookup("other plucked", "banjo").raw_class == "other_residual"
    assert lookup("bass", "bass guitar") == (("bass", None, True, False))
    assert lookup("other", "click track").excluded
    assert not lookup("vocals", "kazoo").matched                 # a default rule decided
    assert taxonomy.slakh_lookup(33, False).parent == "bass"
    assert taxonomy.slakh_lookup(25, False).raw_class == "guitar"
    assert taxonomy.slakh_lookup(None, True).parent == "drums"
    assert normalize_label("Hi-Hat ") == normalize_label("hihat")


def test_min_data_rule(taxonomy):
    songs = {name: set() for names in taxonomy.candidates.values() for name in names}
    many = {f"s{i}" for i in range(30)}
    for name in ("lead_vocals", "kick", "snare", "toms", "cymbals", "drums_residual", "guitar", "piano", "keys",
                 "strings", "percussion", "other_residual"):
        songs[name] = set(many)
    songs["hihat"] = {"s1", "s2"}                    # rare -> folded into cymbals, not the residual
    songs["wind"] = {f"s{i}" for i in range(12)}     # rare -> other's residual
    songs["backing_vocals"] = {f"s{i}" for i in range(25)}
    merges, collapsed, log = plan_merges(taxonomy, songs, 20)
    assert merges == {"hihat": "cymbals", "wind": "other_residual"}
    assert collapsed == set()
    frozen = taxonomy.with_merges(merges, collapsed)
    assert "hihat" not in frozen.class_names and "wind" not in frozen.class_names
    assert frozen.resolve("hihat") == "cymbals" and frozen.resolve("wind") == "other_residual"
    assert frozen.parent_classes("drums") == ["kick", "snare", "toms", "cymbals", "drums_residual"]

    songs["backing_vocals"] = {"s1"}                 # the vocals residual is too rare: vocals stops being split
    merges, collapsed, log = plan_merges(taxonomy, songs, 20)
    assert "vocals" in collapsed and "backing_vocals" not in merges
    assert any("residual of 'vocals'" in line for line in log)
    frozen = taxonomy.with_merges(merges, collapsed)
    assert frozen.parents == ["drums", "other"] and frozen.resolve("lead_vocals") is None
    songs["backing_vocals"] = set(many)
    songs["other_residual"] = {"s1"}                 # other's residual alone being rare does not stop the split
    merges, collapsed, _ = plan_merges(taxonomy, songs, 20)
    assert collapsed == set()


def test_stratified_holdout():
    songs = {f"s{i:03d}": {"common"} for i in range(100)}
    for i in range(23):
        songs[f"s{i:03d}"].add("rare")                # 23 songs: room for 2 held out, 21 left to train
    for i in range(30, 51):
        songs[f"s{i:03d}"].add("edge")                # 21 songs: only 1 can be held out
    for i in range(60, 65):
        songs[f"s{i:03d}"].add("doomed")              # 5 songs: merged anyway, no guarantee
    val = stratified_val(songs, val_fraction=0.1, min_songs=20, min_val=2, seed=0)
    held = lambda c: sum(1 for s in val if c in songs[s])  # noqa: E731
    assert 2 <= held("rare") <= 3 and held("edge") == 1 and len(val) == 10
    assert sum(1 for s in songs if "rare" in songs[s] and s not in val) >= 20
    assert stratified_val(songs, val_fraction=0.1, min_songs=20, min_val=2, seed=0) == val   # deterministic
    assert stratified_val(songs, val_fraction=0.0) == set()


def test_frozen_roundtrip(taxonomy, tmp_path):
    frozen = taxonomy.with_merges({"wind": "other_residual"}, [])
    frozen.save(tmp_path / "t.yaml", header="test")
    again = Taxonomy.load(tmp_path / "t.yaml")
    assert again.frozen and again.class_names == frozen.class_names


def test_moisesdb_index(moisesdb_root, taxonomy):
    tracks = index_moisesdb(moisesdb_root, taxonomy, val_fraction=0.5)
    assert len(tracks) == 8
    track = tracks[0]
    assert "other / click track" in track.excluded
    raw = track.raw_groups("drums")
    assert set(raw) == {"kick", "snare", "hihat", "drums_residual"}
    assert track.parent_files("bass") and not track.raw_groups("bass")
    assert {t.split for t in tracks} == {"train", "val"}
    assert split_of("abc", 0.5) == split_of("abc", 0.5)


def test_slakh_index(slakh_root, taxonomy):
    tracks = index_slakh(slakh_root, taxonomy)
    assert len(tracks) == 4 and {t.split for t in tracks} == {"train", "val"}
    files = tracks[0].files
    assert len(files) == 5                                      # the unrendered stem is skipped
    assert {f.parent for f in files} == {"other", "bass", "drums"}
    assert tracks[0].raw_groups("other") .keys() == {"piano", "guitar", "wind"}


def test_freeze_counts_moisesdb_train_only(moisesdb_root, slakh_root, taxonomy):
    tracks = index_moisesdb(moisesdb_root, taxonomy, val_fraction=0.5) + index_slakh(slakh_root, taxonomy)
    frozen, report, stats = freeze_taxonomy(taxonomy, tracks, min_songs=2, verbose=False)
    train_songs = sum(1 for t in tracks if t.source == "moisesdb" and t.split == "train")
    assert len(stats.songs("kick", ["moisesdb"], "train")) == train_songs
    assert len(stats.songs("wind", ["moisesdb"], "all")) == 0          # wind only exists in Slakh here
    assert frozen.resolve("wind") == "other_residual"                   # so Slakh alone cannot keep it
    assert frozen.resolve("hihat") == "hihat"
    assert "Songs and minutes per candidate class" in report and frozen.frozen
