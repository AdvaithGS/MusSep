"""BS-RoFormer: shapes, band table, community state-dict layout, RoPE."""
import pytest
import torch

from models.band_split import DEFAULT_FREQS_PER_BANDS, get_bs_roformer_bands, validate_band_widths
from models.bs_roformer import BSRoformer, build_bs_roformer, extract_state_dict, load_community_state_dict
from models.roformer import RotaryEmbedding
from utils.config import load_yaml, resolve_tier
from utils.training import count_parameters

TINY = dict(dim=32, depth=1, stereo=True, num_stems=1, time_transformer_depth=1, freq_transformer_depth=1,
            dim_head=8, heads=2, stft_hop_length=441)


def test_band_table_covers_every_bin_once():
    bands = get_bs_roformer_bands()
    assert len(bands) == 62 and sum(DEFAULT_FREQS_PER_BANDS) == 1025
    assert bands[0][0] == 0 and bands[-1][1] == 1025
    assert all(a[1] == b[0] for a, b in zip(bands, bands[1:]))
    with pytest.raises(ValueError):
        validate_band_widths((2,) * 10, 1025)


@pytest.mark.parametrize("stems", [1, 3])
def test_forward_shapes_and_backward(stems):
    model = BSRoformer(**{**TINY, "num_stems": stems})
    audio = torch.randn(2, 2, 22050)
    out = model(audio)
    assert out.shape == (2, stems, 2, 22050)
    out.abs().mean().backward()
    assert all(p.grad is not None for n, p in model.named_parameters() if p.requires_grad)


def test_mono_input():
    model = BSRoformer(**{**TINY, "stereo": False})
    assert model(torch.randn(1, 8820)).shape == (1, 1, 1, 8820)


def test_gradient_checkpointing_matches():
    torch.manual_seed(0)
    a = BSRoformer(**TINY)
    b = BSRoformer(**{**TINY, "use_torch_checkpoint": True})
    b.load_state_dict(a.state_dict())
    x = torch.randn(1, 2, 8820)
    a.train(), b.train()
    torch.testing.assert_close(a(x), b(x))


def test_mask_init_std_gives_near_silent_output():
    model = BSRoformer(**{**TINY, "mask_init_std": 1e-6})
    x = torch.randn(1, 2, 8820)
    assert model(x).abs().max() < 1e-2 * x.abs().max()


def test_state_dict_matches_community_layout():
    """Key names and shapes of the lucidrains / ZFTurbo implementation."""
    dim, heads, dim_head = 32, 2, 8
    model = BSRoformer(dim=dim, depth=2, stereo=True, num_stems=2, time_transformer_depth=1, freq_transformer_depth=1,
                       dim_head=dim_head, heads=heads, mask_estimator_depth=2)
    sd = model.state_dict()
    inner = heads * dim_head
    for layer in range(2):
        for axis in range(2):
            p = f"layers.{layer}.{axis}.layers.0"
            assert sd[f"{p}.0.rotary_embed.freqs"].shape == (dim_head // 2,)
            assert sd[f"{p}.0.norm.gamma"].shape == (dim,)
            assert sd[f"{p}.0.to_qkv.weight"].shape == (3 * inner, dim)
            assert f"{p}.0.to_qkv.bias" not in sd
            assert sd[f"{p}.0.to_gates.weight"].shape == (heads, dim)
            assert sd[f"{p}.0.to_gates.bias"].shape == (heads,)
            assert sd[f"{p}.0.to_out.0.weight"].shape == (dim, inner)
            assert f"{p}.0.to_out.0.bias" not in sd
            assert sd[f"{p}.1.net.0.gamma"].shape == (dim,)
            assert sd[f"{p}.1.net.1.weight"].shape == (4 * dim, dim)
            assert sd[f"{p}.1.net.4.weight"].shape == (dim, 4 * dim)
    assert sd["final_norm.gamma"].shape == (dim,)
    widths = DEFAULT_FREQS_PER_BANDS
    for i, w in enumerate(widths):
        dim_in = 2 * w * 2
        assert sd[f"band_split.to_features.{i}.0.gamma"].shape == (dim_in,)
        assert sd[f"band_split.to_features.{i}.1.weight"].shape == (dim, dim_in)
        for s in range(2):
            assert sd[f"mask_estimators.{s}.to_freqs.{i}.0.0.weight"].shape == (4 * dim, dim)
            assert sd[f"mask_estimators.{s}.to_freqs.{i}.0.2.weight"].shape == (2 * dim_in, 4 * dim)
    # 11 tensors per transformer block, 2 per layer pair, plus final norm, band split and mask heads
    assert len(sd) == 2 * 2 * 11 + 1 + 62 * 3 + 2 * 62 * 4


def test_paper_size_parameter_count():
    """configs/roformer_base.yaml tier 'base' is the paper's ~72M-parameter model."""
    kwargs = resolve_tier(load_yaml("configs/roformer_base.yaml"), "base")
    model = build_bs_roformer({**kwargs, "num_stems": 1})
    assert 71e6 < count_parameters(model) < 73e6


def test_rope_matches_rotary_embedding_torch():
    reference = pytest.importorskip("rotary_embedding_torch")
    ours, theirs = RotaryEmbedding(16), reference.RotaryEmbedding(16)
    torch.testing.assert_close(ours.freqs, theirs.freqs)
    x = torch.randn(2, 3, 50, 16)
    torch.testing.assert_close(ours.rotate_queries_or_keys(x), theirs.rotate_queries_or_keys(x))


def test_load_community_state_dict_roundtrip(tmp_path):
    a, b = BSRoformer(**TINY), BSRoformer(**TINY)
    wrapped = {"state_dict": {f"module.{k}": v for k, v in a.state_dict().items()}, "epoch": 3}
    torch.save(wrapped, tmp_path / "model.ckpt")
    assert load_community_state_dict(b, tmp_path / "model.ckpt") == ([], [])
    x = torch.randn(1, 2, 8820)
    torch.testing.assert_close(a(x), b(x))
    with pytest.raises(RuntimeError, match="does not match"):
        load_community_state_dict(BSRoformer(**{**TINY, "dim": 48}), tmp_path / "model.ckpt")
    assert set(extract_state_dict(wrapped)) == set(a.state_dict())


def test_community_yaml_with_python_tuple(tmp_path):
    (tmp_path / "c.yaml").write_text(
        "model:\n  dim: 32\n  depth: 1\n  stereo: true\n  num_stems: 1\n  time_transformer_depth: 1\n"
        "  freq_transformer_depth: 1\n  dim_head: 8\n  heads: 2\n  sample_rate: 44100\n"
        "  freqs_per_bands: !!python/tuple\n" + "".join(f"  - {w}\n" for w in DEFAULT_FREQS_PER_BANDS))
    model = build_bs_roformer(load_yaml(tmp_path / "c.yaml")["model"])
    assert model.freqs_per_bands == tuple(DEFAULT_FREQS_PER_BANDS)


ZFTURBO_STYLE_YAML = """\
audio:
  chunk_size: 88200
  dim_f: 1024
  hop_length: 441
  n_fft: 2048
  num_channels: 2
  sample_rate: 44100
model:
  dim: 32
  depth: 1
  stereo: true
  num_stems: 1
  time_transformer_depth: 1
  freq_transformer_depth: 1
  linear_transformer_depth: 0
  freqs_per_bands: !!python/tuple
FREQS  dim_head: 8
  heads: 2
  attn_dropout: 0.1
  ff_dropout: 0.1
  flash_attn: true
  dim_freqs_in: 1025
  stft_n_fft: 2048
  stft_hop_length: 441
  stft_win_length: 2048
  stft_normalized: false
  mask_estimator_depth: 2
  multi_stft_resolution_loss_weight: 1.0
  multi_stft_resolutions_window_sizes: !!python/tuple
  - 4096
  - 2048
  multi_stft_hop_size: 147
  multi_stft_normalized: false
  sage_attention: false
training:
  batch_size: 2
  instruments:
  - vocals
  - other
  target_instrument: vocals
inference:
  batch_size: 1
  num_overlap: 2
"""


def test_community_checkpoint_with_its_training_yaml(tmp_path):
    from pipeline.infer import load_stage1, run_stage1

    text = ZFTURBO_STYLE_YAML.replace("FREQS", "".join(f"  - {w}\n" for w in DEFAULT_FREQS_PER_BANDS))
    (tmp_path / "config.yaml").write_text(text)
    with pytest.warns(UserWarning, match="sage_attention"):
        source = build_bs_roformer(load_yaml(tmp_path / "config.yaml")["model"])
    torch.save({"state_dict": source.state_dict()}, tmp_path / "model.ckpt")
    with pytest.warns(UserWarning):
        entry = load_stage1(tmp_path / "model.ckpt", tmp_path / "config.yaml")
    assert entry.stems == ["vocals"] and entry.chunk_size == 88200 and entry.sample_rate == 44100
    out = run_stage1(entry, torch.randn(2, 100_000), torch.device("cpu"))
    assert set(out) == {"vocals"} and out["vocals"].shape == (2, 100_000)


def test_safetensors_checkpoint(tmp_path):
    safetensors = pytest.importorskip("safetensors.torch")
    a, b = BSRoformer(**TINY), BSRoformer(**TINY)
    safetensors.save_file(a.state_dict(), str(tmp_path / "m.safetensors"))
    load_community_state_dict(b, tmp_path / "m.safetensors")
    x = torch.randn(1, 2, 8820)
    torch.testing.assert_close(a(x), b(x))
