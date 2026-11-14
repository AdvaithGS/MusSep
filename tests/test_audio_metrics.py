"""utils/audio.py, utils/metrics.py, utils/loss.py."""
import torch

from utils.audio import chunked_apply, ratio_mask_projection, resample, rms_db, stft, istft
from utils.loss import MultiResolutionComplexSTFTLoss
from utils.metrics import aggregate, chunked_sdr_median, sdr, si_sdr


def test_chunked_apply_reconstructs_identity():
    audio = torch.randn(2, 10_000)
    for window in ("flat", "hann"):
        out = chunked_apply(lambda b: b[:, None], audio, chunk_size=1000, hop_size=400, batch_size=3, window=window)
        assert out.shape == (1, 2, 10_000)
        torch.testing.assert_close(out[0], audio)


def test_ratio_mask_projection_sums_to_parent():
    torch.manual_seed(0)
    parent = torch.randn(2, 30_000)
    guides = torch.rand(3, 1, 1) * torch.randn(3, 2, 30_000)
    out = ratio_mask_projection(parent, guides, segment=10_000)
    assert out.shape == (3, 2, 30_000)
    torch.testing.assert_close(out.sum(0), parent, atol=1e-4, rtol=1e-4)
    zero = ratio_mask_projection(parent, torch.zeros(2, 2, 30_000))
    torch.testing.assert_close(zero.sum(0), parent, atol=1e-4, rtol=1e-4)


def test_stft_roundtrip_and_resample():
    x = torch.randn(2, 8000)
    torch.testing.assert_close(istft(stft(x, 1024, 256), 1024, 256, length=8000), x, atol=1e-5, rtol=1e-5)
    y = resample(x, 16000, 8000)
    assert y.shape == (2, 4000)


def test_metrics():
    ref = torch.randn(2, 44100)
    assert sdr(ref, ref) > 70
    assert abs(sdr(ref, ref + 0.1 * ref.std() * torch.randn_like(ref)) - 20) < 1
    assert si_sdr(ref, 3 * ref) > 70
    assert chunked_sdr_median(ref, ref, 44100) > 70
    scores = aggregate([{"a": 1.0, "b": 3.0}, {"a": 3.0, "b": float("nan")}], "median")
    assert scores == {"a": 2.0, "b": 3.0, "avg": 2.5}
    assert rms_db(torch.zeros(1, 10)) < -90


def test_loss_is_zero_for_identical_and_positive_otherwise():
    loss = MultiResolutionComplexSTFTLoss()
    x = torch.randn(2, 1, 2, 8000)
    assert float(loss(x, x)) == 0.0
    total, parts = loss(x, x + 0.1, return_breakdown=True)
    assert float(total) > 0 and set(parts) >= {"wave", "stft"}


def test_museval_wrapper():
    """BSSEval v4 through museval: the metric behind published MUSDB18-HQ tables."""
    import pytest

    pytest.importorskip("museval")
    from utils.metrics import museval_sdr

    torch.manual_seed(0)
    refs = {"vocals": torch.randn(2, 3 * 44100) * 0.1, "drums": torch.randn(2, 3 * 44100) * 0.1}
    good = {k: v + 0.01 * torch.randn_like(v) for k, v in refs.items()}
    bad = {k: v + 0.1 * torch.randn_like(v) for k, v in refs.items()}
    hi, lo = museval_sdr(refs, good), museval_sdr(refs, bad)
    assert set(hi) == {"vocals", "drums"}
    assert all(hi[k] > lo[k] + 10 for k in hi)          # 20 dB less noise -> clearly higher SDR
