"""DiT, flow matching, samplers, distillation and the VAE wrapper."""
import torch
import torch.nn as nn

from models.dac_vae import DACVAEWrapper, LatentNormalizer, ToyVAE, build_vae
from models.dit import JointSubStemDiT
from pipeline.distillation import ConsistencyFlowDistiller, pseudo_huber
from pipeline.flow_matching import MaskedOTFlowMatcher, ODESolver, drop_labels, predict_x1

C, T, S = 16, 24, 4


def dit(**kw):
    torch.manual_seed(0)
    model = JointSubStemDiT(in_channels=C, hidden_dim=32, num_heads=4, depth=2, num_stem_classes=8, **kw)
    for p in model.parameters():          # adaLN-Zero starts at v = 0; perturb so the tests see real outputs
        nn.init.normal_(p, std=0.05) if p.dim() > 1 else None
    return model


def batch(b=2):
    torch.manual_seed(1)
    x = torch.randn(b, S, C, T)
    labels = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])[:b]
    valid = labels > 0
    return x, torch.rand(b), labels, torch.randn(b, C, T), valid


def test_zero_init_outputs_zero_velocity():
    model = JointSubStemDiT(in_channels=C, hidden_dim=32, num_heads=4, depth=2, num_stem_classes=8)
    x, t, labels, parent, valid = batch()
    assert model(x, t, labels, parent, valid).abs().max() == 0


def test_shapes_and_padding_isolation():
    model = dit().eval()
    x, t, labels, parent, valid = batch()
    out = model(x, t, labels, parent, valid)
    assert out.shape == x.shape
    x2 = x.clone()
    x2[~valid] = 100 * torch.randn_like(x2[~valid])      # padding slots must not influence valid slots
    torch.testing.assert_close(model(x2, t, labels, parent, valid)[valid], out[valid])


def test_slot_permutation_equivariance():
    model = dit().eval()
    x, t, labels, parent, valid = batch(1)
    perm = torch.tensor([2, 0, 1, 3])
    out = model(x, t, labels, parent, valid)
    out_p = model(x[:, perm], t, labels[:, perm], parent, valid[:, perm])
    torch.testing.assert_close(out_p, out[:, perm], atol=1e-5, rtol=1e-5)


def test_all_padding_row_is_finite():
    model = dit().eval()
    x, t, labels, parent, _ = batch()
    valid = torch.zeros(2, S, dtype=torch.bool)
    valid[0, 0] = True
    assert torch.isfinite(model(x, t, labels, parent, valid)).all()


def test_flow_matching_target_and_mask():
    fm = MaskedOTFlowMatcher(sigma_min=0.0)
    x1, x0 = torch.randn(2, S, C, T), torch.randn(2, S, C, T)
    xt, target = fm.interpolate(x0, x1, torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(xt[0], x0[0])
    torch.testing.assert_close(xt[1], x1[1])
    torch.testing.assert_close(target, x1 - x0)

    class Oracle(nn.Module):        # predicts the exact target for valid slots, garbage elsewhere
        def forward(self, x, t, labels, parent, valid):
            v = (x1 - x0).clone()
            v[~valid] = 1e3
            return v
    _, _, labels, parent, valid = batch()
    loss = fm(Oracle(), x1, parent, labels, valid, t=torch.rand(2), noise=x0)
    assert float(loss) < 1e-10


def test_label_dropout():
    labels = torch.randint(1, 5, (1000, 3))
    dropped = drop_labels(labels, 0.3)
    rows = (dropped == 0).all(1).float().mean()
    assert 0.2 < rows < 0.4 and ((dropped == 0) | (dropped == labels)).all()


def test_samplers_recover_a_known_flow():
    """With the exact OT velocity for a point target, Euler and consistency sampling land on it."""
    target = torch.randn(1, S, C, T)

    class Exact(nn.Module):
        def forward(self, x, t, labels, parent, valid):
            tt = t.reshape(-1, 1, 1, 1)
            return (target - x) / (1 - tt).clamp_min(1e-4)
    _, _, labels, parent, valid = batch(1)
    out = ODESolver.sample_euler(Exact(), labels, parent, num_steps=10, valid_mask=valid)
    torch.testing.assert_close(out, target, atol=1e-4, rtol=1e-4)
    out = ODESolver.sample_consistency(Exact(), labels, parent, num_steps=3, valid_mask=valid)
    torch.testing.assert_close(out, target, atol=1e-4, rtol=1e-4)
    out = ODESolver.sample_euler(Exact(), labels, parent, num_steps=4, valid_mask=valid, cfg_scale=2.0)
    torch.testing.assert_close(out, target, atol=1e-4, rtol=1e-4)
    xt = torch.randn_like(target)
    torch.testing.assert_close(predict_x1(xt, torch.tensor([0.5]), Exact()(xt, torch.tensor([0.5]), 0, 0, 0)), target)


def test_distiller_step():
    teacher, student = dit(), dit()
    distiller = ConsistencyFlowDistiller(teacher, student, num_scales=8)
    x, _, labels, parent, valid = batch()
    loss = distiller(x, parent, labels, valid)
    loss.backward()
    assert torch.isfinite(loss) and float(loss.detach()) >= 0
    assert all(p.grad is None for p in distiller.teacher.parameters())
    assert any(p.grad is not None for p in distiller.student.parameters())
    before = next(distiller.ema_student.parameters()).clone()
    with torch.no_grad():
        for p in distiller.student.parameters():
            p.add_(1.0)
    distiller.update_ema(0.5)
    assert not torch.equal(before, next(distiller.ema_student.parameters()))
    d = torch.zeros(2, 3, C, T)
    assert torch.allclose(pseudo_huber(d), torch.zeros(2, 3))


def test_toy_vae_wrapper_roundtrip_and_chunking():
    vae = DACVAEWrapper(vae=ToyVAE(8000, 80, 80), chunk_seconds=0.2, margin_seconds=0.05)
    audio = torch.randn(2, 2, 8000)
    z = vae.encode(audio)
    assert z.shape == (2, 160, 100)
    torch.testing.assert_close(vae.decode(z, length=8000), audio, atol=1e-4, rtol=1e-4)
    whole = DACVAEWrapper(vae=ToyVAE(8000, 80, 80), chunk_seconds=100)
    torch.testing.assert_close(whole.encode(audio), z)
    up = vae.encode(audio[0], sample_rate=16000)                 # resampled on the way in
    assert up.shape == (160, 50) and vae.num_frames(8000, 16000) == 50
    assert vae.silence_latent().shape == (160,)
    assert build_vae("toy:8000:80:8").latent_dim == 8


def test_wrapper_uses_posterior_mean_of_dacvae_like_models():
    class Quantizer(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj = nn.Conv1d(4, 2 * 3, 1)
            self.out_proj = nn.Conv1d(3, 4, 1)

    class DacLike(nn.Module):
        sample_rate, hop_length, latent_dim = 1000, 10, 3

        def __init__(self):
            super().__init__()
            self.encoder = nn.Conv1d(1, 4, 10, stride=10)
            self.quantizer = Quantizer()
            self.decoder = nn.ConvTranspose1d(4, 1, 10, stride=10)

        def _pad(self, x):
            return nn.functional.pad(x, (0, (-x.shape[-1]) % 10), mode="reflect")

        def encode(self, x):
            raise AssertionError("the sampled encode must not be used when sample=False")

        def decode(self, z):
            return self.decoder(self.quantizer.out_proj(z))

    inner = DacLike()
    wrapper = DACVAEWrapper(vae=inner)
    audio = torch.randn(1, 2, 500)
    z = wrapper.encode(audio)
    mean = inner.quantizer.in_proj(inner.encoder(audio.reshape(2, 1, 500))).chunk(2, dim=1)[0]
    torch.testing.assert_close(z, mean.reshape(1, 6, 50))
    assert wrapper.decode(z).shape == (1, 2, 500)


def test_latent_normalizer():
    norm = LatentNormalizer(torch.tensor([1.0, -1.0]), torch.tensor([2.0, 0.5]))
    x = torch.randn(3, 2, 7)
    torch.testing.assert_close(norm.denormalize(norm.normalize(x)), x)
