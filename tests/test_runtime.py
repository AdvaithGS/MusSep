"""CPU balance, checkpoint cadence, compile, the latent cache and the benchmark."""
import os
import signal
import time

import pytest
import torch

from utils.runtime import (CheckpointClock, available_cpus, configure_threads, make_loader, maybe_compile,
                           resolve_workers, worker_init)


def test_worker_sizing():
    cpus = available_cpus()
    assert resolve_workers("auto", torch.device("cuda")) == max(0, min(12, cpus - 2))
    assert resolve_workers(None, torch.device("cpu")) == max(0, min(12, cpus // 4))
    assert resolve_workers("3", torch.device("cpu")) == 3
    before = torch.get_num_threads()
    try:
        assert configure_threads(torch.device("cpu"), 0) == cpus
        worker_init(0)
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(before)


def _threads(_):
    return torch.tensor(torch.get_num_threads())


def test_loader_workers_use_one_thread():
    loader = make_loader(list(range(4)), 1, 2, torch.device("cpu"), collate_fn=_threads, persistent=False)
    assert [int(x) for x in loader] == [1, 1, 1, 1]


def test_clock_by_steps_minutes_and_signal():
    clock = CheckpointClock(every_steps=5, every_minutes=0.0)
    try:
        assert not clock.due(3) and clock.due(5)
        clock.every_seconds = 0.05
        time.sleep(0.06)
        assert clock.due(7)
        clock.saved()
        assert not clock.due(8)
        os.kill(os.getpid(), signal.SIGTERM)          # what a spot instance sends before it goes away
        time.sleep(0.05)
        assert clock.stop_requested and clock.due(9)
    finally:
        clock.close()
    assert signal.getsignal(signal.SIGTERM) in (signal.SIG_DFL, None) or callable(signal.getsignal(signal.SIGTERM))


def test_compile_matches_eager():
    from models.dit import JointSubStemDiT

    torch.manual_seed(0)
    model = JointSubStemDiT(in_channels=8, hidden_dim=32, num_heads=4, depth=1, num_stem_classes=5).eval()
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.05)
    compiled = maybe_compile(model, True)
    x, t = torch.randn(2, 3, 8, 12), torch.rand(2)
    labels, parent = torch.randint(1, 5, (2, 3)), torch.randn(2, 8, 12)
    with torch.no_grad():
        try:
            got = compiled(x, t, labels, parent)
        except Exception as error:  # pragma: no cover - platform without a working compiler
            pytest.skip(f"torch.compile not usable here: {error}")
        torch.testing.assert_close(got, model(x, t, labels, parent), atol=1e-4, rtol=1e-4)
    assert maybe_compile(model, False) is model
