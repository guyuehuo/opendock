import os

import pytest
import torch

from opendock.scorer.composite import _flat_bottom


def test_flat_bottom_zero_below_dmin():
    d = torch.tensor([0.0, 3.0, 4.0, 6.0])
    out = _flat_bottom(d, dmin=4.0, exponent=2.0)
    assert torch.allclose(out, torch.tensor([0.0, 0.0, 0.0, 4.0]))


def test_flat_bottom_default_is_raw_distance():
    d = torch.tensor([1.5, 2.5])
    assert torch.allclose(_flat_bottom(d), d)


def test_flat_bottom_differentiable():
    d = torch.tensor([5.0], requires_grad=True)
    _flat_bottom(d, dmin=4.0, exponent=2.0).sum().backward()
    assert d.grad is not None and float(d.grad) != 0.0
