#!/usr/bin/env python3
"""activations.py — registry of custom activations for NNI-Remake models.

All activations are SHAPE-PRESERVING modules so they drop into any block chain
(2-D conv, 1-D, or dense) without changing the channel / feature width. A
proposal uses one via a block of type "activation" with a "name":

    {"type": "activation", "name": "swiglu"}

Registered (builder `make(name, in_ch) -> nn.Module`):
  relu, relu6, silu, elu, selu, prelu   : torch natives (prelu is learnable)
  gelu                                   : Gaussian-error LU (torch)
  mish                                   : Mish (Misra 2019), self-gated
  snake                                  : Snake (Ziyin et al. 2020), periodic+learnable
  swiglu                                 : channel-preserving SwiGLU-style gated unit
                                           out = SiLU(Linear_c(x)) * Linear_c(x)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

REGISTRY = {}


def register(name):
    """Builder decorator: register an activation builder under `name`."""
    def _(f):
        REGISTRY[name] = f
        return f
    return _


@register("relu")
def _relu(in_ch):
    return nn.ReLU()


@register("relu6")
def _relu6(in_ch):
    return nn.ReLU6()


@register("silu")
def _silu(in_ch):
    return nn.SiLU()


@register("elu")
def _elu(in_ch):
    return nn.ELU()


@register("selu")
def _selu(in_ch):
    return nn.SELU()


@register("prelu")
def _prelu(in_ch):
    return nn.PReLU()


@register("gelu")
def _gelu(in_ch):
    return nn.GELU()


@register("mish")
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


@register("snake")
class Snake(nn.Module):
    """Snake activation (Ziyin et al. 2020): x + (1/a) * sin^2(a x), a learnable."""

    def __init__(self, in_ch=None):
        super().__init__()
        # one learnable frequency per channel (or scalar if in_ch unknown)
        self.a = nn.Parameter(torch.ones(1 if in_ch is None else in_ch))

    def forward(self, x):
        # broadcast over the (optional) channel dim
        a = self.a.reshape(1, -1, *([1] * (x.dim() - 2))) if x.dim() > 2 else self.a
        return x + (1.0 / a) * (torch.sin(a * x) ** 2)


@register("swiglu")
class SwiGLU(nn.Module):
    """Channel-preserving SwiGLU-style gated linear unit.

    out = SiLU(Linear_c(x)) * Linear_c(x); preserves the channel/feature width.
    Behaves as a shape-preserving nonlinear activation in a block chain.
    """

    def __init__(self, in_ch):
        super().__init__()
        self.u = nn.Linear(in_ch, in_ch)
        self.v = nn.Linear(in_ch, in_ch)

    def forward(self, x):
        return F.silu(self.u(x)) * self.v(x)



@register("hardtanh")
def _hardtanh(in_ch):
    return nn.Hardtanh()

def make(name, in_ch):
    """Resolve an activation NAME -> module. Raises on unknown name."""
    if name not in REGISTRY:
        raise KeyError(f"custom activation {name!r} not in activations.REGISTRY "
                       f"(have: {sorted(REGISTRY)})")
    return REGISTRY[name](in_ch)
