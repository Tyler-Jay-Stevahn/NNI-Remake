#!/usr/bin/env python3
"""normalizations.py — registry of custom normalization layers.

A proposal uses one via a block of type "norm" with a "name":

    {"type": "norm", "name": "groupnorm"}
    {"type": "norm", "name": "rmsnorm"}

`make(name, in_ch, ndims)` returns the right module for the calling context:
  ndims == 2  -> 2-D conv chain  (BatchNorm2d / GroupNorm / InstanceNorm2d)
  ndims == 1  -> 1-D chain       (BatchNorm1d / GroupNorm / InstanceNorm1d)
  ndims == 0  -> dense / tabular (LayerNorm / BatchNorm1d)

Registered:
  layernorm    : channel/feature norm (GroupNorm(1) for conv, LayerNorm for dense)
  groupnorm    : GroupNorm(min(32,in_ch), in_ch)
  batchnorm    : BatchNorm for the modality
  instancenorm : InstanceNorm for the modality
  rmsnorm      : channel-wise RMSNorm (learned scale), works for any ndims
"""

import torch
import torch.nn as nn

REGISTRY = {}


def register(name):
    """Builder decorator: register a norm builder under `name`."""
    def _(f):
        REGISTRY[name] = f
        return f
    return _


class RMSNorm(nn.Module):
    """Channel-wise RMSNorm: normalize over the channel dim, learned gain.

    Works for 2-D (B,C,H,W), 1-D (B,C,T), and dense (B,F) by normalizing over
    dimension 1 (the channel/feature dim).
    """

    def __init__(self, in_ch):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(in_ch))

    def forward(self, x):
        # normalize over the channel dim (1)
        rms = x.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
        x = x / rms
        # apply gain per channel
        shape = [1, -1] + [1] * (x.dim() - 2)
        return x * self.gain.reshape(shape)


@register("layernorm")
def _layernorm(in_ch, ndims):
    if ndims == 0:
        return nn.LayerNorm(in_ch)
    return nn.GroupNorm(1, in_ch)


@register("groupnorm")
def _groupnorm(in_ch, ndims):
    groups = min(32, max(1, in_ch))
    return nn.GroupNorm(groups, in_ch)


@register("batchnorm")
def _batchnorm(in_ch, ndims):
    if ndims == 2:
        return nn.BatchNorm2d(in_ch)
    if ndims == 1:
        return nn.BatchNorm1d(in_ch)
    return nn.BatchNorm1d(in_ch)


@register("instancenorm")
def _instancenorm(in_ch, ndims):
    if ndims == 2:
        return nn.InstanceNorm2d(in_ch, affine=True)
    if ndims == 1:
        return nn.InstanceNorm1d(in_ch, affine=True)
    return nn.LayerNorm(in_ch)


@register("rmsnorm")
def _rmsnorm(in_ch, ndims):
    return RMSNorm(in_ch)


def make(name, in_ch, ndims):
    """Resolve a norm NAME -> module for the given context. Raises if unknown."""
    if name not in REGISTRY:
        raise KeyError(f"custom norm {name!r} not in normalizations.REGISTRY "
                       f"(have: {sorted(REGISTRY)})")
    return REGISTRY[name](in_ch, ndims)
