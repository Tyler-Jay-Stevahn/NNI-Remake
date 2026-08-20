#!/usr/bin/env python3
"""regularizations.py — registry of regularization techniques.

Two forms, mirroring how regularizers appear in real pipelines:

1) LAYER regularizers — used as a block of type "regularize" with a "name":
       {"type": "regularize", "name": "dropout2d", "p": 0.2}
   These preserve the tensor shape and sit directly in the block chain.

2) WEIGHT regularizers — applied to the whole model after build (train.py),
   driven by a proposal's "regularization" list. Currently:
       "spectral_norm"  -> torch.nn.utils.spectral_norm on every Linear/Conv

Registered layer regularizers:
  dropout      : nn.Dropout (dense / 1-D / 2-D elementwise)
  dropout2d    : nn.Dropout2d (spatial — Srivastava et al. 2014)
  alphadropout : nn.AlphaDropout (self-normalizing, pairs with SELU)
  dropout1d    : nn.Dropout1d (channel-wise 1-D)
"""

import torch
import torch.nn as nn

REGISTRY = {}


def register(name):
    """Layer-regularizer builder decorator: register under `name`."""
    def _(f):
        REGISTRY[name] = f
        return f
    return _


@register("dropout")
def _dropout(in_ch, **kw):
    return nn.Dropout(kw.get("p", 0.1))


@register("dropout1d")
def _dropout1d(in_ch, **kw):
    return nn.Dropout1d(kw.get("p", 0.1))


@register("dropout2d")
def _dropout2d(in_ch, **kw):
    return nn.Dropout2d(kw.get("p", 0.1))


@register("alphadropout")
def _alphadropout(in_ch, **kw):
    return nn.AlphaDropout(kw.get("p", 0.1))


def make_regularizer(name, in_ch, **kw):
    """Resolve a layer regularizer NAME -> module. Raises if unknown."""
    if name not in REGISTRY:
        raise KeyError(
            f"custom regularizer {name!r} not in regularizations.REGISTRY "
            f"(have layer: {sorted(REGISTRY)}; also: spectral_norm)")
    return REGISTRY[name](in_ch, **kw)


def apply_spectral_norm(model):
    """Wrap every Linear/Conv in the model with spectral normalization."""
    from torch.nn.utils import spectral_norm

    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            spectral_norm(m)
    return model
