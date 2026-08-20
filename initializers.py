#!/usr/bin/env python3
"""initializers.py — registry of weight initializers for NNI-Remake models.

Applied to the whole model after build_model (train.py calls apply(name, model)
when a proposal sets "initializer": "custom:<name>"). Each initializer resets
the Linear/Conv weights and zeros biases (BatchNorm weights -> 1, biases -> 0).

Registered:
  xavier_uniform, xavier_normal   : Glorot (Xavier) init
  kaiming_uniform, kaiming_normal : He (Kaiming) init
  lecun_normal                    : LeCun normal (SELU-friendly)
  orthogonal                      : orthogonal init
"""

import torch
import torch.nn as nn

REGISTRY = {}


def register(name):
    """Initializer decorator: register under `name`."""
    def _(f):
        REGISTRY[name] = f
        return f
    return _


def _walk(model, init_weight, init_bias=None):
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            init_weight(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                            nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d,
                            nn.InstanceNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


@register("xavier_uniform")
def xavier_uniform(model):
    _walk(model, lambda w: nn.init.xavier_uniform_(w))


@register("xavier_normal")
def xavier_normal(model):
    _walk(model, lambda w: nn.init.xavier_normal_(w))


@register("kaiming_uniform")
def kaiming_uniform(model):
    _walk(model, lambda w: nn.init.kaiming_uniform_(w, nonlinearity="relu"))


@register("kaiming_normal")
def kaiming_normal(model):
    _walk(model, lambda w: nn.init.kaiming_normal_(w, nonlinearity="relu"))


@register("lecun_normal")
def lecun_normal(model):
    _walk(model, lambda w: nn.init.lecun_normal_(w))


@register("orthogonal")
def orthogonal(model):
    _walk(model, lambda w: nn.init.orthogonal_(w))


def apply(name, model):
    """Apply a custom initializer NAME to `model`. Raises if unknown."""
    if name not in REGISTRY:
        raise KeyError(f"custom initializer {name!r} not in initializers.REGISTRY "
                       f"(have: {sorted(REGISTRY)})")
    REGISTRY[name](model)
