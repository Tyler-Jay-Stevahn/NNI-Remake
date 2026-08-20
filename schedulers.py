#!/usr/bin/env python3
"""schedulers.py — registry of custom LR schedulers for NNI-Remake training.

A scheduler is a pure function `f(step, total_steps, **kwargs) -> float` that
returns an LR *factor* (multiplied by the base LR from the proposal). train.py
calls it every batch and sets each param-group's LR to `base_lr * factor`.

This keeps schedulers stateless and easy to drop into the per-batch loop
without a torch.optim.lr_scheduler object. Custom optimizers that manage their
own LR (e.g. D-Adaptation Adam, flagged with `.manages_lr = True`) are exempt
from scheduler scaling in train.py.

Registered:
  constant        : factor 1.0
  cosine_anneal   : 1 -> 0 cosine decay over total_steps
  cosine_warmup   : linear warmup -> cosine decay to `min_factor`
  onecycle        : linear warmup to 1 -> cosine decay to `min_factor`
  wsd             : warmup -> stable -> decay (Warmup-Stable-Decay, Hu et al. 2024)
  step_decay      : drop by `gamma` every `step_size` steps
"""

import math

REGISTRY = {}


def register(name):
    """Function decorator: register a scheduler under `name`."""
    def _(f):
        REGISTRY[name] = f
        return f
    return _


def _warmup_steps(total, warmup):
    if warmup is None:
        return max(1, int(0.1 * total))
    if warmup >= 1:
        return max(1, int(warmup))
    return max(1, int(warmup * total))


@register("constant")
def constant(step, total_steps, **kw):
    return 1.0


@register("cosine_anneal")
def cosine_anneal(step, total_steps, min_factor=0.0, **kw):
    if total_steps <= 1:
        return 1.0
    prog = min(1.0, step / max(1, total_steps - 1))
    return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * prog))


@register("cosine_warmup")
def cosine_warmup(step, total_steps, warmup=None, min_factor=0.0, **kw):
    w = _warmup_steps(total_steps, warmup)
    if step < w:
        return (step + 1) / w
    if total_steps > w:
        prog = min(1.0, (step - w) / max(1, total_steps - w))
        return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * prog))
    return 1.0


@register("onecycle")
def onecycle(step, total_steps, warmup=None, min_factor=0.001, **kw):
    w = _warmup_steps(total_steps, warmup)
    if step < w:
        return (step + 1) / w
    if total_steps > w:
        prog = min(1.0, (step - w) / max(1, total_steps - w))
        return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * prog))
    return 1.0


@register("wsd")
def wsd(step, total_steps, warmup_frac=0.1, decay_frac=0.1, min_factor=0.0, **kw):
    """Warmup-Stable-Decay (Hu et al. 2024).

    Linear warmup over `warmup_frac` of training, constant `stable` phase, then
    decay over the final `decay_frac` (cosine to `min_factor`).
    """
    w = max(1, int(warmup_frac * total_steps))
    d = max(1, int(decay_frac * total_steps))
    if step < w:
        return (step + 1) / w
    if step >= total_steps - d:
        prog = min(1.0, (step - (total_steps - d)) / max(1, d))
        return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * prog))
    return 1.0


@register("step_decay")
def step_decay(step, total_steps, step_size=1000, gamma=0.5, **kw):
    return gamma ** (step // max(1, step_size))


def get(name):
    """Resolve a scheduler NAME -> function. Raises on unknown name."""
    if name not in REGISTRY:
        raise KeyError(f"custom scheduler {name!r} not in schedulers.REGISTRY "
                       f"(have: {sorted(REGISTRY)})")
    return REGISTRY[name]
