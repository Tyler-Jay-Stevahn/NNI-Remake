#!/usr/bin/env python3
"""optimizers.py — registry of custom optimizers for NNI-Remake training.

Stock torch optimizers (adam, adamw, sgd, rmsprop) are handled in train.py via
OPT_MAP. Any from-scratch optimizer strategy lives here, registered by name in
REGISTRY. To add one, write a subclass of torch.optim.Optimizer and decorate it
with @register("<name>"). train.py looks it up when a proposal sets
"optimizer": "custom:<name>" and forwards "optimizer_kwargs" to its constructor.

Contract for a custom optimizer:
  - subclass torch.optim.Optimizer
  - accept (params, **kwargs); store per-group config in `defaults`
  - implement step(closure=None)

Three diffusion-model recipes are included:
  - diffadamw  : AdamW with the Stable Diffusion / Latent Diffusion recipe
                 (lr=1e-4, betas=(0.9,0.999), eps=1e-8, weight_decay=1e-4). Pure torch.
  - diffadam   : DDPM (Ho et al. 2020) Adam recipe (lr=1e-4, betas=(0.9,0.999),
                 no weight decay). Pure torch.
  - diff8bit   : 8-bit Adam from bitsandbytes (the memory-efficient optimizer used
                 by Stable Diffusion / LoRA finetunes). CUDA-only; requires the
                 'bitsandbytes' package, imported lazily so this module stays
                 importable on CPU-only / no-dep hosts.
"""

import torch

# name -> optimizer class
REGISTRY = {}


def register(name):
    """Class decorator: register a custom optimizer under `name`."""
    def _(cls):
        REGISTRY[name] = cls
        return cls
    return _


@register("mymomentum")
class MyMomentum(torch.optim.Optimizer):
    """Example custom optimizer: SGD + momentum, lr taken from the proposal.

    Demonstrates the custom-optimizer contract end to end. Replace or extend
    with real strategies as needed.
    """

    def __init__(self, params, lr=1e-3, momentum=0.9):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(p.data)
                buf = state["buf"]
                buf.mul_(mu).add_(g)
                p.data.add_(buf, alpha=-lr)
        return loss


@register("diffadamw")
class DiffAdamW(torch.optim.AdamW):
    """AdamW with the Stable Diffusion / Latent Diffusion recipe baked in.

    Defaults: lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4.
    train.py passes lr from the proposal; override any of these via the
    proposal's "optimizer_kwargs" (e.g. {"weight_decay": 0.05}).
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=1e-4):
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay)


@register("diffadam")
class DiffAdam(torch.optim.Adam):
    """DDPM (Ho et al. 2020) Adam recipe.

    Defaults: lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0.
    train.py passes lr from the proposal; override via "optimizer_kwargs".
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay)


@register("diff8bit")
class Diff8BitAdam(torch.optim.Optimizer):
    """8-bit Adam from bitsandbytes — the memory-efficient optimizer used by
    Stable Diffusion / LoRA finetunes. CUDA-only; requires the 'bitsandbytes'
    package. Import is lazy so this module still imports on CPU-only hosts; the
    missing dependency only surfaces when this optimizer is actually constructed.
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.0):
        try:
            import bitsandbytes as bnb  # lazy: keep module importable w/o dep
        except ImportError:
            raise SystemExit(
                "custom optimizer 'diff8bit' needs the 'bitsandbytes' package "
                "(pip install bitsandbytes) and a CUDA GPU; it is not available here.")
        # Materialize params ONCE. model.parameters() is a generator; the
        # parent __init__ does list(params) internally and exhausts it, so
        # handing the same iterator to bnb.optim.Adam8bit yields "empty
        # parameter list". Share one concrete list with both consumers.
        params = list(params)
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay))
        self._inner = bnb.optim.Adam8bit(params, lr=lr, betas=betas,
                                         eps=eps, weight_decay=weight_decay)
        # Delegate to bitsandbytes' real param_groups / state / defaults.
        self.param_groups = self._inner.param_groups
        self.state = self._inner.state
        self.defaults = self._inner.defaults

    def step(self, closure=None):
        return self._inner.step(closure)
