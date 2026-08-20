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
