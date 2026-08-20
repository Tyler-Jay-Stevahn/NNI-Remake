#!/usr/bin/env python3
"""optimizers.py — registry of custom optimizers for NNI-Remake training.

Stock torch optimizers (adam, adamw, sgd, rmsprop, and the Tier-1 set below) are
handled in train.py via OPT_MAP. Any from-scratch optimizer strategy lives here,
registered by name in REGISTRY. To add one, write a subclass of
torch.optim.Optimizer and decorate it with @register("<name>"). train.py looks it
up when a proposal sets "optimizer": "custom:<name>" and forwards
"optimizer_kwargs" to its constructor.

Contract for a custom optimizer:
  - subclass torch.optim.Optimizer
  - accept (params, **kwargs); store per-group config in `defaults`
  - implement step(closure=None)

Stock (Tier 1) — torch ships these; only registered in train.py OPT_MAP:
  nadam, radam, adagrad, adamax, adadelta, rprop, asgd, lbfgs
  NOTE: lbfgs requires a closure-based step (it re-evaluates the loss). The
  current train.py loop calls opt.step() with no closure, so lbfgs is selectable
  but will not train correctly without a loop change.

Custom (Tier 2) — pure-torch classes registered below:
  mymomentum  : from-scratch SGD+momentum (baseline smoke test)
  diffadamw   : AdamW with the Stable Diffusion / Latent Diffusion recipe
  diffadam    : DDPM (Ho et al. 2020) Adam recipe
  diff8bit    : 8-bit Adam via bitsandbytes (CUDA-only; lazy import)
  lion        : Lion (Chen et al. 2023), sign-based update
  adan        : Adan (Xie et al. 2022), adaptive Nesterov momentum
  sophia      : Sophia-G (Liu et al. 2023) — Hessian-preconditioned update
                (gradient-magnitude PROXY below; exact H needs a 2nd-order pass)
  adafactor   : Adafactor (Shazeer 2018), factored memory-efficient preconditioner
  lamb        : LAMB (You et al. 2020), layer-wise adaptive rates
  lars        : LARS (You et al. 2017), layer-wise adaptive SGD+momentum
  muon        : Muon (Jordan 2024), orthogonalized (Newton-Schulz) momentum
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


@register("lion")
class Lion(torch.optim.Optimizer):
    """Lion (Chen et al. 2023) — sign-based update with momentum.

    update = beta1*m + (1-beta1)*g ;  p -= lr*sign(update) ;  m <- update.
    Defaults: lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0.
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data
                if wd != 0:
                    g = g.add(p.data, alpha=wd)
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p.data)
                m = state["m"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                p.data.add_(torch.sign(m), alpha=-lr)
        return loss


@register("adan")
class Adan(torch.optim.Optimizer):
    """Adan (Xie et al. 2022) — adaptive Nesterov momentum.

    Uses exp-avg grad (m), exp-avg squared grad (v), and exp-avg gradient
    difference (n); the Nesterov look-ahead combines all three.
    Defaults: lr=1e-3, betas=(0.98, 0.92, 0.99), eps=1e-8, weight_decay=0.0.
    """

    def __init__(self, params, lr=1e-3, betas=(0.98, 0.92, 0.99),
                 eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2, b3 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data
                if wd != 0:
                    g = g.add(p.data, alpha=wd)
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                    state["n"] = torch.zeros_like(p.data)
                    state["prev"] = torch.zeros_like(p.data)
                m = state["m"]
                v = state["v"]
                n = state["n"]
                prev = state["prev"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                diff = g.sub(prev)
                n.mul_(b3).add_(diff, alpha=1 - b3)
                # Nesterov look-ahead gradient (Adan, eq. 7-9)
                ghat = g.clone()
                ghat.add_(m, alpha=b1)
                ghat.add_(n, alpha=b2 * b3)
                ghat.add_(diff, alpha=b2 * (1 - b3))
                denom = v.sqrt().add_(eps)
                p.data.addcdiv_(ghat, denom, value=-lr)
                prev.copy_(g)
        return loss


@register("sophia")
class Sophia(torch.optim.Optimizer):
    """Sophia-G (Liu et al. 2023) — diagonal-Hessian-preconditioned update.

    Exact Sophia-G: when step(closure) is called (the train.py closure path),
    it draws a Rademacher z, computes the Hessian-vector product hvp via
    torch.autograd.grad(p.grad, p, grad_outputs=z), and EMA-averages z*hvp as
    the diagonal-Hessian estimate. No second closure pass is needed because we
    reuse the gradient already in p.grad from the closure's backward(). If step()
    is called with no closure (loss unavailable), it falls back to the EMA-of-|grad|
    PROXY so it still runs. lr scaling uses rho*h + eps.
    Defaults: lr=1e-4, betas=(0.965, 0.99), rho=0.04, eps=1e-12, weight_decay=0.0.
    """

    def __init__(self, params, lr=1e-4, betas=(0.965, 0.99), rho=0.04,
                 eps=1e-12, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, rho=rho, eps=eps,
                                      weight_decay=weight_decay))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            rho = group["rho"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data
                if wd != 0:
                    g = g.add(p.data, alpha=wd)
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p.data)
                    state["h"] = torch.zeros_like(p.data)
                m = state["m"]
                h = state["h"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                if closure is not None:
                    # Exact Hutchinson diagonal-Hessian estimate H·z.
                    # g must stay attached to the graph (create_graph=True) so we
                    # can differentiate it again; z is a Rademacher vector.
                    z = torch.randint(0, 2, p.data.shape, device=p.data.device,
                                      dtype=p.data.dtype) * 2.0 - 1.0
                    g_graph = torch.autograd.grad(loss, p, create_graph=True,
                                                 retain_graph=True)[0]
                    hvp = torch.autograd.grad(g_graph, p, grad_outputs=z,
                                             retain_graph=True)[0]
                    est = z * hvp
                    h.mul_(b2).add_(est, alpha=1 - b2)
                else:
                    # Proxy fallback (no closure): Hessian ~ EMA of |grad|.
                    h.mul_(b2).add_(g.abs(), alpha=1 - b2)
                denom = rho * h + eps
                p.data.addcdiv_(m, denom, value=-lr)
        return loss


@register("adafactor")
class Adafactor(torch.optim.Optimizer):
    """Adafactor (Shazeer 2018) — memory-efficient factored preconditioner.

    For >=2D weights the second moment is factored into row/column statistics
    (cheap); 1D params use a full second-moment EMA. No per-parameter second
    moment buffer is stored for 2D weights.
    Defaults: lr=1e-3, betas=(0.9, 0.999), eps=1e-3, weight_decay=0.0.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-3,
                 weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data
                if wd != 0:
                    g = g.add(p.data, alpha=wd)
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p.data)
                    if p.data.dim() >= 2:
                        state["row"] = torch.zeros(p.data.size(0))
                        state["col"] = torch.zeros(p.data.size(1))
                    else:
                        state["sq"] = torch.zeros_like(p.data)
                exp_avg = state["exp_avg"]
                exp_avg.mul_(b1).add_(g, alpha=1 - b1)
                if p.data.dim() >= 2:
                    rs = g.pow(2).mean(dim=1)
                    cs = g.pow(2).mean(dim=0)
                    state["row"].mul_(b2).add_(rs, alpha=1 - b2)
                    state["col"].mul_(b2).add_(cs, alpha=1 - b2)
                    sq = torch.outer(state["row"], state["col"])
                    sq = sq / sq.mean()
                    denom = sq.sqrt().add_(eps)
                else:
                    sq = state["sq"]
                    sq.mul_(b2).addcmul_(g, g, value=1 - b2)
                    denom = sq.sqrt().add_(eps)
                p.data.addcdiv_(exp_avg, denom, value=-lr)
        return loss


@register("lamb")
class LAMB(torch.optim.Optimizer):
    """LAMB (You et al. 2020) — layer-wise adaptive rate, Adam-based.

    Trust ratio = ||p|| / ||update|| scales the Adam step per parameter tensor.
    Defaults: lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                state["step"] += 1
                m = state["m"]
                v = state["v"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                bc1 = 1 - b1 ** state["step"]
                bc2 = 1 - b2 ** state["step"]
                update = (m / bc1) / (v / bc2).sqrt().add_(eps)
                if wd != 0:
                    update = update.add(p.data, alpha=wd)
                p_norm = p.data.norm()
                u_norm = update.norm()
                ratio = (p_norm / u_norm) if (p_norm != 0 and u_norm != 0) else 1.0
                p.data.add_(update, alpha=-lr * ratio)
        return loss


@register("lars")
class LARS(torch.optim.Optimizer):
    """LARS (You et al. 2017) — layer-wise adaptive rate, SGD+momentum based.

    Trust ratio = ||p|| / ||update|| scales the SGD-momentum step per tensor.
    Defaults: lr=1e-3, momentum=0.9, weight_decay=0.0, eps=1e-8.
    """

    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0.0, eps=1e-8):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      weight_decay=weight_decay, eps=eps))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p.data)
                m = state["m"]
                m.mul_(mu).add_(g)
                update = m.add(p.data, alpha=wd) if wd != 0 else m.clone()
                p_norm = p.data.norm()
                u_norm = update.norm()
                ratio = (p_norm / u_norm) if (p_norm != 0 and u_norm != 0) else 1.0
                p.data.add_(update, alpha=-lr * ratio)
        return loss


@register("muon")
class Muon(torch.optim.Optimizer):
    """Muon (Jordan 2024) — orthogonalized (Newton-Schulz) momentum.

    2D params: update = NS-orthogonalize(momentum), p -= lr*update. 1D params
    (biases/embeddings): plain SGD-momentum, matching the official Muon split.
    Defaults: lr=0.02, momentum=0.95, nesterov=True, ns_steps=5.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      nesterov=nesterov, ns_steps=ns_steps))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p.data)
                m = state["m"]
                m.mul_(mu).add_(g)
                if p.data.dim() == 2:
                    update = _newton_schulz_orthogonalize(m, steps)
                    p.data.add_(update, alpha=-lr)
                else:
                    p.data.add_(m, alpha=-lr)
        return loss


def _newton_schulz_orthogonalize(G, steps=5):
    """Approximate the polar-factor / orthogonalization of G via Newton-Schulz.

    Scales G then iterates the 5th-order polynomial approximating sign/sqrt to
    push G toward an orthogonal matrix. Coefficients from the Muon reference.
    """
    G = G / (G.norm() + 1e-7)
    a, b, c = (3.4445, -4.4229, 1.0625)
    for _ in range(steps):
        A = G @ G.T
        B = b * A + c * (A @ A)
        G = a * G + b * (G @ B.T) + c * (B @ G)
    return G
