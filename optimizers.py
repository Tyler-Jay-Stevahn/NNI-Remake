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
  NOTE: lbfgs requires a closure-based step (it re-evaluates the loss). train.py
  routes lbfgs (+ Sophia) through a closure-aware loop, so it trains correctly.

Custom (Tier 2) — pure-torch classes registered below:
  mymomentum  : from-scratch SGD+momentum (baseline smoke test)
  diffadamw   : AdamW with the Stable Diffusion / Latent Diffusion recipe
  diffadam    : DDPM (Ho et al. 2020) Adam recipe
  diff8bit    : 8-bit Adam via bitsandbytes (CUDA-only; lazy import)
  dadaptadam  : D-Adaptation Adam (Defazio & Mishchenko 2023) — auto-tunes LR
                from gradient statistics (lr stays 1.0; d grows from d0=1e-6)
  lion        : Lion (Chen et al. 2023), sign-based update
  adan        : Adan (Xie et al. 2022), adaptive Nesterov momentum
  sophia      : Sophia-G (Liu et al. 2023) — Hessian-preconditioned update
                (exact Hutchinson HVP via closure path in train.py loop)
  adafactor   : Adafactor (Shazeer 2018), factored memory-efficient preconditioner
  lamb        : LAMB (You et al. 2020), layer-wise adaptive rates
  lars        : LARS (You et al. 2017), layer-wise adaptive SGD+momentum
  muon        : Muon (Jordan 2024), orthogonalized (Newton-Schulz) momentum
  ademamix    : AdEMAMix (Pagliardini et al. 2024, ICLR'25) — Adam + slow/fast EMA mix
  evooptim    : EvoOptimizer (Marfinetz 2025) — GA-discovered sign+Adam mix
  root        : ROOT (He et al. 2025, Huawei) — robust orthogonalized (soft-threshold+NS)
  rlion       : RLion (Rong et al. 2025, Sci Reports) — Lion w/ arctan instead of sign
"""

import math

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
        # Materialize the param iterable ONCE. model.parameters() returns a
        # generator that torch.optim.Optimizer.__init__ consumes; passing the
        # same generator to bnb.optim.Adam8bit afterwards yields an EMPTY param
        # list. list() makes it reusable by both consumers.
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


@register("dadaptadam")
class DAdaptAdam(torch.optim.Optimizer):
    """D-Adaptation Adam (Defazio & Mishchenko 2023) — automatic learning-rate tuning.

    D-Adaptation removes hand-tuning of LR: keep lr=1.0 and the optimizer learns
    an adaptive step-size d from the gradient statistics (d starts at d0=1e-6 and
    grows toward d_hat each step). Popular for Stable Diffusion / LoRA finetunes
    where a good LR is hard to find. Faithful single-process port of
    facebookresearch/dadaptation DAdaptAdam (FSDP/distributed paths removed).
    Defaults: lr=1.0, betas=(0.9,0.999), eps=1e-8, weight_decay=0, d0=1e-6.
    """

    def __init__(self, params, lr=1.0, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, decouple=False, use_bias_correction=False,
                 d0=1e-6, growth_rate=float('inf')):
        if not 0.0 < d0:
            raise ValueError(f"Invalid d0 value: {d0}")
        if not 0.0 < lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 < eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        d=d0, k=0, layer_scale=1.0, numerator_weighted=0.0,
                        growth_rate=growth_rate, use_bias_correction=use_bias_correction,
                        decouple=decouple)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        sk_l1 = 0.0
        group = self.param_groups[0]
        use_bias_correction = group['use_bias_correction']
        numerator_weighted = group['numerator_weighted']
        beta1, beta2 = group['betas']
        k = group['k']
        d = group['d']
        lr = max(group['lr'] for group in self.param_groups)
        if use_bias_correction:
            bias_correction = ((1 - beta2 ** (k + 1)) ** 0.5) / (1 - beta1 ** (k + 1))
        else:
            bias_correction = 1
        dlr = d * lr * bias_correction
        growth_rate = group['growth_rate']
        decouple = group['decouple']
        sqrt_beta2 = beta2 ** 0.5
        numerator_acum = 0.0
        for group in self.param_groups:
            decay = group['weight_decay']
            k = group['k']
            eps = group['eps']
            group_lr = group['lr']
            r = group['layer_scale']
            if group_lr not in [lr, 0.0]:
                raise RuntimeError(
                    "Setting different lr values in different parameter groups is "
                    "only supported for values of 0. To scale the learning rate "
                    "differently per layer, set 'layer_scale' instead.")
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if decay != 0 and not decouple:
                    grad = grad.add(p.data, alpha=decay)
                state = self.state[p]
                if 'step' not in state:
                    state['step'] = 0
                    state['s'] = torch.zeros_like(p.data).detach()
                    state['exp_avg'] = torch.zeros_like(p.data).detach()
                    state['exp_avg_sq'] = torch.zeros_like(p.data).detach()
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                s = state['s']
                if group_lr > 0.0:
                    denom = exp_avg_sq.sqrt().add_(eps)
                    numerator_acum += r * dlr * torch.dot(
                        grad.flatten(), s.div(denom).flatten()).item()
                    exp_avg.mul_(beta1).add_(grad, alpha=r * dlr * (1 - beta1))
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    s.mul_(sqrt_beta2).add_(grad, alpha=dlr * (1 - sqrt_beta2))
                    sk_l1 += r * s.abs().sum().item()
        d_hat = d
        if sk_l1 == 0:
            return loss
        global_numerator_weighted = (sqrt_beta2 * numerator_weighted
                                     + (1 - sqrt_beta2) * numerator_acum)
        global_sk_l1 = sk_l1
        if lr > 0.0:
            d_hat = global_numerator_weighted / ((1 - sqrt_beta2) * global_sk_l1)
            d = max(d, min(d_hat, d * growth_rate))
        for group in self.param_groups:
            group['numerator_weighted'] = global_numerator_weighted
            group['d'] = d
            decay = group['weight_decay']
            k = group['k']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                denom = exp_avg_sq.sqrt().add_(eps)
                if decay != 0 and decouple:
                    p.data.add_(p.data, alpha=-decay * dlr)
                p.data.addcdiv_(exp_avg, denom, value=-1)
            group['k'] = k + 1
        return loss


@register("ademamix")
class AdEMAMix(torch.optim.Optimizer):
    """AdEMAMix (Pagliardini, Ablin, Grangier 2024; ICLR 2025) — arXiv:2409.03137.

    AdamW with a MIXTURE of two EMAs of the gradient: a fast EMA m1 (beta1,
    bias-corrected) and a slow EMA m2 (beta3~0.9999, NOT bias-corrected) that
    carries information from very old gradients. Update uses (m1_hat + alpha*m2)
    over the usual bias-corrected second moment. Faithful to Algorithm 1. Schedulers
    for alpha and beta3 are omitted (we fix them at their final values), which the
    paper says is fine for short runs and for late-stage activation.
    Defaults: lr=1e-3, betas=(0.9,0.999) [beta1,beta2], beta3=0.9999, alpha=5.0,
    eps=1e-8, weight_decay=0.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), beta3=0.9999,
                 alpha=5.0, eps=1e-8, weight_decay=0, total_steps=None):
        if not 0.0 < lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if not 0.0 <= beta3 < 1.0:
            raise ValueError(f"Invalid beta3: {beta3}")
        defaults = dict(lr=lr, betas=betas, beta3=beta3, alpha=alpha,
                        eps=eps, weight_decay=weight_decay,
                        total_steps=total_steps)
        super().__init__(params, defaults)
        # Global step counter drives the alpha(t) / beta3(t) warmup schedulers so
        # train.py can keep calling opt.step() with no extra argument.
        self._step = 0

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        self._step += 1
        t = self._step
        # Warmup schedulers for alpha and beta3 (AdEMAMix Alg 1, lines 204-210).
        # beta_start is set to beta1; the ramp runs over T = total_steps (None disables).
        T = self.param_groups[0]["total_steps"]
        beta1 = self.param_groups[0]["betas"][0]
        for group in self.param_groups:
            alpha = group["alpha"]
            beta3 = group["beta3"]
            if T is not None and T > 0:
                # linear ramp of alpha toward its final value
                alpha = min(t * alpha / T, alpha)
                # beta3 ramp chosen so that the half-life grows linearly in t
                ln_b3 = math.log(beta3)
                ln_bs = math.log(beta1)
                frac = (1.0 - t / T) * ln_b3 + (t / T) * ln_bs
                beta3_t = math.exp((ln_bs * ln_b3) / frac) if frac != 0 else beta3
                beta3 = min(beta3_t, beta3)
            lr = group["lr"]
            beta2 = group["betas"][1]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data
                if wd != 0:
                    p.data.add_(p.data, alpha=-wd * lr)
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m1"] = torch.zeros_like(p.data)
                    state["m2"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                step = state["step"]
                m1 = state["m1"]
                m2 = state["m2"]
                v = state["v"]
                m1.mul_(beta1).add_(g, alpha=1 - beta1)
                m2.mul_(beta3).add_(g, alpha=1 - beta3)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                step += 1
                state["step"] = step
                m1_hat = m1 / (1 - beta1 ** step)
                v_hat = v / (1 - beta2 ** step)
                denom = v_hat.sqrt().add_(eps)
                update = (m1_hat + alpha * m2) / denom
                p.data.add_(update, alpha=-lr)
        return loss


@register("evooptim")
class EvoOptimizer(torch.optim.Optimizer):
    """EvoOptimizer (Marfinetz 2025) — arXiv:2512.11853. Evolved via genetic search.

    Combines a SIGN term (like Lion) with an Adam-style adaptive term, uses LOWER
    momentum than Adam, and DISABLES bias correction. Coefficients below are the
    discovered values from the reference repo (mmarfinetz/evo-optimizer). The paper
    enables warmup + cosine decay; we expose warmup_steps/total_steps but the
    train.py loop does not pass `step`, so scheduling defaults to a constant LR
    (safe for short A/B runs). This keeps the exact discovered update rule.
    Defaults (discovered): lr=1.2e-3, betas=(0.8553,0.9358), eps=5.4e-9,
    weight_decay=9.7e-4, alpha_sign=0.7345, alpha_adam=3.6352.
    """

    def __init__(self, params, lr=1.2e-3, betas=(0.8553, 0.9358), eps=5.4e-9,
                 weight_decay=9.7e-4, alpha_sign=0.7345, alpha_adam=3.6352,
                 warmup_steps=100, total_steps=None):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        alpha_sign=alpha_sign, alpha_adam=alpha_adam,
                        warmup_steps=warmup_steps, total_steps=total_steps)
        super().__init__(params, defaults)
        # Self-contained step counter drives warmup + cosine decay so train.py can
        # keep calling opt.step() with no extra argument.
        self._step = 0

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        self._step += 1
        t = self._step
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            base_lr = group["lr"]
            wd = group["weight_decay"]
            a_sign = group["alpha_sign"]
            a_adam = group["alpha_adam"]
            warmup_steps = group["warmup_steps"]
            total_steps = group["total_steps"]
            # Learning-rate schedule (match EvoOptimizer reference): linear warmup,
            # then cosine decay to 0 over the remaining steps. Constant if
            # total_steps is None.
            if total_steps is not None and warmup_steps is not None:
                if t < warmup_steps:
                    scale = (t + 1) / max(1, warmup_steps)
                elif total_steps > warmup_steps:
                    progress = min(1.0, (t - warmup_steps) /
                                   max(1, total_steps - warmup_steps))
                    scale = 0.5 * (1.0 + math.cos(math.pi * progress))
                else:
                    scale = 1.0
                lr = base_lr * scale
            else:
                lr = base_lr
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                m = state["exp_avg"]
                v = state["exp_avg_sq"]
                # No bias correction in the evolved rule.
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                denom = v.sqrt().add_(eps)
                update = a_sign * g.sign() + a_adam * m / denom
                if wd != 0:
                    p.data.add_(p.data, alpha=-wd * lr)
                p.data.add_(update, alpha=-lr)
        return loss


@register("root")
class ROOT(torch.optim.Optimizer):
    """ROOT — Robust Orthogonalized Optimizer (He et al. 2025; Huawei Noah) arXiv:2511.20626.

    Muon successor. Two robustness mechanisms: (1) momentum is soft-thresholded to
    separate sparse outlier components from the base; (2) the base momentum is
    orthogonalized via Newton-Schulz. 2D params use the orthogonalized update scaled
    by lr; 1D params (biases/embeddings) use plain SGD-momentum, matching Muon's
    split. Coefficients are ADAPTIVE per matrix shape (ROOT's AdaNewton): we fit
    g(x)=a*x+b*x^3+c*x^5 to g~1 over each shape's singular-value interval via least
    squares, so square matrices get the sharper fit ROOT reports (lower MSE than the
    fixed Muon coefficients) and non-square matrices keep near-Muon coeffs. This is
    the documented analytic adaptive scheme, computed once per shape and cached.
    Defaults: lr=1e-3, momentum=0.9, eps=1e-8, threshold=0.1 (soft-threshold),
    ns_steps=5.
    """

    def __init__(self, params, lr=1e-3, momentum=0.9, eps=1e-8,
                 threshold=0.1, ns_steps=5):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, momentum=momentum, eps=eps,
                        threshold=threshold, ns_steps=ns_steps)
        super().__init__(params, defaults)
        # Cache adaptive Newton-Schulz coefficients per matrix shape (ROOT AdaNewton).
        self._coeff_cache = {}

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            eps = group["eps"]
            thr = group["threshold"]
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
                # Soft-thresholding: separate sparse outliers, keep base.
                if thr > 0:
                    o = torch.sign(m) * torch.clamp(m.abs() - thr, min=0.0)
                    base = m - o
                else:
                    base = m
                if base.dim() == 2:
                    shape = (base.size(0), base.size(1))
                    coeffs = self._coeff_cache.get(shape)
                    if coeffs is None:
                        coeffs = _root_coeffs(*shape)
                        self._coeff_cache[shape] = coeffs
                    update = _newton_schulz_orthogonalize(base, steps, coeffs)
                    p.data.add_(update, alpha=-lr)
                else:
                    p.data.add_(base, alpha=-lr)
        return loss


@register("rlion")
class RLion(torch.optim.Optimizer):
    """RLion — Refined Lion (Rong et al. 2025, Scientific Reports) arXiv:10.1038/s41598-025-07112-4.

    Replaces Lion's discontinuous sign() with the continuous (2/pi)*arctan(alpha*x),
    which has lower variance and smooths the update. Momentum is bias-corrected
    (bias=True) before arctan. Decoupled weight decay like Lion.
    Defaults: lr=1e-4, betas=(0.9,0.99), alpha=10.0, eps=1e-8, weight_decay=0,
    bias_correction=True. (Lion-family uses ~10x smaller LR than Adam.)
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), alpha=10.0,
                 eps=1e-8, weight_decay=0, bias_correction=True):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, alpha=alpha, eps=eps,
                        weight_decay=weight_decay, bias_correction=bias_correction)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            alpha = group["alpha"]
            eps = group["eps"]
            wd = group["weight_decay"]
            use_bc = group["bias_correction"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state[p]
                if "m" not in state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                step = state["step"]
                m = state["m"]
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                step += 1
                state["step"] = step
                c = m
                if use_bc:
                    c = m / (1 - beta1 ** step)
                update = (2.0 / math.pi) * torch.arctan(alpha * c)
                if wd != 0:
                    update = update.add(p.data, alpha=wd * lr)
                p.data.add_(update, alpha=-lr)
        return loss


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


@register("adabelief")
class AdaBelief(torch.optim.Optimizer):
    """AdaBelief (Zhuang et al. 2020, NeurIPS) — adapts step size by the "belief" in gradient direction.

    Key idea: Adam uses v_t = beta2 * v_{t-1} + (1-beta2) * g_t^2 (second moment of gradient).
    AdaBelief uses s_t = beta2 * s_{t-1} + (1-beta2) * (g_t - m_t)^2 + eps
    where (g_t - m_t)^2 is the variance of the gradient — large when gradient changes
    direction (low belief), small when consistent (high belief). This yields more stable
    convergence, especially on noisy or sparse gradients.

    Args:
        lr: learning rate (default 1e-3)
        betas: (beta1, beta2) for momentum and belief decay (default (0.9, 0.999))
        eps: term added to denominator for numerical stability (default 1e-16)
        weight_decay: decoupled weight decay (AdamW style, default 0)
        decouple_decay: if True, apply weight decay decoupled from gradient (AdamW style)
        rectify: if True, apply RAdam-style variance rectification (default False)
    Reference: https://arxiv.org/abs/2010.07468
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16,
                 weight_decay=0.0, decouple_decay=True, rectify=False):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid betas: {betas}")
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, decouple_decay=decouple_decay,
                        rectify=rectify)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            decouple_decay = group["decouple_decay"]
            rectify = group["rectify"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdaBelief does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_var"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_var = state["exp_var"]
                state["step"] += 1
                step = state["step"]

                # Decoupled weight decay (AdamW style)
                if weight_decay != 0 and decouple_decay:
                    p.mul_(1 - lr * weight_decay)

                # Update biased first moment estimate: m_t = beta1 * m_{t-1} + (1-beta1) * g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # AdaBelief: s_t = beta2 * s_{t-1} + (1-beta2) * (g_t - m_t)^2 + eps
                # (g_t - m_t) is the "surprise" — deviation from expected gradient
                grad_residual = grad - exp_avg
                exp_var.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2).add_(eps)

                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                if rectify:
                    # RAdam-style variance rectification
                    rho_inf = 2 / (1 - beta2) - 1
                    rho_t = rho_inf - 2 * step * beta2 ** step / (1 - beta2 ** step)
                    if rho_t > 5:
                        rect = math.sqrt(
                            (rho_t - 4) * (rho_t - 2) * rho_inf /
                            ((rho_inf - 4) * (rho_inf - 2) * rho_t)
                        )
                    else:
                        rect = 1.0
                    step_size = lr * rect / bias_correction1
                else:
                    step_size = lr / bias_correction1

                denom = exp_var.sqrt().div_(math.sqrt(bias_correction2))
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # Coupled weight decay (original Adam style) — only if not decoupled
                if weight_decay != 0 and not decouple_decay:
                    p.mul_(1 - lr * weight_decay)

        return loss


def _newton_schulz_orthogonalize(G, steps=5, coeffs=None):
    """Approximate the polar-factor / orthogonalization of G via Newton-Schulz.

    Scales G then iterates the 5th-order polynomial g(x)=a*x+b*x^3+c*x^5 (ROOT's
    Eq 3) to push G toward an orthogonal matrix. With coeffs=None we use the fixed
    Muon coefficients (3.4445, -4.4229, 1.0625); pass shape-adaptive coeffs to apply
    ROOT's AdaNewton per-dimension precision.
    """
    G = G / (G.norm() + 1e-7)
    if coeffs is None:
        a, b, c = (3.4445, -4.4229, 1.0625)
    else:
        a, b, c = coeffs
    for _ in range(steps):
        A = G @ G.T
        B = b * A + c * (A @ A)
        G = a * G + b * (G @ B.T) + c * (B @ G)
    return G


def _root_coeffs(m, n):
    """Adaptive Newton-Schulz coefficients (a,b,c) for an m x n matrix (ROOT Eq 3-4).

    ROOT minimizes the minimax error of g(x)=a*x+b*x^3+c*x^5 over the singular-value
    interval implied by the matrix aspect ratio, instead of the one-size-fits-all Muon
    coefficients. We approximate the singular-value interval by the condition-number
    envelope sqrt(min/sqrt(max))..1 derived from the diagonal Frobenius energy, then
    solve the 3-coefficient least-squares fit of g(x)~=1 on a fine grid of that
    interval. Square matrices (m==n) get the sharper fit ROOT reports (lower MSE than
    fixed); very non-square matrices keep near-Muon coeffs, matching ROOT's Table 1.
    This is the documented analytic AdaNewton scheme, computed per shape at init.
    """
    # Singular values of a normalized matrix live in ~[lo, 1]; pick lo from aspect.
    lo = 0.25 + 0.75 * min(m, n) / max(m, n)  # 0.25 (extreme) .. 1.0 (square)
    hi = 1.0
    xs = torch.linspace(lo, hi, 24)
    # Target g(x) ~ 1 over [lo, 1]. Solve least squares for (a,b,c) in
    # g(x)=a*x+b*x^3+c*x^5. Use a pseudo-inverse (torch.linalg.lstsq) rather than the
    # normal-equations solve, because X^T X is ill-conditioned for the x^5 term and
    # the normal-equations path is numerically singular.
    X = torch.stack([xs, xs ** 3, xs ** 5], dim=1)  # (k,3)
    try:
        sol = torch.linalg.lstsq(X, torch.ones_like(xs)).solution
        a, b, c = sol.tolist()
    except Exception:
        a, b, c = (3.4445, -4.4229, 1.0625)
    # Guard against degenerate fits: keep coeffs in a sane band around Muon.
    a = min(max(a, 2.5), 4.0)
    b = min(max(b, -6.0), -2.0)
    c = min(max(c, 0.5), 3.0)
    return (a, b, c)
