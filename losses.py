#!/usr/bin/env python3
"""losses.py — registry of custom loss functions for NNI-Remake training.

Stock cross-entropy is used when a proposal sets no custom loss. Any from-
scratch loss lives here, registered by name in REGISTRY. train.py looks it up
when a proposal sets "loss": "custom:<name>" and forwards "loss_kwargs" to the
constructor. Every loss is a callable `loss(logits, targets, reduction="mean")`
that returns a scalar, so it works for both the per-batch training step (mean)
and the evaluate() loss sum (reduction="sum").

Contract for a custom loss:
  - subclass torch.nn.Module
  - __init__ accepts kwargs + an optional `reduction` ("mean" | "sum")
  - forward(logits, targets) returns a PER-SAMPLE scalar tensor (shape (B,));
    the wrapper reduces it according to `reduction`.

Implemented (faithful to the named recipe, operating on (logits, targets)):
  focal            : Focal Loss (Lin et al. 2017) — down-weights easy samples
  label_smoothing  : CE with label smoothing (Szegedy et al. 2016)
  dice             : soft multi-class Dice (Milletari et al. 2016)
  tversky          : Tversky (Salehi et al. 2017) — Dice + false-pos/neg weights
  ghm              : Gradient Harmonized Loss (Li et al. 2019) — bins gradient mag
  triplet          : lifted-structured soft triplet, LOGIT-SPACE proxy of
                     Oh et al. (2016); same-class positives vs other-class
                     negatives, computed over final logits (not embeddings)
  contrastive      : supervised NT-Xent, LOGIT-SPACE proxy of SupCon (Khosla et
                     al. 2020); same-class pairs pulled together over logits

NOTE on triplet/contrastive: a true metric loss acts on penultimate embeddings
with paired/mining samplers. These variants run on the final logits so they fit
the existing (logits, targets) per-batch contract — useful as a training signal,
but not a drop-in replacement for embedding-space metric learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

REGISTRY = {}


def register(name):
    """Class decorator: register a custom loss under `name`."""
    def _(cls):
        REGISTRY[name] = cls
        return cls
    return _


def _reduce(per_sample, reduction):
    if reduction == "sum":
        return per_sample.sum()
    return per_sample.mean()


def _onehot(targets, n_classes, dtype):
    return F.one_hot(targets, num_classes=n_classes).to(dtype)


@register("focal")
class FocalLoss(nn.Module):
    """Focal Loss (Lin et al. 2017). gamma down-weights easy examples.

    alpha (optional) is uniform class weight applied to the positive class.
    """

    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        ce = -logp.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal = (1.0 - pt) ** self.gamma * ce
        if self.alpha is not None:
            # alpha as scalar weight on the positive (target) class
            focal = self.alpha * focal
        return focal  # per-sample (B,)

    def __call__(self, logits, targets, reduction=None):
        return _reduce(self.forward(logits, targets), reduction or self.reduction)


@register("label_smoothing")
class LabelSmoothingCE(nn.Module):
    """Cross-entropy with label smoothing (Szegedy et al. 2016)."""

    def __init__(self, smoothing=0.1, reduction="mean"):
        super().__init__()
        self.smoothing = float(smoothing)
        self.reduction = reduction

    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets, label_smoothing=self.smoothing,
                               reduction=self.reduction)

    def __call__(self, logits, targets, reduction=None):
        return F.cross_entropy(
            logits, targets, label_smoothing=self.smoothing,
            reduction=reduction or self.reduction)


@register("dice")
class DiceLoss(nn.Module):
    """Soft multi-class Dice loss (Milletari et al. 2016)."""

    def __init__(self, eps=1e-6, reduction="mean"):
        super().__init__()
        self.eps = float(eps)
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        n = logits.shape[1]
        onehot = _onehot(targets, n, probs.dtype)  # (B, n)
        dims = (0,)  # over classes+batch -> scalar per sample via per-sample loop
        # per-sample dice
        per = torch.empty(probs.shape[0], device=probs.device)
        for b in range(probs.shape[0]):
            p = probs[b]            # (n,)
            t = onehot[b]           # (n,)
            inter = (p * t).sum()
            union = p.sum() + t.sum()
            per[b] = 1.0 - (2.0 * inter + self.eps) / (union + self.eps)
        return per

    def __call__(self, logits, targets, reduction=None):
        return _reduce(self.forward(logits, targets), reduction or self.reduction)


@register("tversky")
class TverskyLoss(nn.Module):
    """Tversky loss (Salehi et al. 2017): Dice with FP/FN weighting."""

    def __init__(self, alpha=0.3, beta=0.7, eps=1e-6, reduction="mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        n = logits.shape[1]
        onehot = _onehot(targets, n, probs.dtype)
        per = torch.empty(probs.shape[0], device=probs.device)
        for b in range(probs.shape[0]):
            p = probs[b]
            t = onehot[b]
            tp = (p * t).sum()
            fp = (p * (1.0 - t)).sum()
            fn = ((1.0 - p) * t).sum()
            per[b] = 1.0 - (tp + self.eps) / (
                tp + self.alpha * fp + self.beta * fn + self.eps)
        return per

    def __call__(self, logits, targets, reduction=None):
        return _reduce(self.forward(logits, targets), reduction or self.reduction)


@register("ghm")
class GHMLoss(nn.Module):
    """Gradient Harmonized Loss (Li et al. 2019), CE form.

    Each sample's gradient magnitude g = |p_t - 1| is binned into `bins`
    equal-width bins over [0,1]; the per-sample weight is 1 / (bin_density)
    so rare-gradient (hard) samples are up-weighted. Computed per-batch.
    """

    def __init__(self, bins=10, momentum=0.75, reduction="mean"):
        super().__init__()
        self.bins = int(bins)
        self.momentum = float(momentum)
        self.reduction = reduction
        self.register_buffer("acc", torch.zeros(self.bins))

    def forward(self, logits, targets):
        g = torch.sigmoid(logits)
        p_t = g.gather(1, targets.unsqueeze(1)).squeeze(1)
        grad = torch.abs(1.0 - p_t).detach()  # (B,)
        # bin edges
        edges = torch.linspace(0.0, 1.0, self.bins + 1, device=logits.device)
        idx = torch.bucketize(grad, edges) - 1
        idx = idx.clamp(0, self.bins - 1)
        # online density estimate
        with torch.no_grad():
            cnt = torch.zeros(self.bins, device=logits.device)
            cnt.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float))
            self.acc.mul_(self.momentum).add_(cnt, alpha=1.0 - self.momentum)
        dens = self.acc[idx].clamp_min(1e-6)
        weight = (1.0 - self.momentum) / dens
        weight = weight / weight.mean().clamp_min(1e-6)
        ce = F.cross_entropy(logits, targets, reduction="none")
        return ce * weight  # per-sample (B,)

    def __call__(self, logits, targets, reduction=None):
        return _reduce(self.forward(logits, targets), reduction or self.reduction)


@register("triplet")
class LogitTripletLoss(nn.Module):
    """Lifted structured soft triplet (Oh et al. 2016), LOGIT-SPACE proxy.

    Over final logits (not embeddings): for each sample, same-class logits are
    positives, other-class logits are negatives. See module docstring for the
    embedding-space caveat. margin controls separability.
    """

    def __init__(self, margin=1.0, reduction="mean"):
        super().__init__()
        self.margin = float(margin)
        self.reduction = reduction

    def forward(self, logits, targets):
        # cosine similarity over the logit vectors
        x = F.normalize(logits, dim=1)  # (B, C)
        sim = x @ x.t()                 # (B, B)
        B = logits.shape[0]
        same = (targets.unsqueeze(0) == targets.unsqueeze(1)).float()  # (B,B)
        per = torch.empty(B, device=logits.device)
        for i in range(B):
            pos = [j for j in range(B) if j != i and same[i, j] == 1]
            neg = [j for j in range(B) if same[i, j] == 0]
            if not pos or not neg:
                per[i] = 0.0
                continue
            inner = 0.0
            for j in pos:
                for k in neg:
                    inner = inner + torch.exp(sim[i, k] - sim[i, j] + self.margin)
            per[i] = torch.log1p(inner)
        return per

    def __call__(self, logits, targets, reduction=None):
        return _reduce(self.forward(logits, targets), reduction or self.reduction)


@register("contrastive")
class LogitContrastiveLoss(nn.Module):
    """Supervised NT-Xent (Khosla et al. 2020), LOGIT-SPACE proxy.

    Same-class pairs are pulled together over cosine-similar logits; see module
    docstring for the embedding-space caveat. temperature scales similarities.
    """

    def __init__(self, temperature=0.1, reduction="mean"):
        super().__init__()
        self.temperature = float(temperature)
        self.reduction = reduction

    def forward(self, logits, targets):
        x = F.normalize(logits, dim=1)
        sim = x @ x.t() / self.temperature  # (B, B)
        B = logits.shape[0]
        same = (targets.unsqueeze(0) == targets.unsqueeze(1)).float()
        mask = 1.0 - torch.eye(B, device=logits.device)
        # numerator: same-class, i != j
        num = torch.exp(sim) * same * mask
        den = torch.exp(sim) * mask
        per = torch.empty(B, device=logits.device)
        for i in range(B):
            pos = (same[i] * mask[i]).sum()
            if pos == 0:
                per[i] = 0.0
                continue
            per[i] = -torch.log((num[i].sum() + 1e-8) / (den[i].sum() + 1e-8))
        return per

    def __call__(self, logits, targets, reduction=None):
        return _reduce(self.forward(logits, targets), reduction or self.reduction)


def get(name):
    """Resolve a custom loss NAME -> class. Raises on unknown name."""
    if name not in REGISTRY:
        raise KeyError(f"custom loss {name!r} not in losses.REGISTRY "
                       f"(have: {sorted(REGISTRY)})")
    return REGISTRY[name]
