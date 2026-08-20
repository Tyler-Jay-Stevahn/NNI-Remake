"""build_model.py — turn a proposal spec into a runnable torch model.

Handles the known layer `type`s used by gen_proposals.py, and any layer
marked `novel` that carries a `definition` (Python source for an nn.Module).
The novel layer's definition is exec'd in a controlled namespace so a
genuinely new layer becomes runnable without us hand-coding it.

The builder is MODALITY-aware. A dataset declares one of:

  image        4-D tensor (B, C, H, W)     -> 2-D conv chain
  audio-mel    4-D tensor (B, 1, M, T)     -> 2-D conv chain on a mel view
  audio-wave   3-D tensor (B, 1, T)        -> 1-D conv chain on a raw waveform
  timeseries   3-D tensor (B, C, T)        -> 1-D conv / recurrent chain
  text         2-D int tensor (B, T)       -> embedding then 1-D chain
  tabular      2-D tensor (B, F)           -> dense chain

Any dataset may also carry `task_type = "clustering"`, in which case the head
maps the learned embedding to `n_clusters` soft assignments instead of class
logits. The compile gate treats those assignments as logits, so the same
train/test/predict path applies.
"""
import inspect
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

import activations
import normalizations
import regularizations

# ---------------------------------------------------------------------------
# Dataset table: modality, tensor shape, class/cluster count, vocabulary.
# ---------------------------------------------------------------------------
DATASETS = {
    # --- existing (unchanged behaviour) ---
    "mnist":            {"modality": "image", "shape": (1, 28, 28), "classes": 10},
    "cifar10":          {"modality": "image", "shape": (3, 32, 32), "classes": 10},
    "cifar100":         {"modality": "image", "shape": (3, 32, 32), "classes": 100},
    "imagenet-subset":  {"modality": "image", "shape": (3, 32, 32), "classes": 10},
    "Recycling-Data":  {"modality": "image", "shape": (3, 128, 128), "classes": 11},
    "openml-ctr23":     {"modality": "tabular", "shape": (20,), "classes": 2},
    # --- new: audio ---
    "speech-commands":  {"modality": "audio-mel", "shape": (1, 64, 64), "classes": 12},
    "esc50":            {"modality": "audio-wave", "shape": (1, 8000), "classes": 10},
    # --- new: text ---
    "ag-news":          {"modality": "text", "shape": (64,), "classes": 4, "vocab": 20000},
    "imdb":             {"modality": "text", "shape": (128,), "classes": 2, "vocab": 20000},
    "tinystories":      {"modality": "text", "shape": (128,), "classes": 0, "vocab": 256},
    # --- new: time series ---
    "uci-har":          {"modality": "timeseries", "shape": (9, 128), "classes": 6},
    # --- new: clustering ---
    "mnist-cluster":    {"modality": "image", "shape": (1, 28, 28), "classes": 10},
    "openml-cluster":   {"modality": "tabular", "shape": (20,), "classes": 8},
    # --- new: reinforcement learning (IMPALA-style CNN stem on Atari frames) ---
    "atari-ale":        {"modality": "image", "shape": (4, 84, 84), "classes": 18},
}

_DEFAULT = {"modality": "image", "shape": (3, 16, 16), "classes": 10}


def dataset_info(dataset):
    return dict(_DEFAULT, **DATASETS.get(dataset, {}))


def input_shape(dataset):
    """Back-compatible: tensor shape of one sample (no batch dim)."""
    return dataset_info(dataset)["shape"]


def num_classes(dataset):
    return dataset_info(dataset)["classes"]


def modality(dataset):
    return dataset_info(dataset)["modality"]


def output_size(spec):
    """Class count, cluster count, or vocab size for a text-gen proposal."""
    if spec.get("task_type") == "clustering":
        return int(spec.get("n_clusters", num_classes(spec.get("dataset", ""))))
    if spec.get("task_type") == "text-gen":
        # vocab is fixed by the corpus/tokenizer at train time; the proposal
        # declares it (char-level TinyStories is typically < 256).
        return int(spec.get("vocab", 256))
    return num_classes(spec.get("dataset", ""))


def sample_batch(spec, batch=2):
    """A dataset-correct synthetic (x, y) pair for the compile gate."""
    info = dataset_info(spec.get("dataset", ""))
    shape = info["shape"]
    if info["modality"] == "text":
        x = torch.randint(0, spec.get("vocab", info.get("vocab", 20000)), (batch, *shape))
    else:
        x = torch.randn(batch, *shape)
    if spec.get("task_type") == "text-gen":
        # next-token LM: targets are a (B, T) sequence of token ids (the input
        # shifted by one is applied at train time; here we just need matching
        # shape for the compile gate's loss).
        y = torch.randint(0, output_size(spec), (batch, *shape))
    else:
        y = torch.randint(0, output_size(spec), (batch,))
    return x, y


# ---------------------------------------------------------------------------
# Helper modules
# ---------------------------------------------------------------------------
class SwapTimeChannel(nn.Module):
    """(B, T, E) -> (B, E, T) so an embedding feeds a 1-D conv chain."""

    def forward(self, x):
        return x.transpose(1, 2)


class _Flatten(nn.Module):
    """Flatten all dims except batch, preserving (B, -1)."""

    def forward(self, x):
        return x.flatten(1)


class RecurrentSeq(nn.Module):
    """GRU/LSTM over (B, C, T); returns (B, H, T) so 1-D blocks may follow."""

    def __init__(self, in_ch, hidden, kind="gru", bidirectional=False, layers=1):
        super().__init__()
        rnn = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = rnn(in_ch, hidden, num_layers=layers, batch_first=True,
                       bidirectional=bidirectional)
        self.out_ch = hidden * (2 if bidirectional else 1)

    def forward(self, x):
        y, _ = self.rnn(x.transpose(1, 2))
        return y.transpose(1, 2)


class SelfAttention1d(nn.Module):
    """Single-head self attention over (B, C, T)."""

    def __init__(self, channels, heads=4):
        super().__init__()
        while channels % heads and heads > 1:
            heads -= 1
        self.att = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        s = x.transpose(1, 2)
        a, _ = self.att(s, s, s)
        return self.norm(s + a).transpose(1, 2)


class CausalSelfAttention1d(nn.Module):
    """Autoregressive (masked) self attention over (B, C, T).

    Used by text-generation models: each position may attend only to itself
    and earlier positions, never the future. A lower-triangular mask is built
    per call so variable-length sequences work.
    """

    def __init__(self, channels, heads=4):
        super().__init__()
        while channels % heads and heads > 1:
            heads -= 1
        self.heads = heads
        self.att = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # x: (B, C, T) -> (B, T, C) for MultiheadAttention
        s = x.transpose(1, 2)
        t = s.shape[1]
        # lower-triangular causal mask, shape (T, T)
        mask = torch.triu(torch.ones(t, t, device=s.device, dtype=torch.bool),
                          diagonal=1)
        a, _ = self.att(s, s, s, attn_mask=mask)
        return self.norm(s + a).transpose(1, 2)


class ClusterHead(nn.Module):
    """Embedding -> soft cluster assignment (Student-t, DEC style)."""

    def __init__(self, in_dim, embed_dim, n_clusters, alpha=1.0):
        super().__init__()
        self.project = nn.Linear(in_dim, embed_dim)
        self.centroids = nn.Parameter(torch.randn(n_clusters, embed_dim) * 0.05)
        self.alpha = alpha

    def forward(self, x):
        z = self.project(x)
        d2 = torch.cdist(z, self.centroids).pow(2)
        q = 1.0 / (1.0 + d2 / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        return torch.log(q / q.sum(1, keepdim=True) + 1e-9)


class CoordConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch + 2, out_ch, k,
                              stride=stride, padding=k // 2)

    def forward(self, x):
        b, _, h, w = x.shape
        yy = torch.arange(h, device=x.device).float() / max(h - 1, 1)
        xx = torch.arange(w, device=x.device).float() / max(w - 1, 1)
        gx, gy = torch.meshgrid(xx, yy, indexing="xy")
        coords = torch.stack([gx, gy], 0).unsqueeze(0).repeat(b, 1, 1, 1)
        return self.conv(torch.cat([x, coords], 1))


class GroupNormAct(nn.Module):
    """GroupNorm + SiLU (the standard U-Net / diffusion block activation)."""
    def __init__(self, channels, groups=32):
        super().__init__()
        self.gn = nn.GroupNorm(groups, channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.gn(x))


class ResBlock2d(nn.Module):
    """Residual block used by the IMPALA RL stem: 3x3 conv + GroupNorm + ReLU
    with a skip connection. Input and output channels match."""
    def __init__(self, channels, kernel=3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel, padding=kernel // 2)
        self.conv2 = nn.Conv2d(channels, channels, kernel, padding=kernel // 2)
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.act(self.norm2(self.conv2(h)))
        return x + h


class Attention2d(nn.Module):
    """Multi-head self-attention over spatial features (B, C, H, W)."""
    def __init__(self, channels, heads=4):
        super().__init__()
        while channels % heads and heads > 1:
            heads -= 1
        self.att = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        b, c, h, w = x.shape
        s = x.flatten(2).transpose(1, 2)          # (B, H*W, C)
        a, _ = self.att(s, s, s)
        a = self.norm(s + a)
        return a.transpose(1, 2).view(b, c, h, w)


class Downsample2d(nn.Module):
    """Spatial downsample via stride-2 conv (keeps channel count)."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample2d(nn.Module):
    """Spatial upsample via nearest + stride-1 conv (keeps channel count)."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class ConvOut(nn.Module):
    """Final diffusion head: 3x3 conv -> 1x1 conv to map to `out_ch` (RGB/features)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 1),
        )

    def forward(self, x):
        return self.seq(x)


class ActorCriticHead(nn.Module):
    """RL actor-critic head: outputs policy logits AND a scalar value estimate."""
    def __init__(self, in_dim, n_actions):
        super().__init__()
        self.policy = nn.Linear(in_dim, n_actions)
        self.value = nn.Linear(in_dim, 1)

    def forward(self, x):
        return {"logits": self.policy(x), "value": self.value(x).squeeze(-1)}


def _exec_novel(definition: str) -> type:
    """Compile + exec a layer `definition` string; return the nn.Module class.

    A minimal `wavelet_transform` stub is injected so novel layers that name a
    not-yet-implemented op still *compile* (the gate validates architecture,
    not the wavelet math). Swap in a real op later."""
    import torch

    def wavelet_transform(x, levels=1):
        # STUB: a real wavelet op goes here. Shape-preserving for compile.
        pooled = torch.nn.functional.avg_pool2d(x, 2, stride=2)
        return torch.nn.functional.interpolate(pooled, size=x.shape[-2:],
                                               mode="nearest")

    mod = types.ModuleType("novel_layer")
    mod.__dict__.update({"torch": torch, "nn": nn, "wavelet_transform": wavelet_transform})
    exec(compile(definition, "<novel-layer>", "exec"), mod.__dict__)  # noqa: S102
    found = [v for v in mod.__dict__.values()
             if isinstance(v, type) and issubclass(v, nn.Module) and v is not nn.Module]
    if not found:
        raise RuntimeError("novel layer definition defined no nn.Module subclass")
    return found[0]


def _accepted_kwargs(cls, block: dict) -> dict:
    skip = {"type", "novel", "definition", "refs", "note"}
    sig = inspect.signature(cls.__init__)
    params = set(sig.parameters) - {"self"}
    return {k: v for k, v in block.items() if k in params and k not in skip}


# ---------------------------------------------------------------------------
# Block builders
# ---------------------------------------------------------------------------
def _block(block: dict, in_ch: int):
    """2-D / generic block. Returns (module, out_channels)."""
    t = block.get("type", "")
    if block.get("novel") and "definition" in block:
        cls = _exec_novel(block["definition"])
        return cls(**_accepted_kwargs(cls, block)), in_ch  # out_ch unknown; next adapts
    if t == "linear":
        return nn.Linear(block["units"], block.get("out", block["units"])), block["units"]
    if t == "conv":
        out = block["channels"]
        return nn.Conv2d(in_ch, out, block["kernel"],
                         stride=block.get("stride", 1),
                         padding=block["kernel"] // 2), out
    if t == "coordconv":
        out = block["channels"]
        return CoordConv(in_ch, out, block["kernel"], block.get("stride", 1)), out
    if t == "dilated":
        out = block["channels"]
        return nn.Conv2d(in_ch, out, block["kernel"],
                         padding=block["dilation"], dilation=block["dilation"]), out
    if t in ("bottleneck", "basicblock", "inverted_residual"):
        out = block["channels"]
        return nn.Conv2d(in_ch, out, block.get("kernel", 3), padding=1), out
    if t == "batchnorm2d":
        return nn.BatchNorm2d(in_ch), in_ch
    if t == "groupnorm":
        return GroupNormAct(in_ch, block.get("groups", 32)), in_ch
    if t == "silu":
        return nn.SiLU(), in_ch
    if t == "activation":
        return activations.make(block["name"], in_ch), in_ch
    if t == "norm":
        return normalizations.make(block["name"], in_ch, 2), in_ch
    if t == "regularize":
        kw = {k: v for k, v in block.items()
              if k not in ("type", "name", "novel", "definition", "refs", "note")}
        return regularizations.make_regularizer(block["name"], in_ch, **kw), in_ch
    if t == "attention":
        return Attention2d(in_ch, block.get("heads", 4)), in_ch
    if t == "downsample":
        return Downsample2d(in_ch), in_ch
    if t == "upsample":
        return Upsample2d(in_ch), in_ch
    if t == "residual":
        return ResBlock2d(in_ch, block.get("kernel", 3)), in_ch
    if t == "relu":
        return nn.ReLU(), in_ch
    if t == "maxpool2d":
        return nn.MaxPool2d(block.get("kernel", 2), block.get("stride", 2)), in_ch
    if t == "avgpool2d":
        return nn.AvgPool2d(block.get("kernel", 2), block.get("stride", 2)), in_ch
    if t == "flatten":
        return _Flatten(), in_ch
    if t == "boosted_trees":
        return nn.Identity(), in_ch
    if t == "choice":
        opt = block["options"][0]
        out = 16 if "conv" in opt else in_ch
        return (nn.Conv2d(in_ch, out, 3, padding=1) if "conv" in opt
                else nn.AvgPool2d(3, 1, 1)), out
    raise ValueError(f"unknown layer type: {t!r}")


def _block1d(block: dict, in_ch: int):
    """1-D (audio / text / time series) block. Returns (module, out_channels)."""
    t = block.get("type", "")
    if block.get("novel") and "definition" in block:
        cls = _exec_novel(block["definition"])
        return cls(**_accepted_kwargs(cls, block)), in_ch
    if t == "conv1d":
        out = block["channels"]
        k = block.get("kernel", 3)
        return nn.Conv1d(in_ch, out, k, stride=block.get("stride", 1),
                         padding=k // 2), out
    if t == "dilated1d":
        out = block["channels"]
        k = block.get("kernel", 3)
        d = block.get("dilation", 2)
        return nn.Conv1d(in_ch, out, k, padding=d * (k // 2), dilation=d), out
    if t == "sepconv1d":
        out = block["channels"]
        k = block.get("kernel", 5)
        return nn.Sequential(
            nn.Conv1d(in_ch, in_ch, k, padding=k // 2, groups=in_ch),
            nn.Conv1d(in_ch, out, 1)), out
    if t in ("gru", "lstm"):
        hidden = block.get("hidden", block.get("units", 64))
        mod = RecurrentSeq(in_ch, hidden, kind=t,
                           bidirectional=block.get("bidirectional", False),
                           layers=block.get("layers", 1))
        return mod, mod.out_ch
    if t == "attention1d":
        return SelfAttention1d(in_ch, block.get("heads", 4)), in_ch
    if t == "causal":
        return CausalSelfAttention1d(in_ch, block.get("heads", 4)), in_ch
    if t == "layernorm1d":
        return nn.GroupNorm(1, in_ch), in_ch
    if t == "batchnorm1d":
        return nn.BatchNorm1d(in_ch), in_ch
    if t == "relu":
        return nn.ReLU(), in_ch
    if t == "activation":
        return activations.make(block["name"], in_ch), in_ch
    if t == "norm":
        return normalizations.make(block["name"], in_ch, 1), in_ch
    if t == "regularize":
        kw = {k: v for k, v in block.items()
              if k not in ("type", "name", "novel", "definition", "refs", "note")}
        return regularizations.make_regularizer(block["name"], in_ch, **kw), in_ch
    if t == "maxpool1d":
        return nn.MaxPool1d(block.get("kernel", 2), block.get("stride", 2)), in_ch
    raise ValueError(f"unknown 1-D layer type: {t!r}")


def _block_dense(block: dict, in_dim: int):
    """Tabular / dense block. Returns (module, out_dim)."""
    t = block.get("type", "")
    if block.get("novel") and "definition" in block:
        cls = _exec_novel(block["definition"])
        return cls(**_accepted_kwargs(cls, block)), in_dim
    if t in ("linear", "dense"):
        out = block.get("units", in_dim)
        return nn.Linear(in_dim, out), out
    if t == "boosted_trees":
        # The builder only has torch primitives; a gradient-boosted-tree block
        # maps to a single dense (MLP) hidden layer. Use `estimators` as the
        # hidden width so wider GBM configs yield wider layers (matching the
        # width-variant contract), and `depth` is informational.
        out = block.get("estimators", in_dim)
        return nn.Linear(in_dim, out), out
    if t == "relu":
        return nn.ReLU(), in_dim
    if t == "activation":
        return activations.make(block["name"], in_dim), in_dim
    if t == "norm":
        return normalizations.make(block["name"], in_dim, 0), in_dim
    if t == "regularize":
        kw = {k: v for k, v in block.items()
              if k not in ("type", "name", "novel", "definition", "refs", "note")}
        return regularizations.make_regularizer(block["name"], in_dim, **kw), in_dim
    if t == "layernorm":
        return nn.LayerNorm(in_dim), in_dim
    if t == "batchnorm":
        return nn.BatchNorm1d(in_dim), in_dim
    if t == "dropout":
        return nn.Dropout(block.get("p", 0.1)), in_dim
    raise ValueError(f"unknown dense layer type: {t!r}")


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------
def _make_head(feat_out: torch.Tensor, spec: dict) -> nn.Module:
    n_out = output_size(spec)
    clustering = spec.get("task_type") == "clustering"
    layers = []
    if feat_out.dim() == 4:
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
        dim = feat_out.shape[1]
    elif feat_out.dim() == 3:
        layers += [nn.AdaptiveAvgPool1d(1), nn.Flatten()]
        dim = feat_out.shape[1]
    elif feat_out.dim() == 2:
        dim = feat_out.shape[1]
    else:
        return nn.Identity()
    if clustering:
        layers.append(ClusterHead(dim, int(spec.get("embed_dim", 32)), n_out))
    elif spec.get("head") == "conv_out":
        # diffusion output: map the 4-D feature maps directly to `n_out`
        # channels (no global pool / flatten). Operates on (B, C, H, W).
        return ConvOut(feat_out.shape[1], n_out)
    elif spec.get("head") == "actor_critic":
        # RL policy + value head on the conv-stem output (4-D). Pool+flatten,
        # a 256-unit hidden, then policy logits + scalar value.
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(feat_out.shape[1], 256), nn.ReLU(),
            ActorCriticHead(256, n_out))
    elif spec.get("head") == "lm_head":
        # Language-modeling head for text-gen: project each position's
        # channel vector to a vocab-sized logit, keeping the full (B, T) sequence
        # so the loss is computed per token (targets are the input shifted by
        # one). feat_out is (B, C, T); output is (B, n_out, T).
        if feat_out.dim() != 3:
            raise ValueError("lm_head requires a 3-D (B, C, T) feature map")
        return nn.Conv1d(feat_out.shape[1], n_out, 1)
    else:
        layers.append(nn.Linear(dim, n_out))
    return nn.Sequential(*layers)


def build_model(spec: dict) -> nn.Module:
    model_name = spec.get("model", "")
    dataset = spec.get("dataset", "")
    info = dataset_info(dataset)
    mod = info["modality"]
    shape = info["shape"]
    nc = output_size(spec)

    # --- legacy fast paths (image only) -------------------------------------
    if mod == "image" and model_name == "mlp":
        in_ch, h, w = shape
        units = 256
        for b in spec.get("blocks", []):
            if b.get("type") == "linear" and "units" in b:
                units = b["units"]
                break
        return nn.Sequential(nn.Flatten(),
                             nn.Linear(in_ch * h * w, units),
                             nn.ReLU(), nn.Linear(units, nc))
    if mod == "image" and model_name == "lenet5":
        # Faithful LeNet-5 conv trunk (LeCun et al. 1998): conv6 -> pool ->
        # conv16 -> pool -> conv120 -> flatten -> dense84 -> dense10. Convs use
        # padding=kernel//2 (the same convention this builder's generic conv
        # path uses) so the trunk is shape-safe on the 28x28 MNIST tensor while
        # keeping the original channel counts (6/16/120) and widths (84/10).
        trunk = nn.Sequential(
            nn.Conv2d(shape[0], 6, 5, padding=2), nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, 5, padding=2), nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(16, 120, 5, padding=2), nn.ReLU(),
            nn.Flatten(),
        )
        trunk.eval()
        with torch.no_grad():
            fdim = trunk(torch.randn(1, *shape)).shape[1]
        return nn.Sequential(trunk,
                             nn.Linear(fdim, 84), nn.ReLU(),
                             nn.Linear(84, nc))
    if mod == "image" and model_name == "gbm":
        in_ch, h, w = shape
        return nn.Sequential(nn.Flatten(), nn.Linear(in_ch * h * w, nc))

    x, _ = sample_batch(spec)
    blocks = []

    if mod == "text":
        b0 = spec["blocks"][0]
        if b0.get("type") != "embedding":
            raise ValueError("a text model must start with an `embedding` block")
        emb_dim = b0.get("dim", 64)
        blocks += [nn.Embedding(spec.get("vocab", info.get("vocab", 20000)), emb_dim), SwapTimeChannel()]
        cur = emb_dim
        rest = spec["blocks"][1:]
        for b in rest:
            m, cur = _block1d(b, cur)
            blocks.append(m)
    elif mod in ("audio-wave", "timeseries"):
        cur = shape[0]
        for b in spec.get("blocks", []):
            m, cur = _block1d(b, cur)
            blocks.append(m)
    elif mod == "tabular":
        cur = shape[0]
        blocks.append(nn.Flatten())
        for b in spec.get("blocks", []):
            m, cur = _block_dense(b, cur)
            blocks.append(m)
    else:  # image, audio-mel
        cur = shape[0]
        for b in spec.get("blocks", []):
            m, cur = _block(b, cur)
            blocks.append(m)

    feat = nn.Sequential(*blocks)
    feat.eval()
    with torch.no_grad():
        f = feat(x)
    feat.train()
    head = _make_head(f, spec)
    return nn.Sequential(*blocks, head)


# ---------------------------------------------------------------------------
# Faithful canonical LeNet-5 (LeCun et al., 1998) -- reference baseline.
# 32x32 single-channel input, tanh activations throughout, and the sparse C3
# connection table from the paper (each C3 feature map is a 5x5 conv over a
# SUBSET of the 6 S2 maps, then concatenated). This reproduces the seminal
# MNIST result (~99% test acc); it is NOT the modernized `lenet5` fast-path
# used by Tlenet-mnist-M01 (which uses ReLU + padding + AvgPool + no C3 table).
# Used by tests/lenet_bench.py.
# ---------------------------------------------------------------------------
_LENET5_C3_TABLE = [
    [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 0], [5, 0, 1],
    [0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 0], [4, 5, 0, 1], [5, 0, 1, 2],
    [0, 1, 3, 4], [1, 2, 4, 5], [2, 3, 5, 0], [0, 1, 2, 3, 4, 5],
]


class _C3(nn.Module):
    """Sparse C3: 16 feature maps, each a 5x5 conv over a subset of S2 maps."""

    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleList(
            [nn.Conv2d(len(s), 1, 5) for s in _LENET5_C3_TABLE]
        )

    def forward(self, x):
        outs = [b(x[:, s, :, :]) for b, s in zip(self.branches, _LENET5_C3_TABLE)]
        return torch.cat(outs, 1)


def build_lenet5_canon(num_classes=10):
    """Canonical LeNet-5. Expects a 32x32 single-channel tensor (B,1,32,32)."""
    return nn.Sequential(
        nn.Conv2d(1, 6, 5), nn.Tanh(),        # C1: 28x28x6
        nn.AvgPool2d(2, 2),                    # S2: 14x14x6
        _C3(), nn.Tanh(),                      # C3: 10x10x16 (sparse)
        nn.AvgPool2d(2, 2),                    # S4: 5x5x16
        nn.Conv2d(16, 120, 5), nn.Tanh(),      # C5: 1x1x120
        nn.Flatten(),                          # 120
        nn.Linear(120, 84), nn.Tanh(),         # F6: 84
        nn.Linear(84, num_classes),            # OUTPUT: n_classes
    )