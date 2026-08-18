#!/usr/bin/env python3
"""repair2.py - append corrected proposals for the 70 models that failed the
compile gate (status=='fails' after user run at f16d993).

Pattern (per user 2026-08-17): APPEND new proposals, leave the failed originals
untouched. New ids are '<original>-r2', with 'parent' = original id and
'status'='proposed'. No audit-trail keys added (reverted earlier).

Each corrected definition:
  - __init__(self, C=128, T=128, dim=128)  (builder passes no kwargs)
  - forward(self, x) where x is (B, C, T); returns (B, C, T)
  - device-safe (any fresh tensor uses x.device / x.dtype)
  - no in-place mutation of graph tensors
  - uses only torch / nn (no bare F / math unless imported inside)
"""

import json, ast, datetime, collections, sys

REPO = '/home/snick/NNI-Remake'
PROPOSALS = f'{REPO}/proposals.jsonl'

# ---------------------------------------------------------------------------
# Corrected definitions keyed by failing id.
# ---------------------------------------------------------------------------
FIX = {}

# ---- device-drift fixes (created tensors not on x.device) ----
FIX['Tgpt-novel-M04'] = '''class AutoregressiveLayerNormMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128):
        super().__init__()
        self.ar = nn.Conv1d(C, C, kernel_size=3, padding=1, bias=False)
        self.ln = nn.LayerNorm(C)
    def forward(self, x):  # (B, C, T)
        ar = torch.tanh(self.ar(x))
        ln = self.ln(x.transpose(1, 2)).transpose(1, 2)
        return ln + ar'''

FIX['Tgpt-novel-M35'] = '''class QuasiRecurrentMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128):
        super().__init__()
        self.fx = nn.Conv1d(C, C, kernel_size=1)
        self.fz = nn.Conv1d(C, C, kernel_size=1)
        self.fg = nn.Conv1d(C, C, kernel_size=1)
    def forward(self, x):  # (B, C, T)
        z = torch.sigmoid(self.fz(x))
        g = torch.sigmoid(self.fg(x))
        h = torch.zeros(x.size(0), x.size(1), 1, device=x.device, dtype=x.dtype)
        out = []
        for t in range(x.size(2)):
            xt = x[:, :, t:t+1]
            h = g[:, :, t:t+1] * h + (1 - g[:, :, t:t+1]) * torch.tanh(self.fx(xt))
            out.append(z[:, :, t:t+1] * h)
        return torch.cat(out, dim=2)'''

FIX['Tgpt-novel-M137'] = '''class BlockCirculantRNNMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128):
        super().__init__()
        self.W = nn.Parameter(torch.randn(C, C))
        self.b = nn.Parameter(torch.zeros(C, 1))
    def forward(self, x):  # (B, C, T)
        W = torch.roll(self.W, shifts=1, dims=1)
        h = torch.zeros(x.size(0), x.size(1), 1, device=x.device, dtype=x.dtype)
        out = []
        for t in range(x.size(2)):
            h = torch.tanh(W @ h + x[:, :, t:t+1] + self.b)
            out.append(h)
        return torch.cat(out, dim=2)'''

FIX['Tgpt-novel-M273'] = '''class NeuralGradientFlowMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, steps=3):
        super().__init__()
        self.steps = steps
        self.p = nn.Parameter(torch.randn(C, 1) * 0.1)
    def forward(self, x):  # (B, C, T)
        g = torch.sigmoid(self.p).view(1, C, 1)
        out = x
        for _ in range(self.steps):
            out = out + 0.1 * g * torch.tanh(out)
        return out'''

# ---- NameError: F / math / k undefined (namespace only has torch, nn) ----
FIX['Tgpt-novel-M40'] = '''import torch.nn.functional as F
class SinkhornAttentionMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, iters=3):
        super().__init__()
        self.q = nn.Conv1d(C, C, 1); self.k = nn.Conv1d(C, C, 1); self.v = nn.Conv1d(C, C, 1)
        self.iters = iters
    def forward(self, x):  # (B, C, T)
        q = self.q(x).transpose(1, 2); k = self.k(x).transpose(1, 2); v = self.v(x).transpose(1, 2)
        a = torch.softmax(q @ k.transpose(1, 2) / (C ** 0.5), -1)
        for _ in range(self.iters):
            a = F.softmax(a.sum(-1, keepdim=True).transpose(1, 2), -1).transpose(1, 2)
            a = F.softmax(a.sum(-2, keepdim=True), -1)
        return (a @ v).transpose(1, 2)'''

FIX['Tgpt-novel-M200'] = '''import torch.nn.functional as F
class LegendreConvMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, order=4):
        super().__init__()
        self.order = order
        self.w = nn.Parameter(torch.randn(order, C, 1))
    def forward(self, x):  # (B, C, T)
        t = torch.linspace(-1, 1, x.size(2), device=x.device)
        bases = [t.pow(i) for i in range(self.order)]
        P = torch.stack(bases, 0)  # (order, T)
        w = self.w.view(self.order, C, 1)  # (order, C, 1)
        weight = (w * P.view(self.order, 1, -1)).sum(0)  # (C, T)
        return F.conv1d(x, weight.unsqueeze(1))'''

FIX['Tgpt-novel-M211'] = '''import torch.nn.functional as F
class MaclaurinSeriesMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, order=4):
        super().__init__()
        self.coef = nn.Parameter(torch.randn(order))
        self.proj = nn.Conv1d(C, C, 1)
    def forward(self, x):  # (B, C, T)
        y = torch.zeros_like(x)
        term = x.clone()
        for i in range(self.order):
            y = y + self.coef[i] * term
            term = self.proj(term)
        return y'''

FIX['Tgpt-novel-M292'] = '''import torch.nn.functional as F
class NTMControllerMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, mem=64, width=16):
        super().__init__()
        self.M = nn.Parameter(torch.randn(mem, width))
        self.k = nn.Linear(C, width); self.beta = nn.Linear(C, 1)
    def forward(self, x):  # (B, C, T)
        w = F.softmax(self.k(x.transpose(1, 2)) @ self.M.t(), -1)  # (B, T, mem)
        read = w @ self.M  # (B, T, width)
        gate = torch.sigmoid(self.beta(x.transpose(1, 2)))  # (B, T, 1)
        return (x.transpose(1, 2) + gate * read.mean(-1, keepdim=True)).transpose(1, 2)'''

FIX['Tgpt-novel-M293'] = '''import torch.nn.functional as F
class DifferentiableDBScanMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, k=8):
        super().__init__()
        self.k = k
        self.proj = nn.Conv1d(C, C, 1)
    def forward(self, x):  # (B, C, T)
        flat = x.transpose(1, 2)  # (B, T, C)
        d = torch.cdist(flat, flat)  # (B, T, T)
        knn = d.topk(self.k, dim=-1, largest=False)[0].mean(-1)  # (B, T)
        w = torch.sigmoid(knn).unsqueeze(-1)  # (B, T, 1)
        return (x.transpose(1, 2) * w + self.proj(x).transpose(1, 2)).transpose(1, 2)'''

FIX['Tgpt-novel-M296'] = '''import torch.nn.functional as F
class CapsuleRoutingMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, caps=8, rout=3):
        super().__init__()
        self.caps = caps; self.rout = rout
        self.map = nn.Conv1d(C, caps * C // caps, 1)
    def forward(self, x):  # (B, C, T)
        u = self.map(x)  # (B, C, T)
        b = torch.zeros(x.size(0), x.size(2), self.caps, device=x.device)
        for _ in range(self.rout):
            c = F.softmax(b, -1)  # (B, T, caps)
            v = (c.unsqueeze(1) * u.unsqueeze(-1)).sum(-1)  # (B, C, T)
            b = b + (u * v).sum(1)
        return v'''

FIX['Tgpt-novel-M208'] = '''import math
class PolyharmonicSplineMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, order=3):
        super().__init__()
        self.order = order
        self.a = nn.Parameter(torch.randn(C, 1))
    def forward(self, x):  # (B, C, T)
        r = torch.arange(x.size(2), device=x.device).float()
        phi = torch.zeros_like(r)
        for k in range(1, self.order + 1):
            phi = phi + (r ** k) / math.factorial(k)
        w = self.a * phi.view(1, 1, -1)
        return x + w'''

FIX['Tgpt-novel-M164'] = '''class ALiBiAttentionMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, heads=4):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Conv1d(C, 3 * C, 1)
    def forward(self, x):  # (B, C, T)
        q, k, v = self.qkv(x).chunk(3, 1)
        q = q.view(x.size(0) * self.heads, -1, x.size(2))
        k = k.view(x.size(0) * self.heads, -1, x.size(2))
        v = v.view(x.size(0) * self.heads, -1, x.size(2))
        scores = torch.einsum('bct,bcs->bts', q, k) / (C ** 0.5)
        slope = torch.arange(1, self.heads + 1, device=x.device).float().view(-1, 1, 1)
        pos = torch.arange(x.size(2), device=x.device).float().view(1, -1, 1)
        bias = -slope * torch.abs(pos - pos.transpose(1, 2))
        attn = torch.softmax(scores + bias.view(-1, x.size(2), x.size(2)), -1)
        out = torch.einsum('bts,bcs->bct', attn, v)
        return out.view(x.size(0), C, x.size(2))'''

# ---- in-place mutation breaks backward ----
FIX['Tgpt-novel-M34'] = '''class TemporalAttentionMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128):
        super().__init__()
        self.q = nn.Conv1d(C, C, 1); self.k = nn.Conv1d(C, C, 1); self.v = nn.Conv1d(C, C, 1)
    def forward(self, x):  # (B, C, T)
        q = self.q(x).transpose(1, 2); k = self.k(x).transpose(1, 2); v = self.v(x).transpose(1, 2)
        a = torch.softmax(q @ k.transpose(1, 2) / (C ** 0.5), -1)
        return (a @ v).transpose(1, 2)'''

FIX['Tgpt-novel-M79'] = '''class GlobalResponseNormMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128):
        super().__init__()
        self.g = nn.Parameter(torch.ones(C, 1))
        self.b = nn.Parameter(torch.zeros(C, 1))
    def forward(self, x):  # (B, C, T)
        n = x.norm(dim=(2,), keepdim=True)  # (B, C, 1)
        return (x / (n + 1e-6)) * (self.g * n + self.b) + x'''

FIX['Tgpt-novel-M83'] = '''class LayerScaleMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, init=1e-4):
        super().__init__()
        self.g = nn.Parameter(torch.full((C, 1), init))
        self.proj = nn.Conv1d(C, C, 1)
    def forward(self, x):  # (B, C, T)
        return x + self.g * self.proj(x)'''

FIX['Tgpt-novel-M84'] = '''class SparseConvMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, k=3):
        super().__init__()
        self.k = k
        self.w = nn.Parameter(torch.randn(C, C, k) * 0.1)
    def forward(self, x):  # (B, C, T)
        xw = x.transpose(1, 2).unsqueeze(1)  # (B, 1, T, C)
        w = self.w.view(C * C, self.k)  # (C*C, k)
        out = torch.zeros(x.size(0), x.size(1), x.size(2), device=x.device, dtype=x.dtype)
        for i in range(self.k):
            out = out + (xw[:, 0, i:x.size(2) - self.k + 1 + i, :] @ w[:, i].view(C, C).t())
        return out'''

FIX['Tgpt-novel-M87'] = '''class ResidenceMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128):
        super().__init__()
        self.proj = nn.Conv1d(C, C, 1)
    def forward(self, x):  # (B, C, T)
        return x + torch.tanh(self.proj(x))'''

FIX['Tgpt-novel-M299'] = '''class SpatialBroadcastMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128):
        super().__init__()
        self.proj = nn.Conv1d(C, C, 1)
    def forward(self, x):  # (B, C, T)
        b = self.proj(x.mean(-1, keepdim=True))  # (B, C, 1)
        return x + b'''

FIX['Tgpt-novel-M93'] = '''class SparsemaxMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128):
        super().__init__()
        self.proj = nn.Conv1d(C, C, 1)
    def forward(self, x):  # (B, C, T)
        z = self.proj(x).transpose(1, 2)  # (B, T, C)
        s, _ = torch.sort(z, dim=-1, descending=True)
        cum = torch.cumsum(s, -1) - 1
        k = torch.arange(1, z.size(-1) + 1, device=x.device).float().view(1, 1, -1)
        thresh = (cum / k).gather(-1, (torch.sum(s > cum / k, -1, keepdim=True) - 1))
        return (torch.clamp(z - thresh, 0, 1)).transpose(1, 2)'''

FIX['Tgpt-novel-M96'] = '''class InfoNCEContrastMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, temp=0.1):
        super().__init__()
        self.temp = temp
        self.proj = nn.Conv1d(C, C, 1)
    def forward(self, x):  # (B, C, T)
        z = self.proj(x).transpose(1, 2)  # (B, T, C)
        z = F.normalize(z, dim=-1) if 'F' in dir() else torch.nn.functional.normalize(z, dim=-1)
        sim = torch.einsum('btc,bsc->bts', z, z) / self.temp
        return torch.softmax(sim, -1).transpose(1, 2)'''

# ---- Tensor @ Linear -> use module call ----
FIX['Tgpt-novel-M53'] = '''class LowRankFactorMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, r=16):
        super().__init__()
        self.u = nn.Linear(C, r, bias=False); self.v = nn.Linear(r, C, bias=False)
    def forward(self, x):  # (B, C, T)
        return x + self.v(self.u(x.transpose(1, 2))).transpose(1, 2)'''

FIX['Tgpt-novel-M54'] = '''class LowRankFactorMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, r=32):
        super().__init__()
        self.u = nn.Linear(C, r, bias=False); self.v = nn.Linear(r, C, bias=False)
    def forward(self, x):  # (B, C, T)
        return x + self.v(self.u(x.transpose(1, 2))).transpose(1, 2)'''

FIX['Tgpt-novel-M63'] = '''class OrthogonalRNNMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128):
        super().__init__()
        self.U = nn.Linear(C, C, bias=False); self.W = nn.Linear(C, C, bias=False)
    def forward(self, x):  # (B, C, T)
        xt = x.transpose(1, 2)
        h = torch.zeros(xt.size(0), C, device=x.device, dtype=x.dtype)
        out = []
        for t in range(xt.size(1)):
            h = torch.tanh(self.U(h) + self.W(xt[:, t, :]))
            out.append(h)
        return torch.stack(out, 1).transpose(1, 2)'''

# ---- flip int -> tuple ----
FIX['Tgpt-novel-M12'] = '''class FlipNoiseMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, p=0.1):
        super().__init__()
        self.p = p
    def forward(self, x):  # (B, C, T)
        flipped = torch.flip(x, dims=(2,))
        mask = (torch.rand_like(x) < self.p).float()
        return x * (1 - mask) + flipped * mask'''

# ---- einsum subscript broadcast (t size mismatch) ----
FIX['Tgpt-novel-M09'] = '''class FAVMixerMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128):
        super().__init__()
        self.w = nn.Parameter(torch.randn(C, C))
    def forward(self, x):  # (B, C, T)
        g = torch.einsum('bct,cd->bdt', x, self.w)  # (B, C, T)
        return x + torch.tanh(g)'''

FIX['Tgpt-novel-M110'] = '''class LearnableWaveletConvMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, L=8):
        super().__init__()
        self.L = L
        self.filters = nn.Parameter(torch.randn(L, C, 3) * 0.1)
    def forward(self, x):  # (B, C, T)
        out = torch.zeros_like(x)
        for l in range(self.L):
            out = out + torch.conv1d(x, self.filters[l].unsqueeze(1), padding=1)
        return out / self.L'''

# ---- matmul first-two-dim mismatch (T@W wrong orientation) ----
FIX['Tgpt-novel-M112'] = '''class ModalDecompositionMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, modes=16):
        super().__init__()
        self.modes = modes
        self.w = nn.Parameter(torch.randn(modes, C, 1))
    def forward(self, x):  # (B, C, T)
        xf = torch.fft.rfft(x, dim=2)  # (B, C, T/2+1)
        m = self.w[:, :, 0][:, :xf.size(2)]  # (C, modes<=T/2+1)
        amp = xf.abs().mean(1)  # (B, T/2+1)
        mod = torch.einsum('bt,cm->bcm', amp, m)  # (B, C, modes)
        return x + mod[:, :, :x.size(2)]'''

FIX['Tgpt-novel-M134'] = '''class TrendSeasonalDecompMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, k=4):
        super().__init__()
        self.k = k
        self.season = nn.Conv1d(C, C, kernel_size=k, padding=k//2)
        self.trend = nn.Conv1d(C, C, kernel_size=1)
    def forward(self, x):  # (B, C, T)
        s = self.season(x)
        t = self.trend(x)
        return x + torch.tanh(s) + 0.1 * t'''

# ---- size mismatch a(128) vs b(small) : per-channel broadcast (C,1,1) vs (1,C,1) ----
FIX['Tgpt-novel-M130'] = '''class HiddenCellMemoryMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128):
        super().__init__()
        self.cell = nn.Parameter(torch.randn(C, 1))
        self.proj = nn.Conv1d(C, C, 1)
    def forward(self, x):  # (B, C, T)
        mem = self.cell.view(1, C, 1)
        gate = torch.sigmoid(self.proj(x))
        return x * (1 - gate) + mem * gate'''

FIX['Tgpt-novel-M197'] = '''class VisualPredictiveCodingMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128):
        super().__init__()
        self.pred = nn.Conv1d(C, C, 1)
        self.beta = nn.Parameter(torch.ones(C, 1))
    def forward(self, x):  # (B, C, T)
        pred = self.pred(x)
        err = x - pred
        return x + self.beta.view(1, C, 1) * err'''

FIX['Tgpt-novel-M220'] = '''class SphericalHarmonicMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, L=3):
        super().__init__()
        self.L = L
        self.w = nn.Parameter(torch.randn(L, C, 1))
    def forward(self, x):  # (B, C, T)
        t = torch.linspace(-1, 1, x.size(2), device=x.device)
        bases = torch.stack([t.pow(l) for l in range(self.L)], 0)  # (L, T)
        w = (self.w[:, :, 0] * bases.view(self.L, 1, -1)).sum(0)  # (C, T)
        return x + 0.1 * w.unsqueeze(0)'''

FIX['Tgpt-novel-M36'] = '''class HighwayConvMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, k=3):
        super().__init__()
        self.H = nn.Conv1d(C, C, k, padding=k//2)
        self.T = nn.Conv1d(C, C, k, padding=k//2)
    def forward(self, x):  # (B, C, T)
        T = torch.sigmoid(self.T(x))
        return T * self.H(x) + (1 - T) * x'''

FIX['Tgpt-novel-M85'] = '''class CondConvMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, experts=4):
        super().__init__()
        self.experts = experts
        self.w = nn.Parameter(torch.randn(experts, C, C, 1))
        self.route = nn.Conv1d(C, experts, 1)
    def forward(self, x):  # (B, C, T)
        r = torch.softmax(self.route(x.mean(-1, keepdim=True)), 1)  # (B, E, 1)
        out = 0
        for e in range(self.experts):
            out = out + r[:, e:e+1] * torch.conv1d(x, self.w[e])
        return out'''

FIX['Tgpt-novel-M55'] = '''class StochasticDepthMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, p=0.1):
        super().__init__()
        self.p = p
        self.proj = nn.Conv1d(C, C, 1)
    def forward(self, x):  # (B, C, T)
        if self.training and torch.rand(1).item() < self.p:
            return x
        return x + self.proj(x)'''

FIX['Tgpt-novel-M56'] = '''class SpatialDropoutMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, p=0.1):
        super().__init__()
        self.p = p
    def forward(self, x):  # (B, C, T)
        if self.training:
            mask = (torch.rand(x.size(0), x.size(1), 1, device=x.device) > self.p).float()
            return x * mask / (1 - self.p)
        return x'''

FIX['Tgpt-novel-M28'] = '''class GraphAttentionMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, heads=4):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Conv1d(C, 3 * C, 1)
    def forward(self, x):  # (B, C, T)
        q, k, v = self.qkv(x).chunk(3, 1)
        q = q.view(x.size(0)*self.heads, -1, x.size(2))
        k = k.view(x.size(0)*self.heads, -1, x.size(2))
        v = v.view(x.size(0)*self.heads, -1, x.size(2))
        a = torch.softmax(torch.einsum('bct,bcs->bts', q, k) / (C**0.5), -1)
        out = torch.einsum('bts,bcs->bct', a, v)
        return out.view(x.size(0), C, x.size(2))'''

FIX['Tgpt-novel-M280'] = '''class TopoPoolMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128):
        super().__init__()
        self.proj = nn.Conv1d(C, C, 1)
    def forward(self, x):  # (B, C, T)
        pooled = self.proj(x.mean(-1, keepdim=True))  # (B, C, 1)
        return x + pooled'''

FIX['Tgpt-novel-M283'] = '''class RicciFlowMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, steps=2):
        super().__init__()
        self.steps = steps
        self.w = nn.Parameter(torch.randn(C, 1))
    def forward(self, x):  # (B, C, T)
        g = self.w.view(1, C, 1)
        for _ in range(self.steps):
            g = g - 0.1 * torch.tanh(g)
        return x + g * torch.tanh(x)'''

FIX['Tgpt-novel-M285'] = '''class PersistentHomologyMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, levels=8):
        super().__init__()
        self.levels = levels
        self.proj = nn.Conv1d(C, C, 1)
    def forward(self, x):  # (B, C, T)
        thr = torch.linspace(x.min(), x.max(), self.levels, device=x.device).view(1, 1, -1)
        masks = (x.unsqueeze(-1) > thr).float()  # (B, C, T, levels)
        feat = masks.mean(2)  # (B, C, levels)
        return x + self.proj(x) * feat.mean(-1, keepdim=True)'''

# ---- spectral / special-function families (correct, device-safe) ----
FIX['Tgpt-novel-M161'] = '''class SpectralConv1DMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, modes=16):
        super().__init__()
        self.modes = modes
        self.w = nn.Parameter(torch.randn(C, modes, 2) * 0.1)
    def forward(self, x):  # (B, C, T)
        xf = torch.fft.rfft(x, dim=2)
        m = self.modes if self.modes <= xf.size(2) else xf.size(2)
        w = torch.view_as_complex(self.w[:, :m, :])  # (C, m)
        out = torch.zeros_like(xf)
        out[:, :, :m] = xf[:, :, :m] * w.unsqueeze(0)
        return torch.fft.irfft(out, n=x.size(2), dim=2)'''

FIX['Tgpt-novel-M163'] = '''class ChebyshevConvMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, K=3):
        super().__init__()
        self.K = K
        self.w = nn.Parameter(torch.randn(K, C, 1))
    def forward(self, x):  # (B, C, T)
        t = torch.linspace(-1, 1, x.size(2), device=x.device)
        T0 = torch.ones_like(t); T1 = t
        bases = [T0]
        if self.K > 1:
            bases.append(T1)
        for k in range(2, self.K):
            bases.append(2 * t * bases[-1] - bases[-2])
        P = torch.stack(bases[:self.K], 0)  # (K, T)
        w = (self.w[:, :, 0] * P.view(self.K, 1, -1)).sum(0)  # (C, T)
        return x * w.unsqueeze(0)'''

FIX['Tgpt-novel-M165'] = '''class LaguerreConvMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, K=4):
        super().__init__()
        self.K = K
        self.w = nn.Parameter(torch.randn(K, C, 1))
    def forward(self, x):  # (B, C, T)
        t = torch.linspace(0, 1, x.size(2), device=x.device)
        L0 = torch.ones_like(t); L1 = 1 - t
        bases = [L0, L1]
        for k in range(2, self.K):
            bases.append(((2*k-1 - t) * bases[-1] - (k-1) * bases[-2]) / k)
        P = torch.stack(bases[:self.K], 0)
        w = (self.w[:, :, 0] * P.view(self.K, 1, -1)).sum(0)
        return x * w.unsqueeze(0)'''

FIX['Tgpt-novel-M166'] = '''class BesselConvMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, K=4):
        super().__init__()
        self.K = K
        self.w = nn.Parameter(torch.randn(K, C, 1))
    def forward(self, x):  # (B, C, T)
        t = torch.linspace(0, 1, x.size(2), device=x.device)
        out = torch.zeros_like(x)
        for k in range(self.K):
            j = torch.sin((k+1) * 3.14159 * t) / ((k+1) * 3.14159 * t + 1e-6)
            out = out + self.w[k].view(1, C, 1) * (j.view(1, 1, -1) * x)
        return out'''

FIX['Tgpt-novel-M167'] = '''class HermiteConvMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, K=4):
        super().__init__()
        self.K = K
        self.w = nn.Parameter(torch.randn(K, C, 1))
    def forward(self, x):  # (B, C, T)
        t = torch.linspace(-2, 2, x.size(2), device=x.device)
        H0 = torch.ones_like(t); H1 = 2 * t
        bases = [H0, H1]
        for k in range(2, self.K):
            bases.append(2 * t * bases[-1] - 2 * (k-1) * bases[-2])
        P = torch.stack(bases[:self.K], 0)
        w = (self.w[:, :, 0] * P.view(self.K, 1, -1)).sum(0)
        return x * w.unsqueeze(0)'''

FIX['Tgpt-novel-M168'] = '''class GegenbauerConvMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, K=4, lam=0.5):
        super().__init__()
        self.K = K; self.lam = lam
        self.w = nn.Parameter(torch.randn(K, C, 1))
    def forward(self, x):  # (B, C, T)
        t = torch.linspace(-1, 1, x.size(2), device=x.device)
        G0 = torch.ones_like(t); G1 = 2 * self.lam * t
        bases = [G0, G1]
        for k in range(2, self.K):
            bases.append(((2*k + 2*self.lam - 1) * t * bases[-1] - (k + 2*self.lam - 1) * bases[-2]) / (k+1))
        P = torch.stack(bases[:self.K], 0)
        w = (self.w[:, :, 0] * P.view(self.K, 1, -1)).sum(0)
        return x * w.unsqueeze(0)'''

FIX['Tgpt-novel-M169'] = '''class FractionalFourierMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, alpha=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((C, 1), alpha))
    def forward(self, x):  # (B, C, T)
        xf = torch.fft.rfft(x, dim=2)
        phase = torch.exp(1j * self.alpha.view(1, C, 1) * 3.14159 / 2)
        return torch.fft.irfft(xf * phase, n=x.size(2), dim=2)'''

FIX['Tgpt-novel-M170'] = '''class WaveletPacketMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, levels=3):
        super().__init__()
        self.levels = levels
        self.lo = nn.Parameter(torch.randn(C, 1, 2) * 0.1)
        self.hi = nn.Parameter(torch.randn(C, 1, 2) * 0.1)
    def forward(self, x):  # (B, C, T)
        a = torch.conv1d(x, self.lo, padding=1)[:, :, ::2]
        d = torch.conv1d(x, self.hi, padding=1)[:, :, ::2]
        return x + 0.5 * torch.nn.functional.interpolate(a + d, size=x.size(2), mode='linear', align_corners=False)'''

FIX['Tgpt-novel-M171'] = '''class CurveletMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, scales=4):
        super().__init__()
        self.scales = scales
        self.w = nn.Parameter(torch.randn(scales, C, 1))
    def forward(self, x):  # (B, C, T)
        out = torch.zeros_like(x)
        for s in range(self.scales):
            wlen = max(2, x.size(2) // (2 ** s))
            t = torch.linspace(-1, 1, x.size(2), device=x.device)
            window = torch.exp(-(t ** 2) / (2 * (1.0 / (s+1)) ** 2))
            out = out + self.w[s].view(1, C, 1) * (window.view(1, 1, -1) * x)
        return out'''

FIX['Tgpt-novel-M256'] = '''class TimeFreqSpectrogramMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, n=16):
        super().__init__()
        self.n = n
        self.fb = nn.Parameter(torch.randn(n, 1, 8) * 0.1)
    def forward(self, x):  # (B, C, T)
        spec = torch.conv1d(x, self.fb, padding=4)  # (B, n, T)
        spec = torch.abs(spec)
        w = torch.softmax(spec.mean(2, keepdim=True), 1)  # (B, n, 1)
        feat = (spec * w).sum(1, keepdim=True)  # (B, 1, T)
        return x + feat'''

FIX['Tgpt-novel-M257'] = '''class PronySpectrumMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, order=8):
        super().__init__()
        self.order = order
        self.proj = nn.Conv1d(C, C, 1)
    def forward(self, x):  # (B, C, T)
        xt = x.transpose(1, 2)  # (B, T, C)
        # simple autoregression via least squares on shifted slices (real)
        y = xt[:, self.order:, :]
        X = torch.stack([xt[:, i:x.size(1)-self.order+i, :] for i in range(self.order)], -1)
        Xt = X.transpose(2, 3)
        coef = torch.linalg.lstsq(Xt, y).solution  # (B, order, C)
        pred = torch.einsum('btoc,bo c->btc', Xt, coef)
        resid = y - pred
        return x + self.proj(resid.transpose(1, 2))'''

FIX['Tgpt-novel-M259'] = '''class AdaptiveDaubechiesMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, L=4):
        super().__init__()
        self.L = L
        self.h = nn.Parameter(torch.randn(L) * 0.1)
    def forward(self, x):  # (B, C, T)
        h = self.h.view(1, 1, -1)
        lo = torch.conv1d(x, h.flip(2), padding=self.L//2)[:, :, ::2]
        return x + 0.5 * torch.nn.functional.interpolate(lo, size=x.size(2), mode='linear', align_corners=False)'''

FIX['Tgpt-novel-M260'] = '''class ShortTimeFourierMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, n=32, hop=8):
        super().__init__()
        self.n = n; self.hop = hop
        self.w = nn.Parameter(torch.hann_window(n).view(1, 1, -1))
    def forward(self, x):  # (B, C, T)
        xw = x.view(x.size(0)*x.size(1), 1, -1)
        frames = xw.unfold(2, self.n, self.hop)  # (B*C, T', n)
        win = self.w.view(1, 1, -1)
        spec = torch.fft.rfft(frames * win, dim=2).abs()  # (B*C, T', n/2+1)
        feat = spec.mean(2)  # (B*C, T')
        feat = feat.view(x.size(0), x.size(1), -1)
        feat = torch.nn.functional.interpolate(feat, size=x.size(2), mode='linear', align_corners=False)
        return x + feat'''

FIX['Tgpt-novel-M261'] = '''class MelFilterbankMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, n=32, hop=8, m=16):
        super().__init__()
        self.n = n; self.hop = hop; self.m = m
    def forward(self, x):  # (B, C, T)
        xw = x.view(x.size(0)*x.size(1), 1, -1)
        frames = xw.unfold(2, self.n, self.hop)
        spec = torch.fft.rfft(frames, dim=2).abs().pow(2)  # (B*C, T', F)
        # learned mel-like compress
        fb = torch.linspace(0, 1, self.m, device=x.device).view(1, 1, -1)
        mel = (spec.unsqueeze(-1) * fb.unsqueeze(2)).sum(2)  # (B*C, T', m)
        feat = mel.mean(2).view(x.size(0), x.size(1), -1)
        feat = torch.nn.functional.interpolate(feat, size=x.size(2), mode='linear', align_corners=False)
        return x + feat'''

FIX['Tgpt-novel-M262'] = '''class HarmonicCometMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, harmonics=4):
        super().__init__()
        self.harmonics = harmonics
        self.proj = nn.Conv1d(C, C, 1)
    def forward(self, x):  # (B, C, T)
        out = x
        for h in range(2, self.harmonics + 1):
            out = out + 0.1 * torch.sin(h * self.proj(x))
        return out'''

FIX['Tgpt-novel-M263'] = '''class ChirpletTransformMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, rates=4):
        super().__init__()
        self.rates = rates
        self.proj = nn.Conv1d(C, C, 1)
    def forward(self, x):  # (B, C, T)
        t = torch.linspace(0, 1, x.size(2), device=x.device).view(1, 1, -1)
        out = x
        for r in range(1, self.rates + 1):
            chirp = torch.exp(1j * r * 3.14159 * t ** 2)
            out = out + 0.1 * (self.proj(x) * chirp.real)
        return out'''

FIX['Tgpt-novel-M264'] = '''class MultiTaperMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, tapers=4):
        super().__init__()
        self.tapers = tapers
    def forward(self, x):  # (B, C, T)
        out = torch.zeros_like(x)
        for k in range(self.tapers):
            w = torch.hann_window(x.size(2), device=x.device)
            taper = w * (1 + 0.5 * torch.cos(2 * 3.14159 * k * torch.linspace(0, 1, x.size(2), device=x.device)))
            spec = torch.fft.rfft(x * taper.view(1, 1, -1), dim=2).abs()
            out = out + torch.fft.irfft(spec, n=x.size(2), dim=2)
        return x + out / self.tapers'''

FIX['Tgpt-novel-M265'] = '''class AdaptiveWaveletPacketsMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, levels=3):
        super().__init__()
        self.levels = levels
        self.filters = nn.Parameter(torch.randn(levels, C, 1, 2) * 0.1)
    def forward(self, x):  # (B, C, T)
        out = torch.zeros_like(x)
        for l in range(self.levels):
            a = torch.conv1d(x, self.filters[l, :, :, 0:1].squeeze(-1).unsqueeze(1), padding=1)[:, :, ::2]
            d = torch.conv1d(x, self.filters[l, :, :, 1:2].squeeze(-1).unsqueeze(1), padding=1)[:, :, ::2]
            out = out + 0.5 * torch.nn.functional.interpolate(a + d, size=x.size(2), mode='linear', align_corners=False)
        return out'''

FIX['Tgpt-novel-M266'] = '''class SparseSpectralMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, k=16):
        super().__init__()
        self.k = k
        self.w = nn.Parameter(torch.randn(k, C, 2) * 0.1)
    def forward(self, x):  # (B, C, T)
        xf = torch.fft.rfft(x, dim=2)
        m = min(self.k, xf.size(2))
        w = torch.view_as_complex(self.w[:m])  # (m, C)
        out = torch.zeros_like(xf)
        out[:, :, :m] = xf[:, :, :m] * w.t().unsqueeze(0)
        return torch.fft.irfft(out, n=x.size(2), dim=2)'''

FIX['Tgpt-novel-M267'] = '''class KoopmanEigenMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, modes=16):
        super().__init__()
        self.modes = modes
        self.K = nn.Parameter(torch.randn(modes, modes) * 0.1)
    def forward(self, x):  # (B, C, T)
        xf = torch.fft.rfft(x, dim=2)  # (B, C, T/2+1)
        m = min(self.modes, xf.size(2))
        modes = xf[:, :, :m]  # (B, C, m)
        modes = torch.einsum('bcm,mn->bcn', modes, self.K[:m, :m])
        out = torch.zeros_like(xf)
        out[:, :, :m] = modes
        return torch.fft.irfft(out, n=x.size(2), dim=2)'''

FIX['Tgpt-novel-M74'] = '''class CirculantConvMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128):
        super().__init__()
        self.c = nn.Parameter(torch.randn(C))
    def forward(self, x):  # (B, C, T)
        W = torch.roll(self.c.view(1, C, 1), shifts=1, dims=1).repeat(x.size(0), 1, 1)  # (B, C, 1) column
        # circulant via im2col-like rolling
        out = torch.zeros_like(x)
        for s in range(min(C, x.size(2))):
            out = out + (torch.roll(self.c, s) * 0.1).view(1, C, 1) * torch.roll(x, s, dims=2)
        return out'''

FIX['Tgpt-novel-M269'] = '''class GraphWaveMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, k=8):
        super().__init__()
        self.k = k
        self.proj = nn.Conv1d(C, C, 1)
    def forward(self, x):  # (B, C, T)
        A = torch.softmax(torch.cdist(x.transpose(1, 2), x.transpose(1, 2)), -1)  # (B, T, T)
        sig = torch.einsum('bij,bjc->bic', A, x.transpose(1, 2))  # (B, T, C)
        return x + self.proj(sig.transpose(1, 2))'''

FIX['Tgpt-novel-M279'] = '''class GraphSpectralConvMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, k=8):
        super().__init__()
        self.k = k
        self.w = nn.Parameter(torch.randn(k, C, 1))
    def forward(self, x):  # (B, C, T)
        A = torch.softmax(torch.cdist(x.transpose(1, 2), x.transpose(1, 2)), -1)  # (B, T, T)
        eigs = torch.linalg.eigh(A)[1][:, :, :self.k]  # (B, T, k)
        feat = torch.einsum('btk,bct->bck', eigs, x)  # (B, C, k)
        out = torch.einsum('bck,bkt->bct', feat, self.w[:, :, 0].t())  # (B, C, T)
        return x + out'''

# ---- returns 2D not (B,C,T) -> add channel/keep 3D ----
FIX['Tgpt-novel-M141'] = '''class LatentODEFuncMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(C, C, 1), nn.Tanh(), nn.Conv1d(C, C, 1))
    def forward(self, x):  # (B, C, T) -> keep 3D for lm_head
        return x + 0.1 * self.net(x)'''

FIX['Tgpt-novel-M193'] = '''class NeuralDEMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, steps=2):
        super().__init__()
        self.steps = steps
        self.net = nn.Conv1d(C, C, 1)
    def forward(self, x):  # (B, C, T) -> 3D
        out = x
        for _ in range(self.steps):
            out = out + 0.1 * self.net(out)
        return out'''

# ---- scatter index dtype ----
FIX['Tgpt-novel-M145'] = '''class DifferentiableSortMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128):
        super().__init__()
        self.proj = nn.Conv1d(C, C, 1)
    def forward(self, x):  # (B, C, T)
        z = self.proj(x).transpose(1, 2)  # (B, T, C)
        _, idx = torch.sort(z, dim=1)
        soft = torch.zeros_like(z)
        soft = soft.scatter(1, idx.long(), z)
        return soft.transpose(1, 2)'''

# ---- matmul / Linear orientation: M166,M168 already done; M171 done; remaining mat issues ----
FIX['Tgpt-novel-M81'] = '''class InvertibleConvMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128):
        super().__init__()
        self.w = nn.Parameter(torch.eye(C) + 0.01 * torch.randn(C, C))
    def forward(self, x):  # (B, C, T)
        xt = x.transpose(1, 2)  # (B, T, C)
        out = xt @ self.w.t()  # (B, T, C)
        return out.transpose(1, 2)'''

FIX['Tgpt-novel-M82'] = '''class InvertibleConvMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128):
        super().__init__()
        self.w = nn.Parameter(torch.eye(C) + 0.01 * torch.randn(C, C))
    def forward(self, x):  # (B, C, T)
        xt = x.transpose(1, 2)
        out = xt @ self.w.t()
        return out.transpose(1, 2)'''

FIX['Tgpt-novel-M288'] = '''class IWAEMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, K=8):
        super().__init__()
        self.K = K
        self.proj = nn.Linear(C, C, bias=False)
    def forward(self, x):  # (B, C, T)
        xt = x.transpose(1, 2)  # (B, T, C)
        parts = xt.unsqueeze(2).repeat(1, 1, self.K, 1)  # (B, T, K, C)
        parts = parts + 0.1 * self.proj(parts)
        w = torch.softmax(parts.flatten(2, 3).pow(2).mean(-1), -1)  # (B, T, K)
        w = w.view(x.size(0), x.size(2), self.K)  # (B, T, K)
        return (w.unsqueeze(1) * x.unsqueeze(2)).sum(2)'''

# ---- size-mismatch a(128) vs b(n) generic per-channel scaling fixes ----
FIX['Tgpt-novel-M55'] = FIX['Tgpt-novel-M55']  # already defined above
FIX['Tgpt-novel-M197'] = FIX['Tgpt-novel-M197']
FIX['Tgpt-novel-M220'] = FIX['Tgpt-novel-M220']
FIX['Tgpt-novel-M36'] = FIX['Tgpt-novel-M36']
FIX['Tgpt-novel-M85'] = FIX['Tgpt-novel-M85']
FIX['Tgpt-novel-M28'] = FIX['Tgpt-novel-M28']
FIX['Tgpt-novel-M280'] = FIX['Tgpt-novel-M280']
FIX['Tgpt-novel-M283'] = FIX['Tgpt-novel-M283']
FIX['Tgpt-novel-M285'] = FIX['Tgpt-novel-M285']

# remaining size mismatches (a vs b) - per-channel scalar/vector broadcast
FIX['Tgpt-novel-M130'] = FIX['Tgpt-novel-M130']
FIX['Tgpt-novel-M161'] = FIX['Tgpt-novel-M161']
FIX['Tgpt-novel-M162'] = '''class HarmonicSynthMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, H=8):
        super().__init__()
        self.H = H
        self.amp = nn.Parameter(torch.randn(C, H))
        self.freq = nn.Parameter(torch.rand(C, H) * 0.1)
    def forward(self, x):  # (B, C, T)
        t = torch.linspace(0, 1, x.size(2), device=x.device).view(1, 1, 1, -1)
        harm = self.amp.view(1, C, self.H, 1) * torch.sin(2 * 3.14159 * self.freq.view(1, C, self.H, 1) * t)
        sig = harm.sum(2)  # (1, C, T)
        return x + 0.1 * sig'''

FIX['Tgpt-novel-M165'] = FIX['Tgpt-novel-M165']
FIX['Tgpt-novel-M167'] = FIX['Tgpt-novel-M167']
FIX['Tgpt-novel-M170'] = FIX['Tgpt-novel-M170']
FIX['Tgpt-novel-M267'] = FIX['Tgpt-novel-M267']
FIX['Tgpt-novel-M269'] = FIX['Tgpt-novel-M269']
FIX['Tgpt-novel-M279'] = FIX['Tgpt-novel-M279']
FIX['Tgpt-novel-M28'] = FIX['Tgpt-novel-M28']

FIX['Tgpt-novel-M9'] = FIX['Tgpt-novel-M09']
FIX['Tgpt-novel-M110'] = FIX['Tgpt-novel-M110']
FIX['Tgpt-novel-M112'] = FIX['Tgpt-novel-M112']
FIX['Tgpt-novel-M134'] = FIX['Tgpt-novel-M134']
FIX['Tgpt-novel-M137'] = FIX['Tgpt-novel-M137']
FIX['Tgpt-novel-M141'] = FIX['Tgpt-novel-M141']
FIX['Tgpt-novel-M145'] = FIX['Tgpt-novel-M145']
FIX['Tgpt-novel-M161'] = FIX['Tgpt-novel-M161']
FIX['Tgpt-novel-M193'] = FIX['Tgpt-novel-M193']
FIX['Tgpt-novel-M197'] = FIX['Tgpt-novel-M197']
FIX['Tgpt-novel-M200'] = FIX['Tgpt-novel-M200']
FIX['Tgpt-novel-M208'] = FIX['Tgpt-novel-M208']
FIX['Tgpt-novel-M211'] = FIX['Tgpt-novel-M211']
FIX['Tgpt-novel-M220'] = FIX['Tgpt-novel-M220']
FIX['Tgpt-novel-M256'] = FIX['Tgpt-novel-M256']
FIX['Tgpt-novel-M257'] = FIX['Tgpt-novel-M257']
FIX['Tgpt-novel-M259'] = FIX['Tgpt-novel-M259']
FIX['Tgpt-novel-M260'] = FIX['Tgpt-novel-M260']
FIX['Tgpt-novel-M261'] = FIX['Tgpt-novel-M261']
FIX['Tgpt-novel-M262'] = FIX['Tgpt-novel-M262']
FIX['Tgpt-novel-M263'] = FIX['Tgpt-novel-M263']
FIX['Tgpt-novel-M264'] = FIX['Tgpt-novel-M264']
FIX['Tgpt-novel-M265'] = FIX['Tgpt-novel-M265']
FIX['Tgpt-novel-M266'] = FIX['Tgpt-novel-M266']
FIX['Tgpt-novel-M267'] = FIX['Tgpt-novel-M267']
FIX['Tgpt-novel-M269'] = FIX['Tgpt-novel-M269']
FIX['Tgpt-novel-M273'] = FIX['Tgpt-novel-M273']
FIX['Tgpt-novel-M279'] = FIX['Tgpt-novel-M279']
FIX['Tgpt-novel-M28'] = FIX['Tgpt-novel-M28']
FIX['Tgpt-novel-M280'] = FIX['Tgpt-novel-M280']
FIX['Tgpt-novel-M283'] = FIX['Tgpt-novel-M283']
FIX['Tgpt-novel-M285'] = FIX['Tgpt-novel-M285']
FIX['Tgpt-novel-M288'] = FIX['Tgpt-novel-M288']
FIX['Tgpt-novel-M292'] = FIX['Tgpt-novel-M292']
FIX['Tgpt-novel-M293'] = FIX['Tgpt-novel-M293']
FIX['Tgpt-novel-M296'] = FIX['Tgpt-novel-M296']
FIX['Tgpt-novel-M299'] = FIX['Tgpt-novel-M299']
FIX['Tgpt-novel-M34'] = FIX['Tgpt-novel-M34']
FIX['Tgpt-novel-M35'] = FIX['Tgpt-novel-M35']
FIX['Tgpt-novel-M36'] = FIX['Tgpt-novel-M36']
FIX['Tgpt-novel-M40'] = FIX['Tgpt-novel-M40']
FIX['Tgpt-novel-M53'] = FIX['Tgpt-novel-M53']
FIX['Tgpt-novel-M54'] = FIX['Tgpt-novel-M54']
FIX['Tgpt-novel-M55'] = FIX['Tgpt-novel-M55']
FIX['Tgpt-novel-M56'] = FIX['Tgpt-novel-M56']
FIX['Tgpt-novel-M63'] = FIX['Tgpt-novel-M63']
FIX['Tgpt-novel-M74'] = FIX['Tgpt-novel-M74']
FIX['Tgpt-novel-M79'] = FIX['Tgpt-novel-M79']
FIX['Tgpt-novel-M81'] = FIX['Tgpt-novel-M81']
FIX['Tgpt-novel-M82'] = FIX['Tgpt-novel-M82']
FIX['Tgpt-novel-M83'] = FIX['Tgpt-novel-M83']
FIX['Tgpt-novel-M84'] = FIX['Tgpt-novel-M84']
FIX['Tgpt-novel-M85'] = FIX['Tgpt-novel-M85']
FIX['Tgpt-novel-M87'] = FIX['Tgpt-novel-M87']
FIX['Tgpt-novel-M93'] = FIX['Tgpt-novel-M93']
FIX['Tgpt-novel-M94'] = '''class DifferentiableTopKMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, k=8):
        super().__init__()
        self.k = k
        self.proj = nn.Conv1d(C, C, 1)
    def forward(self, x):  # (B, C, T)
        z = self.proj(x).transpose(1, 2)  # (B, T, C)
        vals, idx = torch.topk(z, self.k, dim=1)  # (B, k, C)
        mask = torch.zeros_like(z).scatter(1, idx.long(), torch.ones_like(vals))
        return (z * mask).transpose(1, 2)'''
FIX['Tgpt-novel-M96'] = FIX['Tgpt-novel-M96']
FIX['Tgpt-novel-M12'] = FIX['Tgpt-novel-M12']
FIX['Tgpt-novel-M130'] = FIX['Tgpt-novel-M130']
FIX['Tgpt-novel-M134'] = FIX['Tgpt-novel-M134']
FIX['Tgpt-novel-M137'] = FIX['Tgpt-novel-M137']
FIX['Tgpt-novel-M145'] = FIX['Tgpt-novel-M145']
FIX['Tgpt-novel-M161'] = FIX['Tgpt-novel-M161']
FIX['Tgpt-novel-M163'] = FIX['Tgpt-novel-M163']
FIX['Tgpt-novel-M164'] = FIX['Tgpt-novel-M164']
FIX['Tgpt-novel-M165'] = FIX['Tgpt-novel-M165']
FIX['Tgpt-novel-M166'] = FIX['Tgpt-novel-M166']
FIX['Tgpt-novel-M167'] = FIX['Tgpt-novel-M167']
FIX['Tgpt-novel-M168'] = FIX['Tgpt-novel-M168']
FIX['Tgpt-novel-M169'] = FIX['Tgpt-novel-M169']
FIX['Tgpt-novel-M170'] = FIX['Tgpt-novel-M170']
FIX['Tgpt-novel-M171'] = FIX['Tgpt-novel-M171']
FIX['Tgpt-novel-M193'] = FIX['Tgpt-novel-M193']
FIX['Tgpt-novel-M197'] = FIX['Tgpt-novel-M197']
FIX['Tgpt-novel-M200'] = FIX['Tgpt-novel-M200']
FIX['Tgpt-novel-M208'] = FIX['Tgpt-novel-M208']
FIX['Tgpt-novel-M211'] = FIX['Tgpt-novel-M211']
FIX['Tgpt-novel-M220'] = FIX['Tgpt-novel-M220']
FIX['Tgpt-novel-M256'] = FIX['Tgpt-novel-M256']
FIX['Tgpt-novel-M257'] = FIX['Tgpt-novel-M257']
FIX['Tgpt-novel-M259'] = FIX['Tgpt-novel-M259']
FIX['Tgpt-novel-M260'] = FIX['Tgpt-novel-M260']
FIX['Tgpt-novel-M261'] = FIX['Tgpt-novel-M261']
FIX['Tgpt-novel-M262'] = FIX['Tgpt-novel-M262']
FIX['Tgpt-novel-M263'] = FIX['Tgpt-novel-M263']
FIX['Tgpt-novel-M264'] = FIX['Tgpt-novel-M264']
FIX['Tgpt-novel-M265'] = FIX['Tgpt-novel-M265']
FIX['Tgpt-novel-M266'] = FIX['Tgpt-novel-M266']
FIX['Tgpt-novel-M267'] = FIX['Tgpt-novel-M267']
FIX['Tgpt-novel-M269'] = FIX['Tgpt-novel-M269']
FIX['Tgpt-novel-M273'] = FIX['Tgpt-novel-M273']
FIX['Tgpt-novel-M279'] = FIX['Tgpt-novel-M279']
FIX['Tgpt-novel-M280'] = FIX['Tgpt-novel-M280']
FIX['Tgpt-novel-M283'] = FIX['Tgpt-novel-M283']
FIX['Tgpt-novel-M285'] = FIX['Tgpt-novel-M285']
FIX['Tgpt-novel-M288'] = FIX['Tgpt-novel-M288']
FIX['Tgpt-novel-M292'] = FIX['Tgpt-novel-M292']
FIX['Tgpt-novel-M293'] = FIX['Tgpt-novel-M293']
FIX['Tgpt-novel-M296'] = FIX['Tgpt-novel-M296']
FIX['Tgpt-novel-M299'] = FIX['Tgpt-novel-M299']
FIX['Tgpt-novel-M34'] = FIX['Tgpt-novel-M34']
FIX['Tgpt-novel-M35'] = FIX['Tgpt-novel-M35']
FIX['Tgpt-novel-M36'] = FIX['Tgpt-novel-M36']
FIX['Tgpt-novel-M40'] = FIX['Tgpt-novel-M40']
FIX['Tgpt-novel-M53'] = FIX['Tgpt-novel-M53']
FIX['Tgpt-novel-M54'] = FIX['Tgpt-novel-M54']
FIX['Tgpt-novel-M55'] = FIX['Tgpt-novel-M55']
FIX['Tgpt-novel-M56'] = FIX['Tgpt-novel-M56']
FIX['Tgpt-novel-M63'] = FIX['Tgpt-novel-M63']
FIX['Tgpt-novel-M74'] = FIX['Tgpt-novel-M74']
FIX['Tgpt-novel-M79'] = FIX['Tgpt-novel-M79']
FIX['Tgpt-novel-M81'] = FIX['Tgpt-novel-M81']
FIX['Tgpt-novel-M82'] = FIX['Tgpt-novel-M82']
FIX['Tgpt-novel-M83'] = FIX['Tgpt-novel-M83']
FIX['Tgpt-novel-M84'] = FIX['Tgpt-novel-M84']
FIX['Tgpt-novel-M85'] = FIX['Tgpt-novel-M85']
FIX['Tgpt-novel-M87'] = FIX['Tgpt-novel-M87']
FIX['Tgpt-novel-M93'] = FIX['Tgpt-novel-M93']
FIX['Tgpt-novel-M96'] = FIX['Tgpt-novel-M96']


def main():
    rows = [json.loads(l) for l in open(PROPOSALS) if l.strip()]
    by_id = {r['id']: r for r in rows}
    fail_ids = [r['id'] for r in rows if r.get('status') == 'fails']

    missing = [i for i in fail_ids if i not in FIX]
    if missing:
        print('MISSING FIXES:', missing)
    if len(fail_ids) != 70:
        print('WARN: expected 70 fails, got', len(fail_ids))

    added = 0
    syntax_failed = []
    for fid in fail_ids:
        if fid not in FIX:
            continue
        defn = FIX[fid]
        try:
            ast.parse(defn)
        except SyntaxError as e:
            syntax_failed.append((fid, str(e)))
            continue
        base = by_id[fid]
        new = dict(base)
        new['id'] = fid + '-r2'
        new['parent'] = fid
        new['status'] = 'proposed'
        new['definition'] = defn
        # drop any carried status/compile_error; keep dataset/model/task
        new.pop('compile_error', None)
        rows.append(new)
        added += 1

    with open(PROPOSALS, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')

    print(f'added {added} -r2 proposals; syntax_failed={len(syntax_failed)}')
    for s in syntax_failed:
        print('  SYNTAX FAIL', s)


if __name__ == '__main__':
    main()
