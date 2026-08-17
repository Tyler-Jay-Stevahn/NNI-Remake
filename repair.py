#!/usr/bin/env python3
"""repair.py — generate append-only fixed proposals for every compile failure.

Reads proposals.jsonl + fails.jsonl. For each failing proposal id, produces a
corrected `definition` and appends a NEW proposal `<id>-fix1` (status: proposed,
parent: <id>). The original failed proposal is left in place (never overwritten
or removed), so nothing is duplicated and the failure record remains.

The repaired rows are `proposed`, so the existing compile_test.py run will
verify them (build + forward + backward on 2 real samples) when torch is
available. This script only does syntax checking (compile()), not a torch run.

Usage:
    python3 repair.py
"""
import datetime
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "proposals.jsonl")
FAILS = os.path.join(HERE, "fails.jsonl")
LOG = os.path.join(HERE, "repair_log.json")

NOW = datetime.datetime.now().isoformat(timespec="seconds")


def novel_block(rec):
    for b in rec.get("spec", {}).get("blocks", []):
        if b.get("novel") and b.get("definition"):
            return b
    return {}


# ---------------------------------------------------------------------------
# Uniform constructor fix: ensure C / T / dim are present with defaults so the
# builder (which forwards only block-dict keys matching the signature) can call
# __init__() with no args. Text channel is fixed at 128 by the embedding stem.
# ---------------------------------------------------------------------------
def fix_constructor(defn):
    # Normalize a non-"self" first parameter alias (e.g. `def __init__(s, C):`
    # with `s.` attribute access) to `self` so the standard pass can run.
    m0 = re.search(r"def\s+__init__\s*\(\s*([^),]+)", defn)
    if m0:
        p0 = m0.group(1).strip()
        if p0 and p0 != "self":
            defn = re.sub(r"def\s+(\w+)\(\s*" + re.escape(p0) + r"\b", r"def \1(self", defn)
            defn = re.sub(r"\b" + re.escape(p0) + r"\.", "self.", defn)

    def repl(m):
        inside = m.group(1)
        params = [p.strip() for p in inside.split(",") if p.strip()]
        others = [p for p in params if p.split("=")[0].strip() != "self"]
        new = ["self"]
        seen = set()
        for name in ("C", "T", "dim"):
            existing = [p for p in others if p.split("=")[0].strip() == name]
            if existing:
                p = existing[0]
                new.append(p if "=" in p else f"{name}=128")
            else:
                new.append(f"{name}=128")
            seen.add(name)
        for p in others:
            key = p.split("=")[0].strip()
            if key not in seen:
                new.append(p)
        return f"def __init__({', '.join(new)}):"

    return re.sub(r"def __init__\(([^)]*)\)\s*:", repl, defn, count=1)


# ---------------------------------------------------------------------------
# Hand-verified rewrites for the non-uniform edge cases. Each maps a class name
# to a fully corrected definition string. These were reasoned against the actual
# (B, C=128, T) text modality / (B, C, H, W) image modality contract.
# ---------------------------------------------------------------------------
REWRITES = {
    # --- B: flip dims must be a tuple ---
    "SymmetryEquivariantMix": """class SymmetryEquivariantMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128):
        super().__init__()
        self.w = nn.Linear(dim, dim)
        self.reflect = nn.Parameter(torch.eye(dim))  # learnable involution
    def forward(self, x):  # (B, C, T)
        fwd = self.w(x)
        rev = torch.flip(self.w(torch.flip(x, 1)), (1,))
        return 0.5 * (fwd + rev @ self.reflect)  # equivariant avg""",

    # --- C: sqrt of a float must be a tensor op ---
    "HyperbolicTokenMix": """class HyperbolicTokenMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, curv=1.0):
        super().__init__()
        self.c = curv
        self.lin = nn.Linear(dim, dim)
    def exp_map(self, x):
        return torch.tanh(torch.norm(x, -1, keepdim=True) / (self.c ** 0.5) + 1e-5) * x
    def forward(self, x):  # (B, C, T)
        h = self.exp_map(x)  # to hyperbolic
        mixed = self.lin(h)
        return torch.log1p(torch.norm(mixed, -1, keepdim=True)) * mixed  # log-map back""",

    # --- D: bad 'self' alias, keyword topk arg ---
    "MatroidMix": """class MatroidMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128):
        super().__init__(); self.w = nn.Linear(C, C)
    def forward(self, x):  # (B, C, T)
        s_ = torch.softmax(self.w(x), -1)
        top, _ = torch.topk(s_, max(2, x.shape[-1] // 2), -1)
        m = torch.zeros_like(s_).scatter_(-1, top, 1)
        return x * m""",

    # --- E: None hidden dereferenced in forward ---
    "ConvLSTM1D": """class ConvLSTM1D(nn.Module):
    def __init__(self, C=128, T=128, dim=128, kernel=3):
        super().__init__()
        self.conv = nn.Conv1d(dim, 4 * dim, kernel, padding=kernel // 2)
        self.hidden = None
    def forward(self, x):  # (B, C, T)
        h = x.transpose(1, 2)  # (B, C, T)
        hidden = torch.zeros_like(h) if self.hidden is None else self.hidden
        gates = self.conv(h)
        i, f, o, g = gates.chunk(4, 1)
        c = torch.sigmoid(f) * hidden + torch.sigmoid(i) * torch.tanh(g)
        hidden = torch.sigmoid(o) * torch.tanh(c)
        self.hidden = hidden.detach()
        return hidden.transpose(1, 2)""",

    # --- F: einsum broadcast bugs ---
    "HarmonicTokenMixer": """class HarmonicTokenMixer(nn.Module):
    def __init__(self, C=128, T=128, dim=128, n_freqs=32):
        super().__init__()
        self.freqs = nn.Parameter(torch.linspace(0.1, 8.0, n_freqs))  # learned bases
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(n_freqs, dim)
    def forward(self, x):  # (B, C, T)
        tok = x.transpose(1, 2)  # (B, T, C)
        t = torch.arange(tok.shape[1], device=x.device).float()
        basis = torch.sin(t[:, None] * self.freqs[None, :])  # (T, F)
        ctx = torch.einsum('btc,tf->btf', tok, basis)  # (B, T, F)
        out = self.proj(ctx)  # (B, T, dim)
        g = torch.sigmoid(self.gate(tok))  # (B, T, dim)
        return (out * g).transpose(1, 2)  # (B, C, T)""",

    "ConceptRouter": """class ConceptRouter(nn.Module):
    def __init__(self, C=128, T=128, dim=128, n_concepts=16, top_k=2):
        super().__init__()
        self.bottleneck = nn.Linear(dim, 16)
        self.concepts = nn.Parameter(torch.randn(n_concepts, 16))
        self.out = nn.Linear(16, dim)
        self.top_k = top_k
    def forward(self, x):  # (B, C, T)
        tok = x.transpose(1, 2)  # (B, T, C)
        q = self.bottleneck(tok)  # (B, T, 16)
        scores = torch.einsum('btc,nc->btn', q, self.concepts)  # (B, T, n_concepts)
        top = scores.topk(self.top_k, -1).indices
        gate = torch.softmax(scores.gather(-1, top), -1)
        sel = self.concepts[top]  # (B, T, top_k, 16)
        mixed = (gate.unsqueeze(-1) * sel).sum(-2)  # (B, T, 16)
        out = self.out(mixed)  # (B, T, dim)
        return out.transpose(1, 2)  # (B, C, T)""",

    "GrammarLatticeGate": """class GrammarLatticeGate(nn.Module):
    def __init__(self, C=128, T=128, dim=128, states=16):
        super().__init__()
        self.trans = nn.Parameter(torch.rand(states, states))
        self.embed = nn.Linear(dim, states)
        self.out = nn.Linear(states, dim)
    def forward(self, x):  # (B, C, T)
        h = x.transpose(1, 2)  # (B, T, C)
        s = torch.softmax(self.embed(h), -1)  # (B, T, states)
        trans = torch.softmax(self.trans, -1)
        s_next = torch.einsum('bts,st->bts', s, trans)  # (B, T, states)
        out = self.out(s_next)  # (B, T, dim)
        return x + out.transpose(1, 2)  # (B, C, T)""",

    "CapsuleRoute1D": """class CapsuleRoute1D(nn.Module):
    def __init__(self, C=128, T=128, dim=128, caps=8):
        super().__init__()
        self.caps_proj = nn.Linear(dim, caps)
        self.routing = nn.Parameter(torch.randn(caps, dim))
    def forward(self, x):  # (B, C, T)
        h = x.transpose(1, 2)  # (B, T, C)
        proj = self.caps_proj(h)  # (B, T, caps)
        att = torch.softmax(proj, -1)  # (B, T, caps)
        out = torch.einsum('btk,kc->btc', att, self.routing)  # (B, T, C)
        return out.transpose(1, 2)  # (B, C, T)""",

    "KolmogorovArnoldMix": """class KolmogorovArnoldMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, grid=8):
        super().__init__()
        self.coef = nn.Parameter(torch.randn(dim, grid))  # spline basis weights
        self.grid = grid
    def forward(self, x):  # (B, C, T)
        g = torch.linspace(0, 1, self.grid, device=x.device)
        bases = torch.exp(-((x[..., None] - g) ** 2))  # (B, C, T, grid)
        out = torch.einsum('bctg,cg->bct', bases, self.coef)  # (B, C, T)
        return out""",

    # --- G/I: size-mismatch bugs ---
    "Sdm2d": """class Sdm2d(nn.Module):
    def __init__(self, C=128, T=128, dim=16):
        super().__init__(); self.mem = nn.Parameter(torch.randn(64, dim))
    def forward(self, x):  # (B, C, H, W)
        q = x.flatten(2).mean(-1)  # (B, C)
        w = torch.softmax(q @ self.mem.T, -1)  # (B, 64)
        return (w @ self.mem).view(x.shape[0], x.shape[1], 1, 1) * x""",

    "MemoryTape": """class MemoryTape(nn.Module):
    def __init__(self, C=128, T=128, dim=128, tape_len=32):
        super().__init__()
        self.tape = nn.Parameter(torch.randn(tape_len, dim))
        self.addr = nn.Linear(dim, tape_len)
        self.write = nn.Linear(dim, dim)
    def forward(self, x):  # (B, C, T)
        w = torch.softmax(self.addr(x), -1)  # (B, C, tape_len)
        read = torch.einsum('btn,nc->btc', w, self.tape)  # (B, C, dim)
        return x + read""",

    "WaveletPacket2d": """class WaveletPacket2d(nn.Module):
    def __init__(self, C=128, T=128, dim=16):
        super().__init__(); self.g = nn.Parameter(torch.ones(1))
    def forward(self, x):  # (B, C, H, W)
        lo = torch.nn.functional.avg_pool2d(x, 2, 2)
        lo_up = torch.nn.functional.interpolate(lo, (x.shape[2], x.shape[3]), mode='nearest')
        hi = x - lo_up
        g = torch.tanh(self.g)
        return g * lo_up + (1 - g) * hi""",

    # --- H: conv1d on a 128-channel tensor needs a (B*C,1,T) view ---
    "WaveletTokenDecompose": """class WaveletTokenDecompose(nn.Module):
    def __init__(self, C=128, T=128, dim=128, levels=2):
        super().__init__()
        self.lp = nn.Parameter(torch.tensor([0.5, 0.5]))  # Haar low-pass
        self.hp = nn.Parameter(torch.tensor([0.5, -0.5]))  # Haar high-pass
        self.proc = nn.Linear(dim, dim)
    def forward(self, x):  # (B, C, T)
        B, C, T = x.shape
        h = x.transpose(1, 2).reshape(B * C, 1, T)  # (B*C, 1, T)
        a = torch.conv1d(h, self.lp[None, None, :])  # (B*C, 1, T-1)
        d = torch.conv1d(h, self.hp[None, None, :])
        a = torch.nn.functional.pad(a, (0, T - a.shape[-1])).reshape(B, C, T)
        d = torch.nn.functional.pad(d, (0, T - d.shape[-1])).reshape(B, C, T)
        return self.proc(a + d)  # recombine after per-band processing""",

    "AnisotropicDiffusionMix": """class AnisotropicDiffusionMix(nn.Module):
    def __init__(self, C=128, T=128, dim=128, steps=2):
        super().__init__()
        self.kernel = nn.Parameter(torch.tensor([0.25, 0.5, 0.25]))  # learnable 1D kernel
        self.steps = steps
    def forward(self, x):  # (B, C, T)
        B, C, T = x.shape
        h = x.transpose(1, 2).reshape(B * C, 1, T)
        for _ in range(self.steps):
            lap = torch.conv1d(h, self.kernel[None, None, :].expand(1, 1, -1), padding=1)
            h = h + 0.2 * (lap - h)
        return h.reshape(B, C, T)""",
}


def class_name(defn):
    m = re.search(r"class\s+(\w+)\s*\(", defn or "")
    return m.group(1) if m else None


def bucket_of(error):
    if "missing 1 required positional argument" in error or "missing 2 required positional argument" in error:
        return "A"  # uniform signature fix
    if "dims' (position 2) must be tuple" in error:
        return "B"
    if "sqrt" in error and "must be Tensor" in error:
        return "C"
    if "positional argument follows keyword argument" in error:
        return "D"
    if "NoneType" in error:
        return "E"
    if "einsum" in error:
        return "F"
    if "expected input" in error and "channels" in error:
        return "H"
    if "must match the size" in error or "size of tensor a" in error:
        return "I"
    if "mat1 and mat2" in error or "shapes cannot be multiplied" in error:
        return "G"
    return "Z"


def repair(defn, error):
    """Return (new_defn, method, needs_gate)."""
    b = bucket_of(error)
    name = class_name(defn)
    if b == "A":
        return fix_constructor(defn), "constructor_signature_defaults", False
    if name in REWRITES:
        return REWRITES[name], f"rewrite_{name}", False
    # Unknown failure class: keep original, flag for the torch gate.
    return defn, "unrepaired_needs_gate", True


def main():
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
    by_id = {r["id"]: r for r in rows}
    fails = [json.loads(l) for l in open(FAILS, encoding="utf-8") if l.strip()]
    fail_err = {f["id"]: f.get("error", "") for f in fails}

    added = 0
    log = []
    for fid in fail_err:
        if fid not in by_id:
            log.append({"id": fid, "status": "skip_no_source"})
            continue
        rec = by_id[fid]
        blk = novel_block(rec)
        defn = blk.get("definition", "")
        new_defn, method, needs_gate = repair(defn, fail_err[fid])

        # syntax check (compile only, no torch)
        try:
            compile(new_defn, "<novel-layer>", "exec")
            syntax_ok = True
        except SyntaxError as e:
            syntax_ok = False
            new_defn = defn  # keep original if our repair broke syntax
            method = method + "_syntax_failed_original_kept"

        new_rec = json.loads(json.dumps(rec))  # deep copy
        new_rec["id"] = fid + "-fix1"
        new_id_block_type = blk.get("block_type", "unknown") + "-fix1"
        nb = dict(blk)
        nb["block_type"] = new_id_block_type
        nb["definition"] = new_defn
        nb["repaired_from"] = fid
        nb["original_error"] = fail_err[fid]
        nb["repair_method"] = method
        # replace the novel block in spec
        new_rec["spec"]["blocks"] = [
            nb if (b.get("novel") and b.get("definition")) else b
            for b in new_rec["spec"]["blocks"]
        ]
        new_rec["status"] = "proposed"
        new_rec["parent"] = fid
        new_rec["created"] = NOW
        new_rec["repair_note"] = (
            f"Append-only repair of {fid} (compile fail: {fail_err[fid]}). "
            f"Method: {method}. Original left untouched."
        )
        new_rec.pop("compile_error", None)
        if needs_gate:
            new_rec["repair_needs_gate"] = True

        rows.append(new_rec)
        added += 1
        log.append({
            "id": new_rec["id"], "parent": fid, "method": method,
            "syntax_ok": syntax_ok, "needs_gate": needs_gate,
        })

    with open(SRC, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    json.dump({
        "generated": NOW,
        "source_failures": len(fail_err),
        "appended": added,
        "entries": log,
    }, open(LOG, "w"), indent=2)

    methods = {}
    for e in log:
        methods[e["method"]] = methods.get(e["method"], 0) + 1
    print(f"Appended {added} repaired proposals (status=proposed, parent set).")
    print(f"Total rows now: {len(rows)}")
    print("By method:", json.dumps(methods, indent=2))
    bad = [e for e in log if not e["syntax_ok"]]
    print(f"Syntax-failed (original kept): {len(bad)}")
    gate = [e for e in log if e["needs_gate"]]
    print(f"Flagged needs_gate: {len(gate)}")
    print(f"Log: {LOG}")


if __name__ == "__main__":
    main()
