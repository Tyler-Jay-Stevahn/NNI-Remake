#!/usr/bin/env python3
"""compile_test.py — compile gate that updates proposal status in place.

For every proposal whose status is "proposed", verify it can:
  1. build from its spec (build_model.build_model),
  2. run a forward pass,
  3. train on 2 real samples from its own dataset (datasets.get_dataloader).
On success the proposal's status becomes "compiles"; on any failure it
becomes "fails". The status is written back into proposals.jsonl (no
separate results file). This replaces the old TF smoke-test stage.

Usage:
    python3 compile_test.py

The script prompts interactively. Leave the proposal id blank to process all
"proposed" proposals, or type a specific id to process only that one.
"""
import json
import os

import torch
import torch.nn.functional as F

import build_model
import datasets

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.exists(os.path.join(HERE, "proposals.jsonl")) else os.path.dirname(HERE)
SRC = os.path.join(ROOT, "proposals.jsonl")

N_SAMPLES = 2


def load_all():
    with open(SRC, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def save_all(records):
    with open(SRC, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def two_samples(dataset):
    """Return a (x, y) batch of exactly N_SAMPLES from the dataset loader."""
    train_dl, _ = datasets.get_dataloader(dataset, batch_size=N_SAMPLES)
    for x, y in train_dl:
        if x.size(0) >= N_SAMPLES:
            return x[:N_SAMPLES], y[:N_SAMPLES]
    # tiny dataset: return whatever we got
    return x, y


def compile_one(rec):
    spec = rec.get("spec", {}) or {}
    dataset = spec.get("dataset")
    if dataset is None:
        raise ValueError("proposal has no spec.dataset")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model.build_model(spec).to(device)
    x, y = two_samples(dataset)
    x, y = x.to(device), y.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    out = model(x)
    # Shape-aware loss for the 2-sample smoke step:
    #  - 2-D output (B, n_out): standard classification -> cross_entropy.
    #  - 4-D spatial output (B, n_out, H, W), e.g. a diffusion/conv_out head:
    #    cross_entropy expects class logits, not a spatial tensor, so compare
    #    against a one-hot-as-channels target of the same shape via MSE. This
    #    still exercises the full forward+backward path for the gate.
    if out.dim() == 2:
        loss = F.cross_entropy(out, y)
    elif out.dim() == 4:
        target = torch.nn.functional.one_hot(y, num_classes=out.shape[1])
        target = target.permute(0, 3, 1, 2).float().to(device)
        loss = F.mse_loss(out, target)
    else:
        raise ValueError(f"unexpected output rank {out.dim()} for compile gate")
    loss.backward()
    opt.step()
    return True


def process(rec):
    try:
        compile_one(rec)
        rec["status"] = "compiles"
        return "compiles"
    except Exception as e:  # noqa: BLE001
        rec["status"] = "fails"
        rec["compile_error"] = f"{type(e).__name__}: {e}"
        return f"fails ({rec['compile_error']})"


def main():
    all_records = load_all()

    pid = input("Proposal id (blank = all 'proposed'): ").strip()
    if pid:
        target = [r for r in all_records if r.get("id") == pid]
        if not target:
            raise SystemExit(f"proposal {pid!r} not found")
    else:
        target = all_records

    changed = []
    for rec in target:
        if rec.get("status") != "proposed":
            continue
        result = process(rec)
        changed.append((rec["id"], result))
        print(f"{rec['id']}: {result}")

    if changed:
        # Save the FULL list (target is a filtered view into the same
        # objects, so process() mutations are reflected). Saving only
        # `target` would erase every other proposal from the file.
        save_all(all_records)
        print(f"\nUpdated {len(changed)} proposal(s) in proposals.jsonl")
    else:
        print("No 'proposed' proposals to process.")


if __name__ == "__main__":
    main()
