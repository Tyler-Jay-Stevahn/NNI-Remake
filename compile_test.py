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
    loss = F.cross_entropy(out, y)
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
    records = load_all()

    pid = input("Proposal id (blank = all 'proposed'): ").strip()
    if pid:
        records = [r for r in records if r.get("id") == pid]
        if not records:
            raise SystemExit(f"proposal {pid!r} not found")

    changed = []
    for rec in records:
        if rec.get("status") != "proposed":
            continue
        result = process(rec)
        changed.append((rec["id"], result))
        print(f"{rec['id']}: {result}")

    if changed:
        save_all(records)
        print(f"\nUpdated {len(changed)} proposal(s) in proposals.jsonl")
    else:
        print("No 'proposed' proposals to process.")


if __name__ == "__main__":
    main()
