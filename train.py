#!/usr/bin/env python3
"""train.py — train one proposal from proposals.jsonl and record results.

PyTorch training path. Reads a proposal by id, builds its model from the
spec via build_model.build_model, loads its dataset via datasets.get_dataloader,
trains for a few epochs, and appends one line to tests/results.jsonl in the
schema the dashboard already consumes (id, declared_dataset, status, val_acc,
train_loss, val_loss, inference_ms, param_count, above_chance, test).

Usage:
    python3 train.py

The script prompts interactively for the proposal id, number of epochs, and
batch size. Press ENTER to accept the shown default.
"""
import json
import os
import time

import torch
import torch.nn.functional as F

import build_model
import datasets

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.exists(os.path.join(HERE, "proposals.jsonl")) else os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "tests", "results.jsonl")


def load_proposal(pid):
    with open(os.path.join(ROOT, "proposals.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("id") == pid:
                return rec
    raise KeyError(f"proposal {pid!r} not found in proposals.jsonl")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += F.cross_entropy(out, y, reduction="sum").item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def train(pid, epochs=3, batch_size=32):
    rec = load_proposal(pid)
    spec = rec.get("spec", {}) or {}
    dataset = spec.get("dataset")
    if dataset is None:
        raise ValueError(f"proposal {pid!r} has no spec.dataset")

    # Strict gate: only train proposals that compiled. Preserves "fails".
    if rec.get("status") != "compiles":
        raise SystemExit(
            f"proposal {pid!r} status is {rec.get('status')!r}, not 'compiles'. "
            f"Run compile_test.py first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model.build_model(spec).to(device)
    n_params = count_params(model)

    train_dl, val_dl = datasets.get_dataloader(dataset, batch_size=batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    start = time.time()
    for ep in range(epochs):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
    train_time_s = time.time() - start

    val_loss, val_acc = evaluate(model, val_dl, device)
    # approximate train loss from the last train step is unreliable; report val.
    # Re-run a quick train-set eval for a train_loss estimate.
    train_loss, _ = evaluate(model, train_dl, device)
    inference_ms = (train_time_s / max(len(train_dl), 1)) * 1000.0

    above_chance = val_acc > (1.0 / max(build_model.num_classes(dataset), 1))

    result = {
        "id": pid,
        "declared_dataset": dataset,
        "status": "ok",
        "val_acc": round(val_acc, 4),
        "train_loss": round(train_loss, 4),
        "val_loss": round(val_loss, 4),
        "inference_ms": round(inference_ms, 4),
        "param_count": n_params,
        "above_chance": bool(above_chance),
        "test": "real",
    }

    with open(RESULTS, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(result) + "\n")

    # Advance the proposal's lifecycle status to "trained" in proposals.jsonl.
    _set_status(pid, "trained")

    return result


def _set_status(pid, status):
    """Rewrite proposals.jsonl setting `status` for one proposal id."""
    path = os.path.join(ROOT, "proposals.jsonl")
    with open(path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    for rec in records:
        if rec.get("id") == pid:
            rec["status"] = status
            break
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def main():
    # Default id: the first proposal whose status is "compiles"; else the
    # legacy default. Press ENTER to accept.
    default_pid = "Thpo-mnist-M01"
    try:
        with open(os.path.join(ROOT, "proposals.jsonl"), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("status") == "compiles":
                    default_pid = rec["id"]
                    break
    except FileNotFoundError:
        pass

    pid = input(f"Proposal id [default: {default_pid}]: ").strip() or default_pid
    epochs_s = input("Epochs [default: 10]: ").strip()
    batch_s = input("Batch size [default: 32]: ").strip()
    epochs = int(epochs_s) if epochs_s else 10
    batch_size = int(batch_s) if batch_s else 32

    res = train(pid, epochs=epochs, batch_size=batch_size)
    print("Trained", res["id"], "->", res)


if __name__ == "__main__":
    main()
