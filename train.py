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
    """Return (avg_loss, accuracy).

    Handles 2-D classification output (B, n_classes) and 3-D text-gen output
    (B, vocab, T) where the target is (B, T) of token ids.
    """
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += F.cross_entropy(out, y, reduction="sum").item()
            if out.dim() == 3:
                # text-gen: per-token accuracy over the sequence.
                correct += (out.argmax(1) == y).sum().item()
                total += y.numel()
            else:
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
    opt_name = "adam"
    opt_lr = opt.defaults["lr"]

    print(f"Training {pid} on {dataset}  params={n_params}  "
          f"epochs={epochs}  batch={batch_size}", flush=True)
    start = time.time()
    n_batches = len(train_dl)
    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        for bi, (x, y) in enumerate(train_dl, 1):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            # Live batch progress (carriage-return overwrite, no external deps).
            print(f"\r[{pid}] epoch {ep + 1}/{epochs}  "
                  f"batch {bi}/{n_batches}  loss={loss.item():.4f}",
                  end="", flush=True)
        avg_loss = running_loss / max(n_batches, 1)
        elapsed = time.time() - start
        # Epoch summary line (newline so the next epoch starts clean).
        print(f"\r[{pid}] epoch {ep + 1}/{epochs}  done  "
              f"avg_train_loss={avg_loss:.4f}  elapsed={elapsed:.1f}s",
              flush=True)
    train_time_s = time.time() - start

    val_loss, val_acc = evaluate(model, val_dl, device)
    # approximate train loss from the last train step is unreliable; report val.
    # Re-run a quick train-set eval for a train_loss estimate.
    train_loss, _ = evaluate(model, train_dl, device)
    inference_ms = (train_time_s / max(len(train_dl), 1)) * 1000.0

    # Chance baseline: image/text classification uses n_classes; a text-gen
    # task uses vocab (next-token chance = 1 / vocab). build_model.output_size
    # returns the right denominator for both (vocab for text-gen, else classes).
    n_classes = build_model.output_size(spec)
    chance = 1.0 / max(n_classes, 1)
    above_chance = val_acc > chance

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
        "text_gen": bool(spec.get("task_type") == "text-gen"),
        "optimizer": opt_name,
        "lr": round(opt_lr, 6),
        "epochs": epochs,
        "batch": batch_size,
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
    # Sequential sweep (mirrors compile_test.py): blank id -> gather all
    # 'compiles' rows, optionally filter by id-prefix / task_family, cap the
    # run to a count, then train each in order. Re-running skips rows already
    # at status 'trained', so the sweep is resumable.
    all_records = []
    try:
        with open(os.path.join(ROOT, "proposals.jsonl"), encoding="utf-8") as fh:
            all_records = [json.loads(l) for l in fh if l.strip()]
    except FileNotFoundError:
        pass

    pid = input("Proposal id (blank = sweep all 'compiles'): ").strip()
    if pid:
        target = [r for r in all_records if r.get("id") == pid]
        if not target:
            raise SystemExit(f"proposal {pid!r} not found")
    else:
        prefix = input("Id prefix filter (blank = none): ").strip()
        fam = input("Task-family filter (blank = none): ").strip()
        target = [r for r in all_records if r.get("status") == "compiles"]
        if prefix:
            target = [r for r in target if r.get("id", "").startswith(prefix)]
        if fam:
            target = [r for r in target if r.get("task_family") == fam]

    count_s = input("Max models to train (blank = all): ").strip()
    count = int(count_s) if count_s else len(target)
    if count < len(target):
        target = target[:count]

    if not target:
        print("No 'compiles' proposals match the filter.")
        return

    epochs_s = input("Epochs [default: 10]: ").strip()
    batch_s = input("Batch size [default: 4]: ").strip()
    epochs = int(epochs_s) if epochs_s else 10
    batch_size = int(batch_s) if batch_s else 4

    done, skipped = 0, 0
    for r in target:
        # Re-read current status in case a prior partial run advanced it.
        try:
            cur = load_proposal(r["id"])
        except KeyError:
            print(f"  {r['id']}: missing (skipped)")
            skipped += 1
            continue
        if cur.get("status") == "trained":
            print(f"  {r['id']}: already trained (skipped)")
            skipped += 1
            continue
        if cur.get("status") != "compiles":
            print(f"  {r['id']}: status {cur.get('status')!r}, not 'compiles' (skipped)")
            skipped += 1
            continue
        try:
            res = train(r["id"], epochs=epochs, batch_size=batch_size)
            print(f"  {r['id']}: ok  val_acc={res['val_acc']}  above_chance={res['above_chance']}")
            done += 1
        except Exception as e:  # noqa: BLE001
            print(f"  {r['id']}: ERROR {type(e).__name__}: {e}")
            skipped += 1

    print(f"\nSweep complete: {done} trained, {skipped} skipped of {len(target)} targeted.")


if __name__ == "__main__":
    main()
