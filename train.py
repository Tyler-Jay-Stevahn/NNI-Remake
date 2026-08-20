#!/usr/bin/env python3
"""train.py — train one proposal from proposals.jsonl and record results.

PyTorch training path. Reads a proposal by id, builds its model from the
spec via build_model.build_model, loads its dataset via datasets.get_dataloader,
trains for a few epochs, and appends one line to tests/results.jsonl in the
schema the dashboard already consumes (id, declared_dataset, status, val_acc,
train_loss, val_loss, inference_ms, param_count, above_chance, test).

Usage:
    python3 train.py

The script prompts interactively for the proposal id, then asks four separate
questions (epochs, batch size, learning rate, optimizer) — one prompt per item.
Leave any prompt blank to use that proposal's own config (read from
proposals.jsonl).
"""
import json
import os
import time

import torch
import torch.nn.functional as F

import build_model
import datasets
import optimizers
import losses
import schedulers
import initializers
import regularizations

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.exists(os.path.join(HERE, "proposals.jsonl")) else os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "tests", "results.jsonl")

# Optimizer-name -> torch optimizer class. Unknown names fall back to Adam.
# Tier-1 torch-native optimizers registered here (no custom class needed).
# NOTE: lbfgs requires a closure-based step (it re-evaluates the loss); the
# current train loop calls opt.step() with no closure, so lbfgs is selectable
# but will not train correctly without a loop change.
OPT_MAP = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
    "nadam": torch.optim.NAdam,
    "radam": torch.optim.RAdam,
    "adagrad": torch.optim.Adagrad,
    "adamax": torch.optim.Adamax,
    "adadelta": torch.optim.Adadelta,
    "rprop": torch.optim.Rprop,
    "asgd": torch.optim.ASGD,
    "lbfgs": torch.optim.LBFGS,
}


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


def evaluate(model, loader, device, loss_fn=None):
    """Return (avg_loss, accuracy).

    Handles 2-D classification output (B, n_classes) and 3-D text-gen output
    (B, vocab, T) where the target is (B, T) of token ids. When a custom
    loss_fn is supplied it is used for the loss sum (reduction="sum");
    otherwise the loss is standard cross-entropy.
    """
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            if loss_fn is not None:
                loss_sum += loss_fn(out, y, reduction="sum").item()
            else:
                loss_sum += F.cross_entropy(out, y, reduction="sum").item()
            if out.dim() == 3:
                # text-gen: per-token accuracy over the sequence.
                correct += (out.argmax(1) == y).sum().item()
                total += y.numel()
            else:
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def train(pid, epochs=None, batch_size=None, lr=None, optimizer=None):
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

    # Pull training config from the proposal (single source of truth).
    # CLI args (below) override only when explicitly provided.
    cfg_epochs = rec.get("epochs")
    cfg_batch = rec.get("batch")
    cfg_lr = rec.get("lr")
    cfg_opt = rec.get("optimizer")

    if epochs is None:
        epochs = int(cfg_epochs) if cfg_epochs is not None else 10
    if batch_size is None:
        batch_size = int(cfg_batch) if cfg_batch is not None else 4
    if lr is None:
        lr = float(cfg_lr) if cfg_lr is not None else 1e-3
    if optimizer is None:
        optimizer = str(cfg_opt) if cfg_opt else "adam"

    opt_name = optimizer.lower()

    # --- loss resolution ---------------------------------------------------
    # A proposal may set "loss": "custom:<name>". Resolution mirrors the
    # optimizer path; unknown custom names are a hard error. Stock CE is used
    # when no custom loss is declared.
    cfg_loss = rec.get("loss")
    if cfg_loss:
        loss_name = str(cfg_loss).lower()
    else:
        loss_name = "ce"
    if loss_name.startswith("custom:"):
        cname = loss_name[len("custom:"):]
        loss_cls = losses.get(cname)
        loss_kwargs = rec.get("loss_kwargs") or {}
        loss_fn = loss_cls(**loss_kwargs)
        print(f"  [custom loss] {cname} kwargs={loss_kwargs}", flush=True)
    else:
        # stock cross-entropy
        loss_fn = None  # F.cross_entropy will be used directly

    # --- scheduler resolution ---------------------------------------------
    # A proposal may set "scheduler": "custom:<name>". Returns an LR factor
    # each batch; multiplied by the base LR. Schedulers that manage their own
    # LR (custom optimizers flagged .manages_lr) are exempt.
    cfg_sched = rec.get("scheduler")
    sched_fn = None
    sched_kwargs = {}
    # total_steps is computed after train_dl is built (see below).
    if cfg_sched:
        sname = str(cfg_sched).lower()
        if sname.startswith("custom:"):
            sc = sname[len("custom:"):]
            sched_fn = schedulers.get(sc)
            sched_kwargs = rec.get("scheduler_kwargs") or {}
            print(f"  [custom scheduler] {sc} kwargs={sched_kwargs}", flush=True)

    # Custom optimizer strategy: "custom:<name>" -> look up in optimizers.REGISTRY.
    # Unknown custom name is a hard error (no silent fallback). Stock optimizers
    # ignore optimizer_kwargs (only forwarded to custom classes).
    if opt_name.startswith("custom:"):
        cname = opt_name[len("custom:"):]
        if cname not in optimizers.REGISTRY:
            raise SystemExit(
                f"custom optimizer {cname!r} not found in optimizers.REGISTRY. "
                f"Add it to optimizers.py with @register({cname!r}).")
        opt_cls = optimizers.REGISTRY[cname]
        opt_kwargs = rec.get("optimizer_kwargs") or {}
        print(f"  [custom optimizer] {cname} kwargs={opt_kwargs}", flush=True)
    else:
        opt_cls = OPT_MAP.get(opt_name, torch.optim.Adam)
        if opt_name not in OPT_MAP:
            print(f"  [warn] unknown optimizer {opt_name!r}; falling back to adam",
                  flush=True)
        opt_kwargs = {}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model.build_model(spec).to(device)

    # --- custom initializer (Tier B) --------------------------------------
    cfg_init = rec.get("initializer")
    if cfg_init:
        iname = str(cfg_init).lower()
        if iname.startswith("custom:"):
            init_name = iname[len("custom:"):]
            initializers.apply(init_name, model)
            print(f"  [custom initializer] {init_name}", flush=True)

    # --- custom regularization (Tier B) -----------------------------------
    reg_list = rec.get("regularization") or []
    if isinstance(reg_list, str):
        reg_list = [reg_list]
    for reg in reg_list:
        rname = str(reg).lower()
        if rname == "spectral_norm":
            regularizations.apply_spectral_norm(model)
            print("  [regularization] spectral_norm", flush=True)

    n_params = count_params(model)

    train_dl, val_dl = datasets.get_dataloader(dataset, batch_size=batch_size)
    opt = opt_cls(model.parameters(), lr=lr, **opt_kwargs)
    opt_lr = opt.defaults["lr"]

    # Total training steps for scheduler scaling (needs train_dl, defined above).
    total_steps = epochs * max(1, len(train_dl)) if epochs else 1

    # LBFGS and Sophia need a closure that re-runs forward/backward (LBFGS
    # re-evaluates the loss; Sophia estimates the diagonal Hessian). All other
    # optimizers use the plain opt.step() path with no extra cost.
    needs_closure = isinstance(opt, (torch.optim.LBFGS, optimizers.Sophia))

    def closure(x, y):
        model.zero_grad()
        out = model(x)
        if loss_fn is not None:
            loss = loss_fn(out, y, reduction="mean")
        else:
            loss = F.cross_entropy(out, y)
        # Sophia needs the graph alive for its Hutchinson HVP; retain it on the
        # closure path (LBFGS also re-evaluates, so retention is safe for both).
        loss.backward(retain_graph=needs_closure)
        return loss

    # Custom LR scheduler (Tier A): scale base LR per batch. Custom optimizers
    # that own their LR (flagged .manages_lr) are exempt so we don't double-scale.
    opt_manages_lr = getattr(opt, "manages_lr", False)

    def _apply_scheduler(step_idx):
        if sched_fn is None or opt_manages_lr:
            return
        factor = sched_fn(step_idx, total_steps, **sched_kwargs)
        for g in opt.param_groups:
            g["lr"] = opt.defaults["lr"] * factor

    print(f"Training {pid} on {dataset}  params={n_params}  "
          f"opt={opt_name}  lr={lr}  epochs={epochs}  batch={batch_size}", flush=True)
    if needs_closure:
        print(f"  [closure step] {opt_name} re-runs forward/backward per step",
              flush=True)
    start = time.time()
    n_batches = len(train_dl)
    gstep = 0
    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        for bi, (x, y) in enumerate(train_dl, 1):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            if needs_closure:
                loss = opt.step(lambda: closure(x, y))
            else:
                loss = closure(x, y)
                opt.step()
            _apply_scheduler(gstep)
            gstep += 1
            running_loss += (loss.item() if torch.is_tensor(loss) else 0.0)
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

    val_loss, val_acc = evaluate(model, val_dl, device, loss_fn=loss_fn)
    # approximate train loss from the last train step is unreliable; report val.
    # Re-run a quick train-set eval for a train_loss estimate.
    train_loss, _ = evaluate(model, train_dl, device, loss_fn=loss_fn)
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
        "loss": loss_name,
        "scheduler": sched_fn.__name__ if sched_fn is not None else "constant",
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

    epochs_s = input("Epochs [blank = use this proposal's config]: ").strip()
    epochs = int(epochs_s) if epochs_s else None

    batch_s = input("Batch size [blank = use this proposal's config]: ").strip()
    batch_size = int(batch_s) if batch_s else None

    lr_s = input("Learning rate [blank = use this proposal's config]: ").strip()
    lr = float(lr_s) if lr_s else None

    opt_s = input("Optimizer (adam/adamw/sgd/rmsprop) [blank = use this proposal's config]: ").strip()
    optimizer = opt_s if opt_s else None

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
            res = train(r["id"], epochs=epochs, batch_size=batch_size,
                        lr=lr, optimizer=optimizer)
            print(f"  {r['id']}: ok  val_acc={res['val_acc']}  above_chance={res['above_chance']}")
            done += 1
        except Exception as e:  # noqa: BLE001
            print(f"  {r['id']}: ERROR {type(e).__name__}: {e}")
            skipped += 1

    print(f"\nSweep complete: {done} trained, {skipped} skipped of {len(target)} targeted.")


if __name__ == "__main__":
    main()
