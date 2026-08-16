#!/usr/bin/env python3
"""download_datasets.py — pre-download datasets into ./data.

For torchvision datasets (mnist, cifar10, cifar100) this asks torchvision to
download them by constructing the loader, mirroring how MNIST was fetched.
For text datasets (tinystories) this pulls a small slice from HuggingFace
(roneneldan/TinyStories) into data/tinystories.txt.

Usage:
    python3 download_datasets.py

Prompts for which datasets to fetch (blank = all). Type a comma-separated
list (e.g. "cifar10,tinystories") or press ENTER for all.
"""
import os

import torch
from torchvision import datasets as tv_datasets

import datasets as ds_mod

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.exists(os.path.join(HERE, "proposals.jsonl")) else os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")


def download_torchvision(name):
    info = ds_mod.DATASETS[name]
    if info["loader"] != "torchvision":
        print(f"  skip {name!r}: not a torchvision dataset (loader={info['loader']})")
        return
    print(f"Downloading {name!r} into {DATA_DIR} ...")
    for split in (True, False):
        ds = ds_mod.TORCHVISION[name](
            root=DATA_DIR, train=split, download=True,
            transform=ds_mod._image_transforms(info["shape"]))
        print(f"  {name} ({'train' if split else 'test'}): {len(ds)} items")


def download_tinystories(max_chars=500_000):
    """Pull a small slice of TinyStories from HuggingFace into data/tinystories.txt."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  skip tinystories: the 'datasets' package is not installed "
              "(pip install datasets).")
        return
    out = os.path.join(DATA_DIR, "tinystories.txt")
    print(f"Downloading TinyStories (first {max_chars} chars) into {out} ...")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    text = ""
    for row in ds:
        text += row["text"] + "\n"
        if len(text) >= max_chars:
            break
    text = text[:max_chars]
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"  wrote {len(text)} chars to {out}")


def download(name):
    info = ds_mod.DATASETS.get(name, {})
    if info.get("loader") == "torchvision":
        download_torchvision(name)
    elif name == "tinystories":
        download_tinystories()
    else:
        print(f"  unknown dataset {name!r} — skipping")


def main():
    names = list(ds_mod.DATASETS.keys())
    print("Available datasets:", ", ".join(names))
    answer = input("Which to download (blank = all): ").strip()
    if answer:
        wanted = [a.strip() for a in answer.split(",") if a.strip()]
    else:
        wanted = names

    for name in wanted:
        if name not in ds_mod.DATASETS:
            print(f"  unknown dataset {name!r} — skipping")
            continue
        download(name)
    print("Done.")


if __name__ == "__main__":
    main()
