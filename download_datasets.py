#!/usr/bin/env python3
"""download_datasets.py — pre-download torchvision datasets into ./data.

Mirrors how MNIST was fetched: for each named torchvision dataset we ask
torchvision to download it (train+test splits) by constructing the loader and
pulling one batch. Datasets are stored under <repo>/data just like MNIST.

Usage:
    python3 download_datasets.py

The script prompts for which datasets to fetch (blank = all torchvision
datasets). Type a comma-separated list (e.g. "cifar10,cifar100") or press
ENTER for all.
"""
import os

import torch
from torchvision import datasets as tv_datasets

import datasets as ds_mod

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.exists(os.path.join(HERE, "proposals.jsonl")) else os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")


def download(name):
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


def main():
    torchvision_names = [n for n, i in ds_mod.DATASETS.items()
                         if i["loader"] == "torchvision"]
    print("Available torchvision datasets:", ", ".join(torchvision_names))
    answer = input("Which to download (blank = all): ").strip()
    if answer:
        wanted = [a.strip() for a in answer.split(",") if a.strip()]
    else:
        wanted = torchvision_names

    for name in wanted:
        if name not in ds_mod.DATASETS:
            print(f"  unknown dataset {name!r} — skipping")
            continue
        download(name)
    print("Done.")


if __name__ == "__main__":
    main()
