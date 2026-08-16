import os

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets as tv_datasets, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.exists(os.path.join(HERE, "proposals.jsonl")) else os.path.dirname(HERE)


recycling_data = os.path.join(ROOT, "data", "Recycling Dataset")

DATASETS = {
    # --- existing (unchanged behaviour) ---
    "mnist":            {"modality": "image", "shape": (1, 28, 28), "classes": 10, "loader": "torchvision", "name": "MNIST"},
    "Recycling-Data":  {"modality": "image", "shape": (3, 128, 128), "classes": 11, "loader": "local", "name": "Recycling Dataset"}
    }


def _image_transforms(shape):
    # shape is (C, H, W); ImageFolder gives (H, W) target size.
    _, h, w = shape
    return transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
    ])


def get_dataloader(name, batch_size=32):
    """Return (train_loader, val_loader) for the named dataset.

    Local image datasets are loaded from class subfolders via ImageFolder.
    torchvision datasets are downloaded on first use.
    """
    info = DATASETS[name]
    tfm = _image_transforms(info["shape"])

    if info["loader"] == "torchvision":
        train_full = tv_datasets.MNIST(
            root=os.path.join(ROOT, "data"), train=True,
            download=True, transform=tfm)
        test_ds = tv_datasets.MNIST(
            root=os.path.join(ROOT, "data"), train=False,
            download=True, transform=tfm)
        n_val = int(0.1 * len(train_full))
        train_ds, val_ds = random_split(train_full, [len(train_full) - n_val, n_val])
        return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
                DataLoader(val_ds, batch_size=batch_size, shuffle=False))

    elif info["loader"] == "local":
        if name == "Recycling-Data":
            full = tv_datasets.ImageFolder(recycling_data, transform=tfm)
            n_val = int(0.1 * len(full))
            train_ds, val_ds = random_split(full, [len(full) - n_val, n_val])
            return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
                    DataLoader(val_ds, batch_size=batch_size, shuffle=False))
        else:
            raise ValueError(f"no local loader for dataset {name!r}")

    raise ValueError(f"unknown loader type {info['loader']!r} for {name!r}")