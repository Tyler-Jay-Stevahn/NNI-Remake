import os

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets as tv_datasets, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.exists(os.path.join(HERE, "proposals.jsonl")) else os.path.dirname(HERE)

recycling_data = os.path.join(ROOT, "data", "Recycling Dataset")

# torchvision dataset classes keyed by dataset name. Each is downloaded into
# ROOT/data on first use (download=True), exactly like MNIST.
TORCHVISION = {
    "mnist":   tv_datasets.MNIST,
    "cifar10": tv_datasets.CIFAR10,
    "cifar100": tv_datasets.CIFAR100,
}

# Full dataset table, carried over from build_model.DATASETS, with the loader
# and display-name fields filled in so datasets.py is the single source of
# truth. `loader` is "torchvision" (auto-downloaded), "local" (class subfolders
# under ROOT/data), or the modality label for datasets not yet wired to a
# loader (e.g. audio/text/tabular) — those remain schema-only for now.
DATASETS = {
    "mnist":            {"modality": "image",      "shape": (1, 28, 28),   "classes": 10,  "loader": "torchvision", "name": "MNIST"},
    "cifar10":          {"modality": "image",      "shape": (3, 32, 32),   "classes": 10,  "loader": "torchvision", "name": "CIFAR-10"},
    "cifar100":         {"modality": "image",      "shape": (3, 32, 32),   "classes": 100, "loader": "torchvision", "name": "CIFAR-100"},
    "Recycling-Data":   {"modality": "image",      "shape": (3, 128, 128), "classes": 11,  "loader": "local",       "name": "Recycling Dataset"},
}


def _image_transforms(shape):
    # shape is (C, H, W); ImageFolder / torchvision give (H, W) target size.
    _, h, w = shape
    return transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
    ])


def get_dataloader(name, batch_size=32):
    """Return (train_loader, val_loader) for the named dataset.

    torchvision datasets (mnist, cifar10, cifar100) are downloaded into
    ROOT/data on first use. The local image dataset (Recycling-Data) is
    loaded from class subfolders under ROOT/data.
    """
    info = DATASETS[name]
    tfm = _image_transforms(info["shape"])

    if info["loader"] == "torchvision":
        ds_cls = TORCHVISION[name]
        data_dir = os.path.join(ROOT, "data")
        train_full = ds_cls(root=data_dir, train=True,
                            download=True, transform=tfm)
        test_ds = ds_cls(root=data_dir, train=False,
                         download=True, transform=tfm)
        n_val = int(0.1 * len(train_full))
        train_ds, val_ds = random_split(train_full, [len(train_full) - n_val, n_val])
        return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
                DataLoader(val_ds, batch_size=batch_size, shuffle=False))

    elif info["loader"] == "local":
        if name == "Recycling-Data":
            full = tv_datasets.ImageFolder(recycling_data, transform=tfm)
        else:
            raise ValueError(f"no local loader for dataset {name!r}")
        n_val = int(0.1 * len(full))
        train_ds, val_ds = random_split(full, [len(full) - n_val, n_val])
        return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
                DataLoader(val_ds, batch_size=batch_size, shuffle=False))

    raise ValueError(f"unknown loader type {info['loader']!r} for {name!r}")
