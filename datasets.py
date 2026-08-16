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
    "imagenet-subset":  {"modality": "image",      "shape": (3, 16, 16),   "classes": 10,  "loader": "local",       "name": "ImageNet Subset"},
    "Recycling-Data":   {"modality": "image",      "shape": (3, 128, 128), "classes": 11,  "loader": "local",       "name": "Recycling Dataset"},
    "openml-ctr23":     {"modality": "tabular",    "shape": (20,),         "classes": 2,   "loader": "tabular",     "name": "OpenML CTR23"},
    "speech-commands":  {"modality": "audio-mel",  "shape": (1, 64, 64),   "classes": 12,  "loader": "audio-mel",   "name": "Speech Commands"},
    "esc50":            {"modality": "audio-wave", "shape": (1, 8000),     "classes": 10,  "loader": "audio-wave", "name": "ESC-50"},
    "ag-news":          {"modality": "text",       "shape": (64,),         "classes": 4,   "loader": "text",       "name": "AG News",    "vocab": 20000},
    "imdb":             {"modality": "text",       "shape": (128,),        "classes": 2,   "loader": "text",       "name": "IMDB",       "vocab": 20000},
    "uci-har":          {"modality": "timeseries", "shape": (9, 128),      "classes": 6,   "loader": "timeseries", "name": "UCI HAR"},
    "mnist-cluster":    {"modality": "image",      "shape": (1, 28, 28),   "classes": 10,  "loader": "torchvision", "name": "MNIST Cluster"},
    "openml-cluster":   {"modality": "tabular",    "shape": (20,),         "classes": 8,   "loader": "tabular",     "name": "OpenML Cluster"},
    "atari-ale":        {"modality": "image",      "shape": (4, 84, 84),   "classes": 18,  "loader": "local",       "name": "Atari ALE"},
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

    torchvision datasets (mnist, cifar10, cifar100, mnist-cluster) are
    downloaded into ROOT/data on first use. Local image datasets
    (Recycling-Data, imagenet-subset, atari-ale) are loaded from class
    subfolders via ImageFolder.
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
            n_val = int(0.1 * len(full))
            train_ds, val_ds = random_split(full, [len(full) - n_val, n_val])
            return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
                    DataLoader(val_ds, batch_size=batch_size, shuffle=False))
        else:
            raise ValueError(f"no local loader for dataset {name!r}")

    raise ValueError(f"unknown loader type {info['loader']!r} for {name!r}")
