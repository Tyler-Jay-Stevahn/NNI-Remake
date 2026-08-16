import os

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets as tv_datasets, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.exists(os.path.join(HERE, "proposals.jsonl")) else os.path.dirname(HERE)

recycling_data = os.path.join(ROOT, "data", "Recycling Dataset")
imagenet_subset_data = os.path.join(ROOT, "data", "ImageNet Subset")
atari_ale_data = os.path.join(ROOT, "data", "Atari ALE")

# torchvision dataset classes keyed by dataset name. Each is downloaded into
# ROOT/data on first use (download=True), exactly like MNIST.
TORCHVISION = {
    "mnist":   tv_datasets.MNIST,
    "cifar10": tv_datasets.CIFAR10,
    "cifar100": tv_datasets.CIFAR100,
    "mnist-cluster": tv_datasets.MNIST,
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


class _StackedFrameFolder(torch.utils.data.Dataset):
    """Load pre-stacked (C,H,W) tensors from class subfolders.

    Used for Atari ALE where each sample is a stack of 4 frames of shape
    (4, 84, 84) rather than a single 3-channel PIL image. Expects layout:

        <root>/<class_name>/<file>.npy      (.npy with array shape (4,84,84))
        <root>/<class_name>/<file>.pt       (torch.save of a (4,84,84) tensor)

    Returns (tensor, class_index). No on-the-fly stacking — the stacks are
    assumed already prepared on disk.
    """
    def __init__(self, root, shape):
        self.root = root
        self.shape = shape
        self.samples = []  # (path, class_idx)
        self.classes = sorted(
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        for cls in self.classes:
            cls_dir = os.path.join(root, cls)
            for fn in os.listdir(cls_dir):
                if fn.lower().endswith((".npy", ".pt")):
                    self.samples.append(
                        (os.path.join(cls_dir, fn), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        if path.lower().endswith(".npy"):
            import numpy as np
            arr = np.load(path)
            tensor = torch.as_tensor(arr, dtype=torch.float32)
        else:
            tensor = torch.load(path, weights_only=True)
        # Defensive: if stored as (H,W,C), move channel to front.
        if tensor.dim() == 3 and tensor.shape[0] != self.shape[0]:
            tensor = tensor.permute(2, 0, 1)
        return tensor, label


def get_dataloader(name, batch_size=32):
    """Return (train_loader, val_loader) for the named dataset.

    torchvision datasets (mnist, cifar10, cifar100, mnist-cluster) are
    downloaded into ROOT/data on first use. Local image datasets
    (Recycling-Data, imagenet-subset, atari-ale) are loaded from class
    subfolders under ROOT/data.
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
        elif name == "imagenet-subset":
            full = tv_datasets.ImageFolder(imagenet_subset_data, transform=tfm)
        elif name == "atari-ale":
            full = _StackedFrameFolder(atari_ale_data, info["shape"])
        else:
            raise ValueError(f"no local loader for dataset {name!r}")
        n_val = int(0.1 * len(full))
        train_ds, val_ds = random_split(full, [len(full) - n_val, n_val])
        return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
                DataLoader(val_ds, batch_size=batch_size, shuffle=False))

    raise ValueError(f"unknown loader type {info['loader']!r} for {name!r}")
