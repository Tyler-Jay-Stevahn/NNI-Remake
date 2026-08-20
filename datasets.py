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
    "tinystories":      {"modality": "text",       "shape": (128,),        "classes": 0,   "loader": "text",        "name": "TinyStories", "seq_len": 128, "vocab": 256},
}


def _image_transforms(shape, augment=False):
    """Return image transforms. If augment=True, use training augmentations.
    augment='light' -> RandomHorizontalFlip only (safe for symmetric objects)."""
    _, h, w = shape
    if augment == "light":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((h, w)),
            transforms.ToTensor(),
        ])
    if augment is True:
        return transforms.Compose([
            transforms.RandomResizedCrop((h, w), scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
    ])

def get_dataloader(name, batch_size=32, max_chars=None, augment=False):
    """Return (train_loader, val_loader) for the named dataset.

    torchvision datasets (mnist, cifar10, cifar100) are downloaded into
    ROOT/data on first use. The local image dataset (Recycling-Data) is
    loaded from class subfolders under ROOT/data. Text datasets (tinystories)
    are tokenised from a corpus via get_text_dataloader (max_chars caps how
    much of the corpus is tokenised — used by the compile smoke-test).
    """
    info = DATASETS[name]

    if info["loader"] == "text":
        return get_text_dataloader(name, batch_size=batch_size, max_chars=max_chars)

    train_tfm = _image_transforms(info["shape"], augment=augment)
    val_tfm = _image_transforms(info["shape"], augment=False)

    if info["loader"] == "torchvision":
        ds_cls = TORCHVISION[name]
        data_dir = os.path.join(ROOT, "data")
        train_full = ds_cls(root=data_dir, train=True,
                            download=True, transform=train_tfm)
        test_ds = ds_cls(root=data_dir, train=False,
                         download=True, transform=val_tfm)
        n_val = int(0.1 * len(train_full))
        train_ds, val_ds = random_split(train_full, [len(train_full) - n_val, n_val])
        return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
                DataLoader(val_ds, batch_size=batch_size, shuffle=False))

    elif info["loader"] == "local":
        if name == "Recycling-Data":
            full = tv_datasets.ImageFolder(recycling_data, transform=val_tfm)
        else:
            raise ValueError(f"no local loader for dataset {name!r}")
        n_val = int(0.1 * len(full))
        train_ds, val_ds = random_split(full, [len(full) - n_val, n_val])
        train_ds.dataset.transform = train_tfm
        val_ds.dataset.transform = val_tfm
        return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
                DataLoader(val_ds, batch_size=batch_size, shuffle=False))

    elif info["loader"] == "text":
        return get_text_dataloader(name, batch_size=batch_size)

    raise ValueError(f"unknown loader type {info['loader']!r} for {name!r}")


# ---------------------------------------------------------------------------
# Text dataset: TinyStories (char-level, next-token targets)
#
# Expected on-disk layout (already downloaded by the user):
#   data/TinyStories/TinyStories-train.txt
#   data/TinyStories/TinyStories-valid.txt
# ---------------------------------------------------------------------------
def _tinystories_paths():
    """Return (train_path, val_path) for the TinyStories corpus."""
    base = os.path.join(ROOT, "data", "TinyStories")
    train = os.path.join(base, "TinyStories-train.txt")
    val = os.path.join(base, "TinyStories-valid.txt")
    return train, val


def _load_text_file(path):
    with open(path, encoding="utf-8", errors="ignore") as fh:
        return fh.read()


def _build_vocab(text):
    # Char-level vocab over the corpus, plus <unk>/<pad>.
    chars = sorted(set(text))
    vocab = {"<pad>": 0, "<unk>": 1}
    for i, c in enumerate(chars):
        vocab[c] = i + 2
    return vocab


def _tokenize(text, vocab, seq_len):
    """Chunk the corpus into (input, target) pairs for next-token LM training.

    Each sample is a fixed-length window; target is the window shifted by one
    (predict the next character). The final window drops the last char so
    input/target lengths match.
    """
    unk = vocab["<unk>"]
    ids = [vocab.get(c, unk) for c in text]
    samples = []
    for i in range(0, len(ids) - seq_len, seq_len):
        window = ids[i:i + seq_len + 1]
        if len(window) < seq_len + 1:
            continue
        x = window[:-1]
        y = window[1:]
        samples.append((x, y))
    return samples


class _TextDataset(torch.utils.data.Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x, y = self.pairs[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def get_text_dataloader(name, batch_size=32, max_chars=None):
    """Return (train_loader, val_loader) for a text dataset (next-token LM).

    For tinystories, reads data/TinyStories/TinyStories-train.txt (train) and
    TinyStories-valid.txt (val), builds a char-level vocab over the combined
    corpus, and yields (input_ids, target_ids) where target is the input
    shifted by one. The builder's LM head consumes (x, y) directly.

    `max_chars` (optional) caps how much of the corpus is tokenised. This is
    used by the compile smoke-test (compile_test.two_samples), which only needs
    a couple of batches; without it the loader materialises the entire corpus
    into RAM (tens of GB for TinyStories) and takes minutes, for a 2-sample
    check. Real training passes max_chars=None to use the full corpus.
    """
    info = DATASETS[name]
    seq_len = info.get("seq_len", 128)

    if name == "tinystories":
        train_path, val_path = _tinystories_paths()
        if not (os.path.exists(train_path) and os.path.getsize(train_path) > 0):
            raise FileNotFoundError(
                f"no TinyStories corpus at {train_path}; expected the dataset "
                f"under data/TinyStories/ (TinyStories-train.txt / -valid.txt)")
        train_text = _load_text_file(train_path)
        if max_chars is not None:
            train_text = train_text[:max_chars]
        val_text = _load_text_file(val_path) if os.path.exists(val_path) else ""
        if max_chars is not None:
            val_text = val_text[:max_chars]
        vocab = _build_vocab(train_text + "\n" + val_text)
        train_pairs = _tokenize(train_text, vocab, seq_len)
        val_pairs = _tokenize(val_text, vocab, seq_len) if val_text else train_pairs[-max(1, int(0.1 * len(train_pairs))):]
    else:
        corpus_path = os.path.join(ROOT, "data", f"{name}.txt")
        if not (os.path.exists(corpus_path) and os.path.getsize(corpus_path) > 0):
            raise FileNotFoundError(f"no corpus found at {corpus_path}")
        text = _load_text_file(corpus_path)
        vocab = _build_vocab(text)
        pairs = _tokenize(text, vocab, seq_len)
        if not pairs:
            raise ValueError(f"corpus too small to form seq_len={seq_len} windows")
        n_val = max(1, int(0.1 * len(pairs)))
        train_pairs, val_pairs = pairs[:-n_val], pairs[-n_val:]

    if not train_pairs:
        raise ValueError(f"corpus too small to form seq_len={seq_len} windows")

    return (DataLoader(_TextDataset(train_pairs), batch_size=batch_size, shuffle=True),
            DataLoader(_TextDataset(val_pairs), batch_size=batch_size, shuffle=False))
