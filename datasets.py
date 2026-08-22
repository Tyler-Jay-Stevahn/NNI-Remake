import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets as tv_datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.exists(os.path.join(HERE, "proposals.jsonl")) else os.path.dirname(HERE)

recycling_data = os.path.join(ROOT, "data", "Recycling Dataset")

# torchvision dataset classes keyed by dataset name. Each is downloaded into
# ROOT/data on first use (download=True), exactly like MNIST.
TORCHVISION = {
    "mnist":     tv_datasets.MNIST,
    "cifar10":   tv_datasets.CIFAR10,
    "cifar100":  tv_datasets.CIFAR100,
    "food101":   tv_datasets.Food101,
    "oxfordpet": tv_datasets.OxfordIIITPet,
    "voc2012":   tv_datasets.VOCDetection,
}

# Constructor kwargs for torchvision classes that do not use train=True/False.
# Maps dataset name -> (kwargs for the train split, kwargs for the val split).
# Datasets absent from this table use the classic train=True / train=False form.
_TV_SPLIT_KWARGS = {
    "food101":   ({"split": "train"}, {"split": "test"}),
    "oxfordpet": ({"split": "trainval"}, {"split": "test"}),
    "voc2012":   ({"year": "2012", "image_set": "train"},
                  {"year": "2012", "image_set": "val"}),
}

# Full dataset table — the single source of truth (build_model.dataset_info
# falls back to this table for names missing from its legacy copy). `loader`
# is one of:
#   "torchvision"  auto-downloaded torchvision dataset (TORCHVISION registry)
#   "local"        class subfolders under ROOT/data
#   "text"         local char-level corpus (data/TinyStories or data/<name>.txt)
#   "hf-text"      HuggingFace Hub dataset via the `datasets` package
#   "rl"           gymnasium environment transitions (random-policy behaviour
#                  cloning; needs `pip install gymnasium`)
#   None           schema-only: shape/classes registered, no loader wired yet
DATASETS = {
    # --- torchvision image classification ---
    "mnist":            {"modality": "image",      "shape": (1, 28, 28),   "classes": 10,    "loader": "torchvision", "name": "MNIST"},
    "cifar10":          {"modality": "image",      "shape": (3, 32, 32),   "classes": 10,    "loader": "torchvision", "name": "CIFAR-10"},
    "cifar100":         {"modality": "image",      "shape": (3, 32, 32),   "classes": 100,   "loader": "torchvision", "name": "CIFAR-100"},
    "food101":          {"modality": "image",      "shape": (3, 224, 224), "classes": 101,   "loader": "torchvision", "name": "Food-101"},
    "oxfordpet":        {"modality": "image",      "shape": (3, 224, 224), "classes": 37,    "loader": "torchvision", "name": "Oxford-IIIT Pet"},
    "voc2012":          {"modality": "image",      "shape": (3, 224, 224), "classes": 20,    "loader": "torchvision", "name": "VOC 2012 (largest-box class)"},
    # --- local ---
    "Recycling-Data":   {"modality": "image",      "shape": (3, 128, 128), "classes": 11,    "loader": "local",       "name": "Recycling Dataset"},
    # --- local text (char-level next-token LM) ---
    "tinystories":      {"modality": "text",       "shape": (128,),        "classes": 0,     "loader": "text",        "name": "TinyStories", "seq_len": 128, "vocab": 256},
    # --- HuggingFace text (auto-download + cache; needs `pip install datasets`)
    "wikitext103":      {"modality": "text",       "shape": (256,),        "classes": 0,     "loader": "hf-text",     "name": "WikiText-103 (LM)", "seq_len": 256, "vocab": 2048},
    "cnndm":            {"modality": "text",       "shape": (256,),        "classes": 0,     "loader": "hf-text",     "name": "CNN/DailyMail (LM)", "seq_len": 256, "vocab": 2048},
    "wmt14":            {"modality": "text",       "shape": (128,),        "classes": 0,     "loader": "hf-text",     "name": "WMT14 en-de (LM)", "seq_len": 128, "vocab": 2048},
    "glue-sst2":        {"modality": "text",       "shape": (160,),        "classes": 2,     "loader": "hf-text",     "name": "GLUE SST-2 (sentiment)", "seq_len": 160, "vocab": 256},
    # SQuAD as answer-span-start classification: the label is the char index
    # (0..seq_len-1) at which the answer starts inside the context window.
    "squad":            {"modality": "text",       "shape": (256,),        "classes": 256,   "loader": "hf-text",     "name": "SQuAD v1.1 (span start)", "seq_len": 256, "vocab": 256},
    # --- reinforcement learning (gymnasium; needs `pip install gymnasium`) ---
    # modality "tabular" so build_model routes state vectors through the dense
    # stem; the loader yields (state, action-taken) behaviour-cloning batches.
    "cartpole":         {"modality": "tabular",    "shape": (4,),          "classes": 2,     "loader": "rl",          "name": "CartPole-v1 (RL)", "env": "CartPole-v1"},
    "mountaincar":      {"modality": "tabular",    "shape": (2,),          "classes": 3,     "loader": "rl",          "name": "MountainCar-v0 (RL)", "env": "MountainCar-v0"},
    "lunarlander":      {"modality": "tabular",    "shape": (8,),          "classes": 4,     "loader": "rl",          "name": "LunarLander (RL)", "env": ["LunarLander-v3", "LunarLander-v2"], "needs": "box2d"},
    # --- schema-only: registered for builders/proposals, no loader yet ---
    "imagenet1k":       {"modality": "image",         "shape": (3, 224, 224),   "classes": 1000, "loader": None, "name": "ImageNet-1k"},
    "inat":             {"modality": "image",         "shape": (3, 224, 224),   "classes": 10000,"loader": None, "name": "iNaturalist 2021"},
    "coco":             {"modality": "image-detect",  "shape": (3, 640, 640),   "classes": 80,   "loader": None, "name": "COCO 2017 detection"},
    "lvis":             {"modality": "image-detect",  "shape": (3, 640, 640),   "classes": 1203, "loader": None, "name": "LVIS v1"},
    "cityscapes":       {"modality": "image-segment", "shape": (3, 1024, 2048), "classes": 19,   "loader": None, "name": "Cityscapes"},
}


# ---------------------------------------------------------------------------
# VOC detection -> classification wrapper
# ---------------------------------------------------------------------------
# The harness trains (x, y=int label) models, so VOCDetection is projected to
# single-label classification: the image's label is the class of its largest
# ground-truth bounding box. Full detection heads remain future work.
_VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


class VOCDetectionClassifier(Dataset):
    """Wrap tv_datasets.VOCDetection as (image, largest-object class index)."""

    def __init__(self, root, year="2012", image_set="train",
                 download=True, transform=None):
        self._ds = tv_datasets.VOCDetection(
            root=root, year=year, image_set=image_set,
            download=download, transform=transform)

    def __len__(self):
        return len(self._ds)

    @staticmethod
    def _label(target):
        # XML quirk: with exactly one object torchvision returns a dict of
        # that object instead of a list of objects.
        objs = target["annotation"]["object"]
        if isinstance(objs, dict):
            objs = [objs]
        best_name, best_area = None, -1.0
        for obj in objs:
            bb = obj["bndbox"]
            w = max(0.0, float(bb["xmax"]) - float(bb["xmin"]))
            h = max(0.0, float(bb["ymax"]) - float(bb["ymin"]))
            if w * h > best_area:
                best_name, best_area = obj["name"], w * h
        return _VOC_CLASSES.index(best_name)

    @classmethod
    def _from_tv(cls, tv_ds):
        """Wrap an already-constructed tv_datasets.VOCDetection instance."""
        wrapper = cls.__new__(cls)
        wrapper._ds = tv_ds
        return wrapper

    def __getitem__(self, idx):
        img, target = self._ds[idx]
        return img, self._label(target)



def _image_transforms(shape, augment=False):
    """Return image transforms. If augment=True, use training augmentations.
    augment='light' -> RandomHorizontalFlip only (safe for symmetric objects)."""
    _, h, w = shape
    if augment == "cifar":
        # Canonical CIFAR recipe: pad-4 random crop + horizontal flip. Stronger
        # than 'light' (no crop) and free of the aspect/rotation distortions
        # that augment=True applies — those hurt natural-image classes.
        return transforms.Compose([
            transforms.RandomCrop((h, w), padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
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
# ---------------------------------------------------------------------------
# Mixup augmentation wrapper
# ---------------------------------------------------------------------------
class MixupDataset(Dataset):
    """Wraps a dataset to apply mixup augmentation on-the-fly."""
    def __init__(self, dataset, alpha=0.2, num_classes=11):
        self.dataset = dataset
        self.alpha = alpha
        self.num_classes = num_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x1, y1 = self.dataset[idx]
        idx2 = torch.randint(0, len(self.dataset), ()).item()
        x2, y2 = self.dataset[idx2]

        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0
        x = lam * x1 + (1 - lam) * x2

        # Soft labels for mixup
        y1_oh = F.one_hot(torch.tensor(y1), self.num_classes).float()
        y2_oh = F.one_hot(torch.tensor(y2), self.num_classes).float()
        y = lam * y1_oh + (1 - lam) * y2_oh

        return x, y

def get_dataloader(name, batch_size=32, max_chars=None, augment=False, augment_kwargs=None):
    """Return (train_loader, val_loader) for the named dataset.

    torchvision datasets (TORCHVISION registry) are downloaded into ROOT/data
    on first use. The local image dataset (Recycling-Data) is loaded from
    class subfolders under ROOT/data. Text datasets are tokenised from a
    corpus (tinystories: local file; hf-text names: HuggingFace Hub download,
    cached after first use) — max_chars caps how much of the corpus is
    tokenised, used by the compile smoke-test. RL datasets yield random-policy
    (state, action) transitions from their gymnasium environment.
    """
    info = DATASETS[name]
    augment_kwargs = augment_kwargs or {}

    if not info.get("loader"):
        raise ValueError(
            f"{name!r} is registered schema-only; no data loader is wired yet")

    if info["loader"] == "text":
        return get_text_dataloader(name, batch_size=batch_size, max_chars=max_chars)
    if info["loader"] == "hf-text":
        return get_hf_text_dataloader(name, batch_size=batch_size, max_chars=max_chars)
    if info["loader"] == "rl":
        return get_rl_dataloader(name, batch_size=batch_size)

    train_tfm = _image_transforms(info["shape"], augment=augment)
    val_tfm = _image_transforms(info["shape"], augment=False)

    num_classes = info.get("classes", 10)

    if info["loader"] == "torchvision":
        ds_cls = TORCHVISION[name]
        data_dir = os.path.join(ROOT, "data")
        split_kwargs = _TV_SPLIT_KWARGS.get(name)
        if split_kwargs:
            train_kwargs, val_kwargs = split_kwargs
            train_src = ds_cls(root=data_dir, download=True, transform=train_tfm, **train_kwargs)
            test_src = ds_cls(root=data_dir, download=True, transform=val_tfm, **val_kwargs)
        else:
            train_src = ds_cls(root=data_dir, train=True,
                               download=True, transform=train_tfm)
            test_src = ds_cls(root=data_dir, train=False,
                              download=True, transform=val_tfm)
        # VOCDetection yields XML target dicts; project to int class labels.
        if name == "voc2012":
            train_src = VOCDetectionClassifier._from_tv(train_src)
            test_src = VOCDetectionClassifier._from_tv(test_src)
        train_full, test_ds = train_src, test_src
        n_val = int(0.1 * len(train_full))
        train_ds, val_ds = random_split(train_full, [len(train_full) - n_val, n_val])
        if augment == "mixup":
            alpha = augment_kwargs.get("alpha", 0.2)
            train_ds = MixupDataset(train_ds, alpha=alpha, num_classes=num_classes)
        return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2),
                DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2))

    elif info["loader"] == "local":
        if name == "Recycling-Data":
            full = tv_datasets.ImageFolder(recycling_data, transform=val_tfm)
        else:
            raise ValueError(f"no local loader for dataset {name!r}")
        n_val = int(0.1 * len(full))
        train_ds, val_ds = random_split(full, [len(full) - n_val, n_val])
        train_ds.dataset.transform = train_tfm
        val_ds.dataset.transform = val_tfm
        if augment == "mixup":
            alpha = augment_kwargs.get("alpha", 0.2)
            train_ds = MixupDataset(train_ds, alpha=alpha, num_classes=num_classes)
        return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2),
                DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2))

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


# ---------------------------------------------------------------------------
# HuggingFace text datasets (wikitext103, glue-sst2, squad, cnndm, wmt14)
#
# Downloaded from the Hub on first use, cached under ~/.cache/huggingface
# afterwards. Requires the `datasets` package (pip install datasets), imported
# lazily so everything else works without it.
#
# NOTE: this repo's own datasets.py SHADOWS the pip package of the same name
# whenever Python runs from this directory, so a plain
# `from datasets import load_dataset` would import this very file.
# _import_hf_load_dataset() temporarily drops this repo dir from sys.path (and
# purges any self-shadowed sys.modules entry) to reach the real package.
# ---------------------------------------------------------------------------
_HF_SPECS = {
    # name -> (repo id, config, train split, val split)
    "wikitext103": ("Salesforce/wikitext", "wikitext-103-raw-v1", "train", "validation"),
    "glue-sst2":   ("nyu-mll/glue", "sst2", "train", "validation"),
    "squad":       ("rajpurkar/squad", None, "train", "validation"),
    "cnndm":       ("abisee/cnn_dailymail", "3.0.0", "train", "validation"),
    "wmt14":       ("wmt/wmt14", "de-en", "train", "validation"),
}

# Per-example corpus formatters for the LM-style HF datasets. Until dedicated
# summarisation/translation heads exist, cnndm and wmt14 train as char-level
# next-token LM corpora over formatted documents/pairs (same (x, y=x shifted)
# contract as tinystories).
_HF_LM_FORMATTERS = {
    "wikitext103": lambda ex: ex["text"],
    "cnndm":       lambda ex: f"{ex['article']}\n= {ex['highlights']}\n\n",
    "wmt14":       lambda ex: "{en}\n||| {de}\n".format(**ex["translation"]),
}
def _import_hf_load_dataset():
    """Return load_dataset from the real pip `datasets` package.

    The local datasets.py shadows the pip package when running from this
    repo, so the import temporarily drops this dir from sys.path and purges
    the self-shadowing sys.modules entry. The HF package then stays bound to
    the name "datasets" for the rest of the process — restoring the shadow
    between loads corrupts dill's module hashing (RLock pickling errors on
    every load after the first). Code that needs this repo's registry reads
    the "nni_datasets" alias registered at the bottom of this module (see
    build_model.dataset_info).
    """
    import importlib
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    me = os.path.abspath(__file__)
    shadow = sys.modules.get("datasets")
    is_self = (shadow is not None
               and os.path.abspath(getattr(shadow, "__file__", "") or "") == me)
    saved_path = sys.path[:]
    if is_self:
        del sys.modules["datasets"]
    sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != here]
    try:
        mod = importlib.import_module("datasets")
    except ImportError as exc:
        sys.path[:] = saved_path
        if is_self:
            sys.modules["datasets"] = shadow
        raise ImportError(
            "HF text datasets need the HuggingFace `datasets` package: "
            "pip install datasets") from exc
    return mod.load_dataset


def _hf_load(name):
    """Load the full DatasetDict for an HF dataset in ONE load_dataset call."""
    repo, config, _, _ = _HF_SPECS[name]
    load_dataset = _import_hf_load_dataset()
    return load_dataset(*([repo, config] if config else [repo]))


def _hf_lm_texts(name, max_chars):
    """Return (train_text, val_text) for an LM-style HF dataset.

    Iterates each split, formatting examples into one corpus string. With
    max_chars set, iteration stops early once enough characters are gathered
    (compile smoke-test path).
    """
    fmt = _HF_LM_FORMATTERS[name]
    _, _, train_split, val_split = _HF_SPECS[name]
    texts = []
    dd = _hf_load(name)
    for split in (train_split, val_split):
        parts, total = [], 0
        for ex in dd[split]:
            chunk = fmt(ex)
            parts.append(chunk)
            total += len(chunk)
            if max_chars is not None and total >= max_chars:
                break
        text = "".join(parts)
        texts.append(text[:max_chars] if max_chars is not None else text)
    return texts[0], texts[1]


def _encode_fixed(text, vocab, seq_len):
    """Char-encode text to exactly seq_len ids: truncate, pad with 0."""
    unk = vocab["<unk>"]
    ids = [vocab.get(c, unk) for c in text[:seq_len]]
    return ids + [0] * (seq_len - len(ids))


class _PairedTensors(torch.utils.data.Dataset):
    """Prebuilt (x, y) tensors — char-id windows or RL state/action pairs."""


    def __init__(self, x, y):
        self.x = x if torch.is_tensor(x) else torch.tensor(x, dtype=torch.long)
        self.y = y if torch.is_tensor(y) else torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def _squad_windows(split_ds, vocab, seq_len, limit=None):
    """Extract (question-prefixed context window, answer-start offset) pairs.

    The label is the character offset at which the answer begins inside the
    encoded window (classes == seq_len in the DATASETS table). Examples whose
    gold answer does not occur verbatim in the context, or that do not fit a
    single window, are skipped.
    """
    xs, ys = [], []
    for ex in split_ds:
        answers = ex["answers"]["text"]
        if not answers:
            continue
        ctx = ex["context"]
        answer = answers[0]
        start = ctx.find(answer)
        if start < 0:
            continue
        # Window opens a quarter-seq before the answer, clamped to the context.
        w_start = max(0, min(start - seq_len // 4, len(ctx) - seq_len))
        window = ctx[w_start:w_start + seq_len]
        if len(window) < seq_len:
            continue
        prefix = ex["question"] + " "
        label = len(prefix) + (start - w_start)
        if label >= seq_len:
            continue  # a long question pushes the answer past the window
        xs.append(_encode_fixed(prefix + window, vocab, seq_len))
        ys.append(label)
        if limit is not None and len(xs) >= limit:
            break
    return xs, ys
def _check_vocab(name, vocab, info):
    """Fail loudly if the corpus's char set exceeds the declared embedding
    vocabulary — an overflow would be an out-of-bounds Embedding index."""
    if len(vocab) > info["vocab"]:
        raise ValueError(
            f"{name}: char vocab grew to {len(vocab)} entries but the schema "
            f"declares vocab={info['vocab']}; raise the entry's 'vocab' field")
    return vocab


def get_hf_text_dataloader(name, batch_size=32, max_chars=None):
    """Return (train_loader, val_loader) for an HF-backed text dataset.

    wikitext103 / cnndm / wmt14: char-level next-token LM corpora.
    glue-sst2: fixed-length char-id windows labelled with sentiment (2-way).
    squad: answer-span-start classification (label in [0, seq_len)).

    max_chars caps the materialised corpus/examples per split (compile
    smoke-test path only; real training passes None for the full dataset).
    Note the first call downloads and caches the full split either way.
    """
    info = DATASETS[name]
    seq_len = info["seq_len"]
    cap = max(2, max_chars // seq_len) if max_chars is not None else None

    if name == "glue-sst2":
        splits = _hf_load(name)
        train_raw, val_raw = splits["train"], splits["validation"]
        if cap is not None:
            train_raw = train_raw.select(range(min(cap, len(train_raw))))
            val_raw = val_raw.select(range(min(cap, len(val_raw))))
        vocab = _check_vocab(name, _build_vocab("".join(ex["sentence"] for ex in train_raw)), info)
        tx = [_encode_fixed(ex["sentence"], vocab, seq_len) for ex in train_raw]
        ty = [int(ex["label"]) for ex in train_raw]
        vx = [_encode_fixed(ex["sentence"], vocab, seq_len) for ex in val_raw]
        vy = [int(ex["label"]) for ex in val_raw]
        if not tx or not vx:
            raise ValueError(f"{name}: no examples extracted")
        return (DataLoader(_PairedTensors(tx, ty), batch_size=batch_size, shuffle=True),
                DataLoader(_PairedTensors(vx, vy), batch_size=batch_size, shuffle=False))

    if name == "squad":
        splits = _hf_load(name)
        train_raw, val_raw = splits["train"], splits["validation"]
        # Char vocab over a sample of train questions+contexts.
        sample = []
        for i, ex in enumerate(train_raw):
            if i >= 4000:
                break
            sample.append(ex["question"])
            sample.append(ex["context"])
        vocab = _check_vocab(name, _build_vocab("\n".join(sample)), info)
        tx, ty = _squad_windows(train_raw, vocab, seq_len, limit=cap)
        vx, vy = _squad_windows(val_raw, vocab, seq_len, limit=max(cap // 2, 2) if cap else None)
        if not tx or not vx:
            raise ValueError(f"{name}: no span windows extracted")
        return (DataLoader(_PairedTensors(tx, ty), batch_size=batch_size, shuffle=True),
                DataLoader(_PairedTensors(vx, vy), batch_size=batch_size, shuffle=False))

    # wikitext103 / cnndm / wmt14 — LM corpora, same contract as tinystories.
    train_text, val_text = _hf_lm_texts(name, max_chars)
    vocab = _check_vocab(name, _build_vocab(train_text + "\n" + val_text), info)
    train_pairs = _tokenize(train_text, vocab, seq_len)
    val_pairs = (_tokenize(val_text, vocab, seq_len) if val_text
                 else train_pairs[-max(1, int(0.1 * len(train_pairs))):])
    if not train_pairs:
        raise ValueError(f"{name}: corpus too small to form seq_len={seq_len} windows")
    return (DataLoader(_TextDataset(train_pairs), batch_size=batch_size, shuffle=True),
            DataLoader(_TextDataset(val_pairs), batch_size=batch_size, shuffle=False))


# ---------------------------------------------------------------------------
# Reinforcement learning datasets (gymnasium)
#
# The harness trains supervised (x, y) models, so gymnasium environments are
# exposed as behaviour-cloning data: random-policy transitions
# (state vector -> action taken). That exercises builders, losses and the
# compile gate end-to-end; an online RL training loop remains future work.
# cartpole / mountaincar are dependency-free; lunarlander additionally needs
# box2d (pip install gymnasium[box2d]) and raises a descriptive error without.
# ---------------------------------------------------------------------------


def _import_gymnasium():
    try:
        import gymnasium
        return gymnasium
    except ImportError as exc:
        raise ImportError(
            "RL datasets need gymnasium: pip install gymnasium") from exc


def _make_gym_env(gym, name):
    """Create the first creatable env id registered for this dataset."""
    env_spec = DATASETS[name]["env"]
    ids = [env_spec] if isinstance(env_spec, str) else list(env_spec)
    errors = []
    for env_id in ids:
        try:
            return gym.make(env_id)
        except Exception as exc:  # version renames, missing optional extras
            errors.append(f"{env_id}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        f"could not create any gymnasium env for {name!r} (tried {ids}); "
        "install missing extras (e.g. pip install gymnasium[box2d]); "
        "errors: " + " | ".join(errors))


def get_rl_dataloader(name, batch_size=32, n_samples=20000, seed=0):
    """Return (train_loader, val_loader) of random-policy transitions.

    Each sample is (state_vector, action_taken) with a 90/10 split. Actions
    come from a seeded uniform-random policy, so labels are drawn from the
    environment's action space rather than an expert.
    """
    gym = _import_gymnasium()
    env = _make_gym_env(gym, name)
    states, actions = [], []
    obs, _ = env.reset(seed=seed)
    while len(states) < n_samples:
        action = env.action_space.sample()
        obs_next, _reward, terminated, truncated, _info = env.step(action)
        states.append(np.asarray(obs, dtype=np.float32))
        actions.append(int(action))
        obs = obs_next
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()

    x = torch.from_numpy(np.stack(states))
    y = torch.tensor(actions, dtype=torch.long)
    n_val = max(1, int(0.1 * len(x)))
    return (DataLoader(_PairedTensors(x[:-n_val], y[:-n_val]), batch_size=batch_size, shuffle=True),
            DataLoader(_PairedTensors(x[-n_val:], y[-n_val:]), batch_size=batch_size, shuffle=False))


# Alias so the registry stays reachable after the HF package takes over the
# "datasets" name for the process (see _import_hf_load_dataset).
sys.modules.setdefault("nni_datasets", sys.modules[__name__])
