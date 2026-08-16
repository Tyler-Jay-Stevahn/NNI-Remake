import os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.exists(os.path.join(HERE, "proposals.jsonl")) else os.path.dirname(HERE)


recycling_data = os.path.join(ROOT, "data", "Recycling Dataset")

DATASETS = {
    # --- existing (unchanged behaviour) ---
    "mnist":            {"modality": "image", "shape": (1, 28, 28), "classes": 10, "loader": "torchvision", "name": "MNIST"},
    "Recycling-Data":  {"modality": "image", "shape": (3, 128, 128), "classes": 11, "loader": "local", "name": "Recycling Dataset"}
    }

def get_dataloader(name, batch_size = 32):
    info = DATASETS[name]
    if info["loader"]=="torchvision":
        pass
    elif info["loader"]=="local":
        if name == "Recycling-Data":
            return recycling_data
        else:
            pass