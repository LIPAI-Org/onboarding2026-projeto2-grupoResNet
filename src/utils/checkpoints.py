# checkpoints.py
from pathlib import Path
import torch
from configs.configs_base import DEVICE

def save_checkpoint(path, modelo, otim, epoca, best_val_acc, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoca": epoca,
        "best_val_acc": best_val_acc,
        "modelo_state_dict": modelo.state_dict(),
        "otim_state_dict": otim.state_dict(),
    }

    if extra is not None:
        checkpoint["extra"] = extra

    torch.save(checkpoint, path)


def load_checkpoint(path, modelo, otim=None):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint["modelo_state_dict"]

    # Remove chaves extras de contagem de ops/params
    state_dict = {
        k: v for k, v in state_dict.items()
        if "total_ops" not in k and "total_params" not in k
    }

    modelo.load_state_dict(state_dict, strict=True)
    
    if otim is not None and "otim_state_dict" in checkpoint:
        otim.load_state_dict(checkpoint["otim_state_dict"])

    return checkpoint
