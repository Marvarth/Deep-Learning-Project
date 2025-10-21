from pathlib import Path
import torch
from torch import nn

def save_checkpoint(model: nn.Module, ckpt_path: str) -> None:
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)

def load_checkpoint(model: nn.Module, ckpt_path: str, map_location: str | None = None) -> None:
    state = torch.load(ckpt_path, map_location=map_location or "cpu")
    model.load_state_dict(state)
