from dataclasses import dataclass
from typing import Tuple, Dict, Any
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.cnn import CNN
from data.processing_data import make_dataloaders
from utils.io import save_checkpoint
from utils.seed import set_seed

@dataclass
class TrainConfig:
    data_dir: str
    checkpoints_dir: str
    best_ckpt_name: str
    dataset_name: str
    mean:float
    std:float
    batch_size: int
    num_workers: int
    val_split: float
    download: bool
    pin_memory: bool
    in_channels: int
    num_classes: int
    p_dropout: float
    use_batchnorm: bool
    epochs: int
    lr: float
    weight_decay: float
    seed: int
    scheduler: Dict[str, Any]

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == y).float().mean().item())

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    model.train()
    running_loss, running_acc, n = 0.0, 0.0, 0
    for xb, yb in tqdm(loader, desc="Train", leave=False):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        bs = xb.size(0)
        running_loss += loss.item() * bs
        running_acc  += accuracy(logits, yb) * bs
        n += bs
    return running_loss / n, running_acc / n

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, phase: str = "Val") -> Tuple[float, float]:
    model.eval()
    running_loss, running_correct, n = 0.0, 0.0, 0
    for xb, yb in tqdm(loader, desc=phase, leave=False):
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        bs = xb.size(0)
        running_loss += loss.item() * bs
        running_correct += (logits.argmax(dim=1) == yb).float().sum().item()
        n += bs
    return running_loss / n, running_correct / n

def build_scheduler(optimizer: optim.Optimizer, cfg: Dict[str, Any]):
    name = (cfg.get("name") or "").lower()
    if name == "steplr":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(cfg.get("step_size", 2)),
            gamma=float(cfg.get("gamma", 0.5)),
        )
    return None

def run_training(cfg: TrainConfig) -> str:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, _ = make_dataloaders(
        dataset_name=cfg.dataset_name,
        data_dir=cfg.data_dir,
        mean=cfg.mean,
        std = cfg.std,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        val_split=cfg.val_split,
        download=cfg.download,
        pin_memory=cfg.pin_memory,
        device_type=device.type,
    )

    model = CNN(
        p_dropout=cfg.p_dropout,
        use_batchnorm=cfg.use_batchnorm,
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = build_scheduler(optimizer, cfg.scheduler or {})

    os.makedirs(cfg.checkpoints_dir, exist_ok=True)
    best_path = os.path.join(cfg.checkpoints_dir, cfg.best_ckpt_name)

    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, phase="Val")
        print(f"[Epoch {epoch}] train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f}")
        if scheduler is not None:
            scheduler.step()

        
    save_checkpoint(model, best_path)
    print(f"Model Saved to {best_path}")

    return best_path
