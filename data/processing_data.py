from typing import Tuple
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch

def get_dataset_class(name: str):
    """
    Restituisce dinamicamente la classe dataset da torchvision.datasets.<name>.
    Esempi validi: MNIST, FashionMNIST, CIFAR10, CIFAR100, EMNIST, STL10, ecc.
    """
    if not hasattr(datasets, name):
        raise ValueError(f"Dataset '{name}' non trovato in torchvision.datasets")
    return getattr(datasets, name)

def get_default_transforms(mean:float,std:float) -> transforms.Compose:
    """
    Ritorna trasformazioni base per immagini grayscale o RGB,
    applicando normalizzazione generica se non specificata.
    """
    
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    return tf

def make_dataloaders(
    dataset_name: str ,
    data_dir: str,
    mean:float,
    std:float,
    batch_size: int,
    num_workers: int,
    val_split: float,
    download: bool,
    pin_memory: bool,
    device_type: str,
):
    """
    Crea DataLoader train/val/test per qualsiasi dataset torchvision.
    Basta specificare il nome nel config YAML: data.dataset: "CIFAR10"
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    DatasetClass = get_dataset_class(dataset_name)
    tf = get_default_transforms(mean,std)

    # Alcuni dataset (es. STL10) usano argomenti diversi
    # Cerchiamo di gestire in modo flessibile train/test
    try:
        full_train = DatasetClass(root=str(data_path), train=True, transform=tf, download=download)
        test_ds    = DatasetClass(root=str(data_path), train=False, transform=tf, download=download)
    except Exception as e:
        print("⚠️ Impossibile scaricare . Errore:", str(e))

    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(full_train, [train_size, val_size])

    pmem = pin_memory and (device_type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pmem)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pmem)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pmem)
    return train_loader, val_loader, test_loader
