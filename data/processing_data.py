from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader


def get_dataset_class(name: str):

    if not hasattr(datasets, name):
        raise ValueError(f"Dataset = {name} non trovato in torchvision.datasets")

    return getattr(datasets,name)#la libreria datasets prenderà l'oggetto e il nome e restituirà MNIST

def get_transform(mean: float, std: float) -> transforms.Compose:
    
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])

    return tf



def make_dataloader(
    data_dir: str,
    dataset_name: str,
    mean: float,
    std: float,
    download: bool,
    val_split: float,
    pin_memory: bool,
    device: str,
    batch_size: int,
    num_workers: int
):

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    DatasetClass = get_dataset_class(dataset_name)
    tf = get_transform(mean,std)

    try:
        train_full = DatasetClass(root=str(data_path),train=True,transform=tf,download=download)
        test_ds = DatasetClass(root=str(data_path),train=False,transform=tf,download=download)
    except Exception as e:
        print("Impossibile scaricare",str(e))

    val_size = int(len(train_full)* val_split)
    train_size = len(train_full) - val_size
    train_ds, val_ds = random_split(train_full,[train_size,val_size])

    pmem = pin_memory and (device.type == "cuda")
    train_loader = DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=pmem)
    val_loader = DataLoader(val_ds,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=pmem)
    test_loader = DataLoader(test_ds,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=pmem)
    
    return train_loader, val_loader, test_loader


        
    