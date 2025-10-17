from dataclasses import dataclass
import os
from typing import Dict, Any
from torch import nn, optim
import torch
from data.processing_data import make_dataloader
from models.cnn import CNN
from utils.seed import set_seed




@dataclass #classe che dichiara i valori che andremo ad utilizzare
class TrainConfig:
    data_dir: str
    checkpoint_dir: str
    best_ckpt_name: str
    dataset_name: str
    mean: float
    std: float
    batch_size: int
    num_workes: int
    val_split: float
    download: bool
    pin_memory: bool
    in_channels: int
    num_classes: int
    dropout: float
    use_batchnorm: bool
    epochs: int
    lr: float
    weight_decay: float
    seed: int
    scheduler: Dict[str,Any]

#estraiamo il nome dal file di configurazione
def build_scheduler(optimizer: optim.Optimizer, cfg: Dict[str,Any]):
    name = (cfg.get("name") or "").lower()#"" serve per evitare errori se non c'è scritto nulla nel nome
    if name == "steplr":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(cfg.get("step_size")),
            gamma=float(cfg.get("gamma"))
        )
    else:
        return None

#funzione principale per l'addestramento del modello
def run_training(cfg:TrainConfig):#classe che contiene tutti i parametri dell'addestramento del modello
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device {device}")

    #creiamo i dataloader per l'addestramento, la validazione e il test
    train_loader,val_loader,_ = make_dataloader(
        dataset_name = cfg.dataset_name,
        data_dir= cfg.data_dir,
        mean = cfg.mean,
        std = cfg.std,
        batch_size = cfg.batch_size,
        num_workers= cfg.num_workes,
        val_split = cfg.val_split,
        download = cfg.download,
        pin_memory = cfg.pin_memory,
        device = device.type



    )

    #importiamo il modello CNN
    model = CNN(
        dropout = cfg.dropout,
        use_batchnorm = cfg.in_channels,#use_batchnorm serve per normalizzare i dati
        in_channels = cfg.in_channels,#in_channels sono i canali di input (1 per bianco e nero, 3 per RGB)
        num_classes = cfg.num_classes#num_classes sono le classi di output
    ).to(device)#sposta il modello sulla GPU se disponibile

    criterior = nn.CrossEntropyLoss()
    otpimizer = optim.Adam(model.parameters(),
                           lr = cfg.lr,
                           weight_decay = cfg.weight_decay,
                           )
    #scheduler per variare il learning rate durante l'addestramento
    scheduler = build_scheduler(otpimizer,cfg.scheduler)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    best_path = os.path.join(cfg.checkpoint_dir,cfg.best_ckpt_name)

    for epoch in range(1,cfg.epochs +1):
        train_one_epoch
        evaluate

    save_checkpoint()

    return best_path

    