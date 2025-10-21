import argparse, yaml, torch
from pathlib import Path
from torch import nn
from train.trained import TrainConfig, run_training, evaluate as eval_fn
from data.processing_data import make_dataloaders
from model.cnn import CNN, count_params
from utils.io import load_checkpoint
from predict.inference import predict_image

PATH_CONFIG = 'config/configs.yaml'

def load_config() -> dict:
    with open(PATH_CONFIG, "r") as f:
        return yaml.safe_load(f)

def cmd_train(args: argparse.Namespace) -> None:
    cfg = load_config()
    train_cfg = TrainConfig(
        data_dir=cfg["paths"]["data_dir"],
        checkpoints_dir=cfg["paths"]["checkpoints_dir"],
        best_ckpt_name=cfg["paths"]["best_ckpt_name"],
        dataset_name = cfg['data']['dataset_name'],
        mean=cfg["data"]['mean'],
        std=cfg["data"]["std"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        val_split=cfg["data"]["val_split"],
        download=cfg["data"]["download"],
        pin_memory=cfg["data"].get("pin_memory", True),
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"],
        p_dropout=float(cfg["model"]["p_dropout"]),
        use_batchnorm=bool(cfg["model"]["use_batchnorm"]),
        epochs=cfg["train"]["epochs"],
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        seed=cfg["train"]["seed"],
        scheduler=cfg["train"].get("scheduler", {}),
    )
    best_path = run_training(train_cfg)
    print(f"Best checkpoint saved at: {best_path}")

def cmd_eval(args: argparse.Namespace) -> None:
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = make_dataloaders(
        dataset_name = cfg['data']['dataset_name'],
        data_dir=cfg["paths"]["data_dir"],
        mean= cfg["data"]["mean"],
        std = cfg["data"]["std"],
        batch_size=cfg["eval"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        val_split=cfg["data"]["val_split"],
        download=cfg["data"]["download"],
        pin_memory=cfg["data"].get("pin_memory", True),
        device_type=device.type,
    )
    model = CNN(
        p_dropout=float(cfg["model"]["p_dropout"]),
        use_batchnorm=bool(cfg["model"]["use_batchnorm"]),
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"],
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    ckpt_file = Path(cfg["paths"]["checkpoints_dir"]) / cfg["paths"]["best_ckpt_name"]
    load_checkpoint(model, str(ckpt_file), map_location=str(device))
    test_loss, test_acc = eval_fn(model, test_loader, criterion, device, phase="Test")
    print(f"[Test] loss={test_loss:.4f} acc={test_acc:.4f} | params={count_params(model)}")

def cmd_infer(args: argparse.Namespace) -> None:
    cfg = load_config()
    pred = predict_image(
        image_path=args.image,
        ckpt_path= Path(cfg["paths"]["checkpoints_dir"]) / cfg["paths"]["best_ckpt_name"],
        p_dropout=float(cfg["model"]["p_dropout"]),
        use_batchnorm=bool(cfg["model"]["use_batchnorm"]),
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"],
    )
    print(f"Prediction: {pred}")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MNIST SimpleCNN Orchestrator")
    sub = p.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train SimpleCNN and save best checkpoint")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", help="Evaluate a checkpoint on the test set")
    p_eval.set_defaults(func=cmd_eval)

    p_infer = sub.add_parser("infer", help="Run inference on a single image")
    p_infer.add_argument("--image", type=str, required=True, help="Path to input image")
    p_infer.set_defaults(func=cmd_infer)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()


