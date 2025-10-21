import torch
from PIL import Image
from torchvision import transforms
from model.cnn import CNN
from utils.io import load_checkpoint

IMG_TF = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

MNIST_CLASSES = [str(i) for i in range(10)]

def predict_image(image_path: str, ckpt_path: str, p_dropout: float = 0.0, use_batchnorm: bool = False, in_channels: int = 1, num_classes: int = 10, device_str: str | None = None) -> str:
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = CNN(p_dropout=p_dropout, use_batchnorm=use_batchnorm, in_channels=in_channels, num_classes=num_classes).to(device)
    load_checkpoint(model, ckpt_path, map_location=str(device))
    model.eval()

    img = Image.open(image_path).convert("L")
    x = IMG_TF(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1).item()
    return MNIST_CLASSES[pred]
