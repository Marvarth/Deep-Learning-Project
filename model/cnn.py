import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, p_dropout: float = 0.0, use_batchnorm: bool = False, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        c1, c2 = 5, 10
        layers = []
        layers += [nn.Conv2d(in_channels, c1, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        if use_batchnorm:
            layers += [nn.BatchNorm2d(c1)]
        layers += [nn.MaxPool2d(2, 2)]  # 28 -> 14
        layers += [nn.Conv2d(c1, c2, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        if use_batchnorm:
            layers += [nn.BatchNorm2d(c2)]
        layers += [nn.MaxPool2d(2, 2)]  # 14 -> 7
        self.features = nn.Sequential(*layers)
        self.dropout = nn.Dropout(p_dropout) if p_dropout and p_dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(7 * 7 * c2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)          # (N, c2, 7, 7)
        x = torch.flatten(x, 1)       # (N, 7*7*c2)
        x = self.dropout(x)
        x = self.classifier(x)        # logits
        return x

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
