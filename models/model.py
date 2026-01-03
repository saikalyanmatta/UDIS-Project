import torch.nn as nn
from torchvision.models import efficientnet_b0

class DamageNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = efficientnet_b0(pretrained=True)
        self.model.classifier[1] = nn.Linear(1280, 2)  # intact / damaged

    def forward(self, x):
        return self.model(x)
