#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from thop import profile, clever_format


# Define the SimCLR model with a projection head
class SimCLRModel(nn.Module):
    def __init__(self, base_model, projection_dim=128):
        super(SimCLRModel, self).__init__()
        self.encoder = base_model
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return h, z


def main():
    # Load the ResNet50 model and remove the final classification layer
    base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    base_model.fc = nn.Identity()  # remove the classification head

    # Create the SimCLR model
    model = SimCLRModel(base_model)
    model.eval()  # set to evaluation mode

    # Create a dummy input tensor (batch size = 1, 3 channels, 224x224 image)
    dummy_input = torch.randn(1, 3, 224, 224)

    # Compute MACs and parameters using THOP
    macs, params = profile(model, inputs=(dummy_input,))

    # Format the results for readability
    macs, params = clever_format([macs, params], "%.3f")

    print(f"Network Parameters: {params}")
    print(f"MACs: {macs}")


if __name__ == "__main__":
    main()
