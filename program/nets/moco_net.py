import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F




class MoCoModel(torch.nn.Module):
    def __init__(self, base_model, projection_dim=128):
        super(MoCoModel, self).__init__()
        self.encoder = base_model
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, projection_dim)  # Projection head to output 128-dimensional vector
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return h, z  # Returning both the original and projected feature vectors


def load_moco_model(model_path, device):
    base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    base_model.fc = torch.nn.Identity()  # Removing the final classification layer
    model = MoCoModel(base_model).to(device)

    # Load the checkpoint
    # print(f"Loading MoCo checkpoint from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # Check if checkpoint has query or key model state dict and load accordingly
    if 'query_model_state_dict' in checkpoint:
        state_dict = checkpoint['query_model_state_dict']
    else:
        state_dict = checkpoint['model_state_dict']

    # If the checkpoint was saved with DataParallel, remove 'module.' prefix
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    # print("MoCo model loaded successfully")
    return model
