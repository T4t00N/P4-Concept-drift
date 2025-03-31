import os
import glob
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import wandb


# Define the NT-Xent loss for SimCLR
def nt_xent_loss(z, tau=0.5):
    """
    Compute the NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.
    z: Projections from the model (shape: [2*B, projection_dim])
    tau: Temperature parameter
    """
    B = z.size(0) // 2
    z = F.normalize(z, p=2, dim=1)
    S = torch.matmul(z, z.T) / tau
    labels = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    mask = torch.eye(2 * B, dtype=torch.bool).to(z.device)
    S_masked = S.masked_fill(mask, -1e9)
    denom = torch.logsumexp(S_masked, dim=1)
    pos_sim = S[torch.arange(2 * B), labels]
    loss = -pos_sim + denom
    return loss.mean()


class SimCLRModel(nn.Module):
    def __init__(self, base_model, projection_dim=128):
        super(SimCLRModel, self).__init__()
        self.encoder = base_model
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return h, z


# Updated Dataset for SimCLR training
class SimCLRDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform, step=1):
        """
        Initialize the dataset, selecting every 'step'-th image from image_paths.

        Args:
            image_paths (list): List of paths to image files.
            transform: Transformations to apply to each image.
            step (int): Interval for selecting images (default=1, meaning all images).
        """
        self.image_paths = image_paths[::step]  # Select every 'step'-th image
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        image = Image.open(path).convert("RGB")
        view1 = self.transform(image)
        view2 = self.transform(image)
        return view1, view2


def main():
    # Initialize WandB for the SimCLR project
    wandb.init(project="SimCLR", config={
        "learning_rate": 0.002,
        "epochs": 100,
        "batch_size": 300,
        "optimizer": "Adam",
        "model": "ResNet50 with SimCLR",
    })

    image_dir = "/ceph/project/P4-concept-drift/YOLOv8-Anton/data/cropped_images/train"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Define SimCLR augmentations
    simclr_transform = T.Compose([
        T.RandomResizedCrop(224, scale=(0.2, 1.0)),
        T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0, hue=0)], p=0.8),
        T.RandomApply([T.GaussianBlur(kernel_size=23)], p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")) +
                         glob.glob(os.path.join(image_dir, "*.jpeg")) +
                         glob.glob(os.path.join(image_dir, "*.png")))
    if not image_paths:
        raise FileNotFoundError(f"No image files found in {image_dir}")

    # Create dataset with step=10 to select every 10th image
    simclr_dataset = SimCLRDataset(image_paths, simclr_transform, step=10)
    simclr_loader = torch.utils.data.DataLoader(simclr_dataset, batch_size=wandb.config.batch_size,
                                                shuffle=True, num_workers=20)

    # Log and print the number of selected images
    print(f"Selected {len(simclr_dataset)} images for training.")

    base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    base_model.fc = nn.Identity()
    model = SimCLRModel(base_model).to(device)
    if device == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

    # SimCLR training loop
    num_epochs = wandb.config.epochs
    print("Starting SimCLR training")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for views1, views2 in tqdm(simclr_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
            views = torch.cat([views1, views2], dim=0).to(device)
            optimizer.zero_grad()
            _, z = model(views)
            loss = nt_xent_loss(z)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(simclr_loader)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

        # Log epoch loss to WandB
        wandb.log({"epoch": epoch + 1, "loss": avg_loss})

        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            checkpoint_path = f"simclr_epoch_{epoch + 1}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}.")

    wandb.finish()


if __name__ == "__main__":
    main()